import copy
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.structures import InstanceData
from torch import nn


@MODELS.register_module()
class MeanTeacher3DDetector(Base3DDetector):
    """
    Mean-Teacher wrapper for 3D detectors in MMDetection3D.

    Uses:
        - Student detector (trainable)
        - Teacher detector (EMA updated)
        - Supervised loss from student on labeled data
        - Pseudo-label loss on target (unlabeled) data
        - Consistency loss between teacher & student on unlabeled data
    """

    def __init__(self,
                 detector,          # base detector config (student/teacher)
                 mean_teacher_cfg=dict(
                     point_cloud_range=None,
                     ema_momentum=0.999,
                     use_bev_consistency=True,
                     tau=0.07,
                     lambda_weight=0.05,
                     voxel_size=0.2,
                     # Confidence thresholding params
                     conf_threshold=0.6,
                     use_class_specific_thresh=False,
                     class_thresholds=None,  # dict: {class_id: threshold}
                     # loss weights
                     source_loss_weight=1.0,
                     target_loss_weight=0.5,
                     contrastive_weight=1.0,
                 ),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        # Build Student and Teacher Detectors
        self.student = MODELS.build(copy.deepcopy(detector))
        self.teacher = MODELS.build(copy.deepcopy(detector))

        # Teacher never receives gradients
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        
        for p in self.student.parameters():
            p.requires_grad_(True)
        
        # Set teacher to train mode for BN params update
        self.student.train()
        self.teacher.train()
        
        # Store configs
        self.mean_teacher_cfg = mean_teacher_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Initialize teacher with student weights
        for (t_name, t_param), (s_name, s_param) in zip(
                self.teacher.named_parameters(),
                self.student.named_parameters()):
            if t_name == s_name:
                t_param.data.copy_(s_param.data)
                param_matched += 1
            else:
                print(f"⚠️ WARNING: Parameter name mismatch: {t_name} vs {s_name}")
        
        # Initialize teacher buffers (BN stats)
        for (t_name, t_buf), (s_name, s_buf) in zip(
                self.teacher.named_buffers(),
                self.student.named_buffers()):
            if t_name == s_name:
                t_buf.copy_(s_buf)
            else:
                print(f"⚠️ WARNING: Buffer name mismatch: {t_name} vs {s_name}")

        # Print configuration
        print("\n" + "-"*70)
        print("MEAN-TEACHER CONFIGURATION:")
        print("-"*70)
        print(f"  EMA Momentum: {self.mean_teacher_cfg.get('ema_momentum', 0.999)}")
        print(f"  Use BEV Consistency: {self.mean_teacher_cfg.get('use_bev_consistency', True)}")
        print(f"  Confidence Threshold: {self.mean_teacher_cfg.get('conf_threshold', 0.6)}")
        print(f"  Source Loss Weight: {self.mean_teacher_cfg.get('source_loss_weight', 1.0)}")
        print(f"  Target Loss Weight: {self.mean_teacher_cfg.get('target_loss_weight', 0.5)}")
        print(f"  Contrastive Weight: {self.mean_teacher_cfg.get('contrastive_weight', 1.0)}")
        print(f"  Voxel Size: {self.mean_teacher_cfg.get('voxel_size', 0.16)}")
        print(f"  Temperature (tau): {self.mean_teacher_cfg.get('tau', 0.07)}")

    # EMA Update — called by training hook every iter
    @torch.no_grad()
    def ema_update(self):
        """
        Update teacher parameters and buffers (BN stats) as EMA of student parameters.
        """
        # Only print every N updates (to avoid spam)
        if not hasattr(self, '_ema_update_count'):
            self._ema_update_count = 0
            self._last_param_norm = None
        
        self._ema_update_count += 1
        alpha = self.mean_teacher_cfg.get('ema_momentum', 0.999)

        # Update parameters
        param_updated = 0
        total_param_norm = 0.0
        
        for (t_name, t_param), (s_name, s_param) in zip(
                self.teacher.named_parameters(),
                self.student.named_parameters()):
            assert t_name == s_name, f"Parameter mismatch: {t_name} vs {s_name}"
            
            # Store old value for change tracking
            if self._ema_update_count == 1:
                old_norm = t_param.data.norm().item()
            
            # EMA update
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)
            diff = (s_param - t_param).abs().mean()
            print(diff)

            param_updated += 1
            total_param_norm += t_param.data.norm().item()

        avg_param_norm = None
        # Print every 50 updates
        if self._ema_update_count % 50 == 0:
            avg_param_norm = total_param_norm / max(param_updated, 1)
            
            change_indicator = ""
            if self._last_param_norm is not None:
                change = avg_param_norm - self._last_param_norm
                change_indicator = f" (Δ: {change:+.2e})"
            
            print(f"[EMA Update #{self._ema_update_count}] | "
                f"Avg param norm: {avg_param_norm:.4f}{change_indicator}")
            
            self._last_param_norm = avg_param_norm

    def extract_feat(self, model, inputs):
        """
        Implemented in the detector to extract features (including BEV features).
        """
        return model.extract_feat(inputs, return_bev=True)

    def filter_teacher_predictions(self, teacher_pred):
        """
        Filter teacher predictions based on confidence threshold.
        
        Args:
            teacher_pred: Single sample prediction dict with keys:
                - 'bboxes_3d': 3D bounding boxes
                - 'scores_3d': Confidence scores
                - 'labels_3d': Class labels
                - 'bev_features': BEV feature map (optional)
        
        Returns:
            Filtered prediction dict with only high-confidence detections
        """
        scores = teacher_pred.pred_instances_3d.scores_3d
        labels = teacher_pred.pred_instances_3d.labels_3d
        bboxes = teacher_pred.pred_instances_3d.bboxes_3d
        
        # VERIFICATION: Initial counts
        initial_count = len(scores) if scores is not None else 0
        print(f"[FILTER] Initial predictions: {initial_count}")
        
        if scores is None:
            print("⚠️  [FILTER] No scores available, returning all predictions")
            return teacher_pred
        
        # Get threshold
        conf_threshold = self.mean_teacher_cfg.get('conf_threshold', 0.6)
        use_class_specific = self.mean_teacher_cfg.get('use_class_specific_thresh', False)

        print(f"[FILTER] Confidence threshold: {conf_threshold:.2f} | "
          f"Class-specific: {use_class_specific}")
                
        # Build mask: True if confidence > threshold for that class
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if use_class_specific:
            # Apply different thresholds per class
            class_thresholds = self.mean_teacher_cfg.get('class_thresholds', {})
            print(f"[FILTER] Class thresholds: {class_thresholds}")
            
            for class_id, thresh in class_thresholds.items():
                class_mask = (labels == class_id) & (scores >= thresh)
                mask |= class_mask
                print(f"  Class {class_id}: {class_mask.sum()} kept (thresh={thresh:.2f})")
            
            # Add default threshold for classes not in class_thersholds dict
            default_mask = torch.ones_like(scores, dtype=torch.bool)
            for class_id in class_thresholds.keys():
                default_mask &= (labels != class_id)
            default_mask &= (scores >= conf_threshold)
            mask |= default_mask
            print(f"  Other classes: {default_mask.sum()} kept (default thresh={conf_threshold:.2f})")

        else:
            # Single global threshold
            mask = scores >= conf_threshold

        # VERIFICATION: Filtering results
        kept_count = mask.sum().item()
        filtered_count = initial_count - kept_count
        filter_rate = (filtered_count / initial_count * 100) if initial_count > 0 else 0
        
        print(f"✅ [FILTER] Kept: {kept_count}/{initial_count} ({filter_rate:.1f}% filtered)")
        
        if kept_count == 0:
            print("⚠️  [FILTER WARNING] No predictions passed threshold!")
        
        # Show score statistics
        if kept_count > 0:
            kept_scores = scores[mask]
            print(f"  Score range: [{kept_scores.min():.3f}, {kept_scores.max():.3f}] "
                f"(mean: {kept_scores.mean():.3f})")
        
        # Create filtered prediction (deep copy to avoid modifying original)
        filtered_pred = copy.deepcopy(teacher_pred)
        
        # Filter pred_instances_3d
        filtered_pred.pred_instances_3d.bboxes_3d = bboxes[mask]
        filtered_pred.pred_instances_3d.scores_3d = scores[mask]
        filtered_pred.pred_instances_3d.labels_3d = labels[mask]
        
        # Preserve BEV features (not filtered, it's a spatial feature map)
        if hasattr(teacher_pred, 'bev_features'):
            filtered_pred.bev_features = teacher_pred.bev_features
            print(f"  BEV features preserved: {filtered_pred.bev_features.shape}")
            
        return filtered_pred

    def class_aware_contrastive_loss(self,
        bev_s, bev_t,           # BEV feature maps: C x H x W
        boxes, labels,          # teacher bboxes + labels (N_i objects per image)
        scores=None,            # confidence scores (optional, for weighting)
        tau=0.07,               # temperature
        lambda_weight=0.05):    # balancing weight
        
        """ bev_s,t: student and teacher BEV map  [C,H,W]
            boxes: teacher 3D boxes  (N,7 or N,...)   used to compute BEV centers
            labels: predicted classes for each box (N,)

            Returns:
            Contrastive loss between student and teacher BEV features at object locations.
        """
        
        if len(boxes) == 0:
            return torch.tensor(0., device=bev_s.device)
        
        # Compute BEV centers of boxes (x,y in BEV pixel coords)
        # boxes[:, :2] = (x,y) center in meters
        # Convert to pixel indices
        bev_resolution = self.mean_teacher_cfg["voxel_size"]                    # assume square voxels in x,y
        pc_range = self.mean_teacher_cfg["point_cloud_range"]
        min_x = pc_range[0]
        min_y = pc_range[1]
        xs = (boxes[:, 0] - min_x / bev_resolution).long().clamp(0, bev_s.shape[2]-1)
        ys = (boxes[:, 1] - min_y / bev_resolution).long().clamp(0, bev_s.shape[1]-1)

        # Check for out-of-bounds (before clamping)
        xs_raw = (boxes[:, 0] - min_x / bev_resolution).long()
        ys_raw = (boxes[:, 1] - min_y / bev_resolution).long()
        out_of_bounds = ((xs_raw < 0) | (xs_raw >= bev_s.shape[2]) | 
                        (ys_raw < 0) | (ys_raw >= bev_s.shape[1])).sum()
        if out_of_bounds > 0:
            print(f"⚠️  [CONTRASTIVE WARNING] {out_of_bounds} boxes out of BEV bounds (clamped)")

        # Extract BEV features at object centers
        F_s = bev_s[:, ys, xs].T    # shape N x C
        F_t = bev_t[:, ys, xs].T    # shape N x C

        # Normalize features (optional but recommended)
        F_s = F.normalize(F_s, dim=1)
        F_t = F.normalize(F_t, dim=1)
        device = F_s.device

        # Compute similarity matrix: N x N
        sim_matrix = torch.mm(F_s, F_t.T) / tau

        # Build positive sets
        # pos_mask[i][j] = 1 if class_j == class_i
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float().to(device)
        pos_count = pos_mask.sum(dim=1)  # N

        # Compute contrastive loss
        # log_softmax over j dimension
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # For each i: average only over positive j’s
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-6)
        return lambda_weight * loss.mean()

    def _transform_boxes(self, boxes, metainfo_strong):
        """
        Transform boxes from weak to strong augmentation space.
        Assumption: NO geometric augmentations on the weak pipeline
        
        Args:
            boxes: Boxes from teacher (in weak augmentation space)
            metainfo_strong: Metadata from strong sample (heavy transforms)
        
        Returns:
            Transformed boxes for strong augmentation space
        """
    
        # Convert to tensor
        if isinstance(boxes, LiDARInstance3DBoxes):
            boxes_tensor = boxes.tensor.clone()
        else:
            boxes_tensor = boxes.clone()

        origin = metainfo_strong.get('box_origin', (0.5, 0.5, 0.5))
        device = boxes_tensor.device
        
        # Extract geometric augmentation parameters in strong set
        # stored by GlobalRotScaleTrans and RandomFlip3D
        pcd_rotation = metainfo_strong.get('pcd_rotation', 0.0)
        pcd_scale_factor = metainfo_strong.get('pcd_scale_factor', 1.0)
        pcd_trans = metainfo_strong.get('pcd_trans', torch.zeros(3, device=device))
        flip_horizontal = metainfo_strong.get('pcd_horizontal_flip', False)
         
        # Apply transformations in the same order as the pipeline
        # 1. Random horizontal flip
        if flip_horizontal:
            boxes_tensor[:, 1] = -boxes_tensor[:, 1]  # Flip y coordinate
            boxes_tensor[:, 6] = -boxes_tensor[:, 6]  # Flip yaw angle
        
        # 2. Global rotation
        if abs(pcd_rotation) > 1e-6:
            cos_r = torch.cos(torch.tensor(pcd_rotation, device=device))
            sin_r = torch.sin(torch.tensor(pcd_rotation, device=device))
            
            # Rotate centers
            x_rot = cos_r * boxes_tensor[:, 0] - sin_r * boxes_tensor[:, 1]
            y_rot = sin_r * boxes_tensor[:, 0] + cos_r * boxes_tensor[:, 1]
            boxes_tensor[:, 0] = x_rot
            boxes_tensor[:, 1] = y_rot
            
            # Rotate yaw
            boxes_tensor[:, 6] += pcd_rotation
        
        # 3. Global scaling
        if abs(pcd_scale_factor - 1.0) > 1e-6:
            # Scale positions and dimensions
            boxes_tensor[:, :3] *= pcd_scale_factor
            boxes_tensor[:, 3:6] *= pcd_scale_factor
        
        # 4. Global translation
        if isinstance(pcd_trans, (list, tuple)):
            pcd_trans = torch.tensor(pcd_trans, device=device)
        if torch.abs(pcd_trans).sum() > 1e-6:
            boxes_tensor[:, :3] += pcd_trans
        
        # Normalize yaw to [-π, π]
        boxes_tensor[:, 6] = torch.atan2(
            torch.sin(boxes_tensor[:, 6]),
            torch.cos(boxes_tensor[:, 6])
        )
        
        # Create transformed box object
        transformed_boxes = LiDARInstance3DBoxes(
            boxes_tensor,
            box_dim=boxes_tensor.shape[-1],
            origin=origin
        )

    def _create_pseudo_labels(self, teacher_predictions, target_samples_weak, target_samples_strong):
        """
        Create pseudo-labeled data samples from teacher predictions.
        
        Args:
            teacher_predictions: List of filtered teacher predictions
            target_samples_weak: Weak augmentation samples (where teacher predicted)
            target_samples_strong: Strong augmentation samples (where student learns)
            
        Returns:
            pseudo_labeled_samples: Strong aug samples with transformed pseudo-labels as ground truth
        """
       
        pseudo_labeled_samples = copy.deepcopy(target_samples_strong)
        
        total_boxes = 0

        # Replace ground truth with teacher's pseudo-labels
        for i, (pred, sample_weak, sample_strong) in enumerate(
            zip(teacher_predictions, target_samples_weak, pseudo_labeled_samples)):
            
           # Teacher predictions are in weakly augmented space, while student trains on strongly augmented space
            # Get boxes from teacher prediction (in weakly augmented space)
            pred_instances = pred.pred_instances_3d
            boxes = pred_instances.bboxes_3d
            labels = pred_instances.labels_3d
            scores = pred_instances.scores_3d if hasattr(pred_instances, 'scores_3d') else None
            total_boxes += len(boxes)
            
            # Initialize empty ground truth
            gt_instances = InstanceData()

            if len(boxes) == 0:
                # Set empty but valid ground truth container
                gt_instances.bboxes_3d = boxes
                gt_instances.labels_3d = labels
                gt_instances.scores_3d = scores if scores is not None else None
                
                # Assign all three at once
                sample_strong.gt_instances_3d = gt_instances
                continue

            # Transform boxes to strong augmentation space
            boxes_transformed = self._transform_boxes(
                boxes,
                sample_strong.metainfo)

            # Assign transformed pseudo-labels
            gt_instances.bboxes_3d = boxes_transformed
            gt_instances.labels_3d = labels
            gt_instances.scores_3d = scores if scores is not None else None
            
            sample_strong.gt_instances_3d = gt_instances
        
        if total_boxes == 0:
            print("⚠️  [PSEUDO-LABELS WARNING] No pseudo-labels created (all filtered out?)")
        else: None

        return pseudo_labeled_samples
    
    def loss(self, batch_inputs_dict, batch_data_samples):
        """
        Compute total loss with three components:
        1. Supervised loss on source/labeled data (regression + classification)
        2. Pseudo-label loss on target/unlabeled data (regression + classification)
        3. Contrastive consistency loss on target data (BEV features)
        """
       
        # Load weights from config
        w_source = self.mean_teacher_cfg.get('source_loss_weight', 1.0)
        w_target = self.mean_teacher_cfg.get('target_loss_weight', 1.0)
        w_cont = self.mean_teacher_cfg.get('contrastive_weight', 0.1)
                
        device = next(self.student.parameters()).device
        losses = {}
        
        # ========== TERM 1: Source Loss ==========     
        # labeled data from source domain  
        source_inputs = batch_inputs_dict["labeled"]
        source_samples = batch_data_samples["labeled"]
            
        if w_source > 0:
            try:
                loss_source = self.student.loss(source_inputs, source_samples)
                
                for key, value in loss_source.items():
                    # Handle both tensor and list/tuple values
                    if isinstance(value, (list, tuple)):
                        # If it's a list/tuple of losses, sum them
                        if len(value) > 0 and isinstance(value[0], torch.Tensor):
                            value = sum(value)
                        else:
                            continue
                    
                    losses[f'{key}_source'] = value * w_source
            except RuntimeError as e:
                print(f"  ERROR in student.loss(): {e}")            
                raise e
        else: None
        
        # ========== Teacher Prediction ==========  
        target_weak = batch_inputs_dict["unlabeled"]["weak"]
        target_strong = batch_inputs_dict["unlabeled"]["strong"]
        target_samples_weak = batch_data_samples["unlabeled"]["weak"]
        target_samples_strong = batch_data_samples["unlabeled"]["strong"]
               
        # Teacher forward pass (no grad)
        print("[DEBUG] Teacher forward START")
        self.teacher.train()
        with torch.no_grad():
            teacher_pred = self.teacher.predict(
                target_weak,
                target_samples_weak,
                return_bev_features=True
            )
        print("[DEBUG] Teacher forward END")
        
        # Filter teacher predictions
        filtered_teacher_preds = []
        for i in range(len(teacher_pred)):
            filtered_pred = self.filter_teacher_predictions(teacher_pred[i])
            filtered_teacher_preds.append(filtered_pred)
        
        # ========== TERM 2: Pseudo-Label Loss ==========
        # Add psuedo-labels to strong augmentation samples
        pseudo_labeled_samples = self._create_pseudo_labels(
            filtered_teacher_preds,
            target_samples_weak,
            target_samples_strong)
        
        if w_target > 0:
            loss_target = self.student.loss(target_strong, pseudo_labeled_samples)

            for key, value in loss_target.items():
                # Handle both tensor and list/tuple values
                if isinstance(value, (list, tuple)):
                    losses[f'{key}_target'] = [v * w_target for v in value]
                else:
                    losses[f'{key}_target'] = value * w_target
        else: None
        
        # ========== TERM 3: Contrastive Loss ==========      
        use_bev = self.mean_teacher_cfg.get("use_bev_consistency", False)

        if use_bev and w_cont > 0:
            # Student forward pass on strong augmentation
            student_pred = self.student.predict(
                target_strong,
                target_samples_strong,
                return_bev_features=True
            )
            
            loss_contrastive_total = torch.tensor(0., device=device)
            num_valid_samples = 0
            
            for i in range(len(student_pred)):
                try:
                    # Extract BEV features
                    bev_s = student_pred[i].bev_features
                    bev_t = filtered_teacher_preds[i].bev_features
                    
                    if bev_s is None or bev_t is None:
                        print("⚠️  [CONTRASTIVE] BEV features not available, skipping")
                        continue
                    
                    boxes = filtered_teacher_preds[i].pred_instances_3d.bboxes_3d.tensor
                    labels = filtered_teacher_preds[i].pred_instances_3d.labels_3d
                    
                    if hasattr(filtered_teacher_preds[i].pred_instances_3d, 'scores_3d'):

                        scores = filtered_teacher_preds[i].pred_instances_3d.scores_3d
                    else: None
                    
                    if len(boxes) == 0:
                        continue
                    
                    loss_i = self.class_aware_contrastive_loss(
                        bev_s, bev_t, boxes, labels,
                        scores=scores,
                        tau=self.mean_teacher_cfg.get("tau", 0.07),
                        lambda_weight=self.mean_teacher_cfg.get("lambda_weight", 0.05)
                    )
                    
                    loss_contrastive_total += loss_i
                    num_valid_samples += 1
                    
                except (KeyError, AttributeError) as e:
                    print(f"⚠️  [CONTRASTIVE] Error for sample {i}: {e}")
                    continue
            
            if num_valid_samples > 0:
                avg_contrastive = loss_contrastive_total / num_valid_samples
                losses['loss_contrastive'] = avg_contrastive * w_cont
            else:
                losses['loss_contrastive'] = torch.tensor(0., device=device)
        else:
            losses['loss_contrastive'] = torch.tensor(0., device=device)
        
        # ========== Summary ==========
        # Sum all source losses
        source_total = sum([v for k, v in losses.items() if '_source' in k and not k.startswith('_')])
        # Gather all target losses as tensors
        target_tensors = []
        for k, v in losses.items():
            if '_target' in k and not k.startswith('_'):
                if isinstance(v, (list, tuple)):
                    target_tensors.extend(v)  # flatten the list of tensors
                else:
                    target_tensors.append(v)

        # Sum everything into a single scalar tensor
        target_total = sum(target_tensors)
        contrastive_total = losses.get('loss_contrastive', torch.tensor(0., device=device))
        
        total_loss = source_total + target_total + contrastive_total
        
        # Add metadata for logging
        losses['_source_total'] = source_total
        print(f"source_total: {source_total.item():.4f}")
        losses['_target_total'] = target_total
        print(f"target_total: {target_total.item():.4f}")
        print(f"contrastive_total: {contrastive_total.item():.4f}")
        losses['_num_pseudo_labels'] = torch.tensor(num_valid_samples, device=device)
        losses['_total_loss'] = total_loss
        print(f"TOTAL LOSS: {total_loss.item():.4f}\n")    
    
        return losses

    def predict(self, batch_inputs, batch_data_samples,
                use_teacher=False, **kwargs):
        if use_teacher:
            return self.teacher.predict(batch_inputs, batch_data_samples, **kwargs)
        else:
            return self.student.predict(batch_inputs, batch_data_samples, **kwargs)