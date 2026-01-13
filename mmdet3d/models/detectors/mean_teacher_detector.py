import copy
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.structures import InstanceData


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
                     ema_momentum=0.999,
                     use_bev_consistency=True,
                     tau=0.07,
                     lambda_weight=0.05,
                     voxel_size=0.16,
                     # Confidence thresholding params
                     conf_threshold=0.6,
                     use_class_specific_thresh=False,
                     class_thresholds=None,  # dict: {class_id: threshold}
                     # loss weights
                     source_loss_weight=1.0,      # Weight for labeled data loss
                     target_loss_weight=0.5,      # Weight for pseudo-label loss on unlabeled data
                     contrastive_weight=1.0,      # Weight for BEV contrastive loss (prev. lambda_weight)
                 ),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        print("\n" + "="*70)
        print("INITIALIZING MEAN-TEACHER 3D DETECTOR")
        print("="*70)

        # Build Student and Teacher Detectors
        print("\n[1/5] Building student detector...")
        self.student = MODELS.build(detector)
        print(f"✅ Student detector built: {type(self.student).__name__}")
        
        print("\n[2/5] Building teacher detector...")
        self.teacher = MODELS.build(detector)
        print(f"✅ Teacher detector built: {type(self.teacher).__name__}")

        # Teacher never receives gradients
        print("\n[3/5] Disabling teacher gradients...")
        param_count = 0
        for p in self.teacher.parameters():
            p.requires_grad_(False)
            param_count += 1
        print(f"✅ Disabled gradients for {param_count} teacher parameters")

        # Lock the teacher in eval mode
        self.teacher.eval()
        print("✅ Teacher set to eval mode")

        self.mean_teacher_cfg = mean_teacher_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Initialize teacher with student weights
        print("\n[4/5] Copying student weights to teacher...")
        param_matched = 0
        param_mismatched = 0
        
        for (t_name, t_param), (s_name, s_param) in zip(
                self.teacher.named_parameters(),
                self.student.named_parameters()):
            if t_name == s_name:
                t_param.data.copy_(s_param.data)
                param_matched += 1
            else:
                print(f"⚠️ WARNING: Parameter name mismatch: {t_name} vs {s_name}")
                param_mismatched += 1
        
        print(f"✅ Copied {param_matched} parameters (mismatched: {param_mismatched})")
        
        # Initialize teacher buffers (BN stats)
        print("\n[5/5] Copying BatchNorm buffers to teacher...")
        buffer_matched = 0
        buffer_mismatched = 0
        
        for (t_name, t_buf), (s_name, s_buf) in zip(
                self.teacher.named_buffers(),
                self.student.named_buffers()):
            if t_name == s_name:
                t_buf.copy_(s_buf)
                buffer_matched += 1
            else:
                print(f"⚠️ WARNING: Buffer name mismatch: {t_name} vs {s_name}")
                buffer_mismatched += 1
        
        print(f"✅ Copied {buffer_matched} buffers (mismatched: {buffer_mismatched})")

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
        
        # Verify teacher is in eval mode
        is_training = self.teacher.training
        print("\n" + "-"*70)
        print(f"Teacher mode: {'TRAINING ⚠️' if is_training else 'EVAL ✅'}")
        print(f"Student mode: {'TRAINING ✅' if self.student.training else 'EVAL ⚠️'}")
        
        print("\n" + "="*70)
        print("MEAN-TEACHER INITIALIZATION COMPLETE")
        print("="*70 + "\n")

    # EMA Update — called by training hook every iteration
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
            
            param_updated += 1
            total_param_norm += t_param.data.norm().item()
        
        # Update buffers (BatchNorm stats)
        buffer_updated = 0
        for (t_name, t_buf), (s_name, s_buf) in zip(
                self.teacher.named_buffers(),
                self.student.named_buffers()):
            assert t_name == s_name, f"Buffer mismatch: {t_name} vs {s_name}"
            t_buf.copy_(s_buf)
            buffer_updated += 1
        
        avg_param_norm = None

        # Print every 50 updates
        if self._ema_update_count % 50 == 0:
            avg_param_norm = total_param_norm / max(param_updated, 1)
            
            change_indicator = ""
            if self._last_param_norm is not None:
                change = avg_param_norm - self._last_param_norm
                change_indicator = f" (Δ: {change:+.2e})"
            
            print(f"[EMA Update #{self._ema_update_count}] "
                f"Updated {param_updated} params, {buffer_updated} buffers | "
                f"Avg param norm: {avg_param_norm:.4f}{change_indicator}")
            
            self._last_param_norm = avg_param_norm
        
        # Print first update for verification
        if self._ema_update_count == 1:
            print("\n" + "="*70)
            print("FIRST EMA UPDATE")
            print("="*70)
            print(f"✅ Updated {param_updated} parameters with α={alpha}")
            print(f"✅ Updated {buffer_updated} buffers")
            # Only print if avg_param_norm was calculated
            if avg_param_norm is not None:
                print(f"   Average parameter norm: {avg_param_norm:.4f}")
            print("="*70 + "\n")

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
            labels: predicted classes for each box (N,) """
        
        print(f"\n[CONTRASTIVE] Computing loss for {len(boxes)} boxes")
        
        if len(boxes) == 0:
            print("⚠️  [CONTRASTIVE] No boxes, returning zero loss")
            return torch.tensor(0., device=bev_s.device)
        
        # VERIFICATION: Input shapes
        print(f"  BEV shapes - Student: {bev_s.shape}, Teacher: {bev_t.shape}")
        if bev_s.shape != bev_t.shape:
            print(f"⚠️  [CONTRASTIVE WARNING] BEV shape mismatch!")

        # Compute BEV centers of boxes (x,y in BEV pixel coords)
        # boxes[:, :2] = (x,y) center in meters
        # Convert to pixel indices
        bev_resolution = self.mean_teacher_cfg["voxel_size"]                    # assume square voxels in x,y
        pc_range = self.student.data_preprocessor.voxel_layer.get('point_cloud_range', None)
        min_x = pc_range[0]
        min_y = pc_range[1]
        xs = (boxes[:, 0] - min_x / bev_resolution).long().clamp(0, bev_s.shape[2]-1)
        ys = (boxes[:, 1] - min_y / bev_resolution).long().clamp(0, bev_s.shape[1]-1)

        # VERIFICATION: Box locations
        print(f"  Box centers in BEV: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")

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
        print(f"  Feature shapes - Student: {F_s.shape}, Teacher: {F_t.shape}")

        # Normalize features (optional but recommended)
        F_s = F.normalize(F_s, dim=1)
        F_t = F.normalize(F_t, dim=1)
        device = F_s.device

        # Compute similarity matrix: N x N
        sim_matrix = torch.mm(F_s, F_t.T) / tau
        print(f"  Similarity matrix: {sim_matrix.shape}, range=[{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")

        # Build positive sets
        # pos_mask[i][j] = 1 if class_j == class_i
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float().to(device)
        pos_count = pos_mask.sum(dim=1)  # N

        # VERIFICATION: Class distribution
        unique_labels, label_counts = torch.unique(labels, return_counts=True)
        print(f"  Class distribution: {dict(zip(unique_labels.tolist(), label_counts.tolist()))}")
        print(f"  Positive pairs per sample: min={pos_count.min():.0f}, "
            f"max={pos_count.max():.0f}, mean={pos_count.mean():.1f}")
        
        if pos_count.min() == 1:
            print("⚠️  [CONTRASTIVE WARNING] Some classes have only 1 instance "
                "(no positive pairs)")
            
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
        print(f"\n[TRANSFORM] Transforming {len(boxes)} boxes")
    
        # Convert to tensor
        if isinstance(boxes, LiDARInstance3DBoxes):
            boxes_tensor = boxes.tensor.clone()
        else:
            boxes_tensor = boxes.clone()

        origin = metainfo_strong.get('box_origin', (0.5, 0.5, 0.5))
        device = boxes_tensor.device
        
        # VERIFICATION: Initial box statistics
        print(f"  Initial boxes - Center range: "
            f"x=[{boxes_tensor[:, 0].min():.2f}, {boxes_tensor[:, 0].max():.2f}], "
            f"y=[{boxes_tensor[:, 1].min():.2f}, {boxes_tensor[:, 1].max():.2f}]")
        print(f"  Initial yaw range: [{boxes_tensor[:, 6].min():.3f}, {boxes_tensor[:, 6].max():.3f}]")
   

        # Extract strong augmentation parameters
        # These are stored by GlobalRotScaleTrans and RandomFlip3D
        pcd_rotation = metainfo_strong.get('pcd_rotation', 0.0)
        pcd_scale_factor = metainfo_strong.get('pcd_scale_factor', 1.0)
        pcd_trans = metainfo_strong.get('pcd_trans', torch.zeros(3, device=device))
        flip_horizontal = metainfo_strong.get('pcd_horizontal_flip', False)
        
        # VERIFICATION: Show transformations
        import numpy as np
        print(f"  Augmentations applied:")
        print(f"    - Horizontal flip: {flip_horizontal}")
        print(f"    - Rotation: {pcd_rotation:.3f} rad ({np.rad2deg(pcd_rotation):.1f}°)")
        print(f"    - Scale: {pcd_scale_factor:.3f}")
        print(f"    - Translation: {pcd_trans}")
        
        has_transforms = (flip_horizontal or abs(pcd_rotation) > 1e-6 or 
                        abs(pcd_scale_factor - 1.0) > 1e-6 or 
                        torch.abs(pcd_trans).sum() > 1e-6)
        
        if not has_transforms:
            print("⚠️  [TRANSFORM WARNING] No augmentations detected in strong pipeline!")
 
        # Apply transformations in the same order as the pipeline
        # 1. Random horizontal flip
        if flip_horizontal:
            boxes_tensor[:, 1] = -boxes_tensor[:, 1]  # Flip y coordinate
            boxes_tensor[:, 6] = -boxes_tensor[:, 6]  # Flip yaw angle
            print(f"    ✓ Flip applied")
        
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
            print(f"    ✓ Rotation applied")
        
        # 3. Global scaling
        if abs(pcd_scale_factor - 1.0) > 1e-6:
            # Scale positions and dimensions
            boxes_tensor[:, :3] *= pcd_scale_factor
            boxes_tensor[:, 3:6] *= pcd_scale_factor
            print(f"    ✓ Scaling applied")
        
        # 4. Global translation
        if isinstance(pcd_trans, (list, tuple)):
            pcd_trans = torch.tensor(pcd_trans, device=device)
        if torch.abs(pcd_trans).sum() > 1e-6:
            boxes_tensor[:, :3] += pcd_trans
            print(f"    ✓ Translation applied")
        
        # Normalize yaw to [-π, π]
        boxes_tensor[:, 6] = torch.atan2(
            torch.sin(boxes_tensor[:, 6]),
            torch.cos(boxes_tensor[:, 6])
        )
        
        # VERIFICATION: Final box statistics
        print(f"  Final boxes - Center range: "
            f"x=[{boxes_tensor[:, 0].min():.2f}, {boxes_tensor[:, 0].max():.2f}], "
            f"y=[{boxes_tensor[:, 1].min():.2f}, {boxes_tensor[:, 1].max():.2f}]")
        print(f"  Final yaw range: [{boxes_tensor[:, 6].min():.3f}, {boxes_tensor[:, 6].max():.3f}]")

        # Create transformed box object
        transformed_boxes = LiDARInstance3DBoxes(
            boxes_tensor,
            box_dim=boxes_tensor.shape[-1],
            origin=origin
        )
        print(f"✅ [TRANSFORM] Transformation complete")
        return transformed_boxes
    

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
        print(f"\n{'='*70}")
        print(f"[PSEUDO-LABELS] Creating pseudo-labels for {len(teacher_predictions)} samples")
        print(f"{'='*70}")
        
        pseudo_labeled_samples = copy.deepcopy(target_samples_strong)
        
        total_boxes = 0
        total_transformed = 0

        # Replace ground truth with teacher's pseudo-labels
        for i, (pred, sample_weak, sample_strong) in enumerate(
            zip(teacher_predictions, target_samples_weak, pseudo_labeled_samples)):
            
            print(f"\n[PSEUDO-LABELS] Sample {i+1}/{len(teacher_predictions)}")

            # Teacher predictions are in weakly augmented space, while student trains on strongly augmented space
            # Get boxes from teacher prediction (in weakly augmented space)
            pred_instances = pred.pred_instances_3d
            boxes = pred_instances.bboxes_3d
            labels = pred_instances.labels_3d
            
            print(f"  Teacher predictions: {len(boxes)} boxes")
            total_boxes += len(boxes)
            
            # Initialize empty ground truth
            gt_instances = InstanceData()

            if len(boxes) == 0:
                print("⚠️  [PSEUDO-LABELS WARNING] No boxes for this sample")
                # Set empty but valid ground truth container
                gt_instances.bboxes_3d = boxes
                gt_instances.labels_3d = labels

                if hasattr(pred_instances, 'scores_3d'):
                    gt_instances.scores_3d = pred_instances.scores_3d
                
                # Assign all three at once
                sample_strong.gt_instances_3d = gt_instances
                continue

            # Transform boxes to strong augmentation space
            boxes_transformed = self._transform_boxes(
                boxes,
                sample_strong.metainfo)
            
            total_transformed += len(boxes_transformed)

            # Assign transformed pseudo-labels
            gt_instances.bboxes_3d = boxes_transformed
            gt_instances.labels_3d = labels

            if hasattr(pred_instances, 'scores_3d'):
                scores = pred_instances.scores_3d
                gt_instances.scores_3d = scores

                print(f"  Confidence scores: mean={scores.mean():.3f}, "
                    f"min={scores.min():.3f}, max={scores.max():.3f}")

            # Assign all at once
            sample_strong.gt_instances_3d = gt_instances
                
            print(f"✅ [PSEUDO-LABELS] Sample {i+1} complete: {len(boxes_transformed)} pseudo-labels")

        # VERIFICATION: Summary
        print(f"\n{'='*70}")
        print(f"[PSEUDO-LABELS] Summary:")
        print(f"  Total boxes processed: {total_boxes}")
        print(f"  Total boxes transformed: {total_transformed}")
        print(f"  Average per sample: {total_boxes/len(teacher_predictions):.1f}")
        
        if total_boxes == 0:
            print("⚠️  [PSEUDO-LABELS WARNING] No pseudo-labels created (all filtered out?)")
        else:
            print(f"✅ [PSEUDO-LABELS] Pseudo-labeling complete")
        print(f"{'='*70}\n")

        return pseudo_labeled_samples
    
    def loss(self, batch_inputs_dict, batch_data_samples):
        """
        Compute total loss with three components:
        1. Supervised loss on source/labeled data (regression + classification)
        2. Pseudo-label loss on target/unlabeled data (regression + classification)
        3. Contrastive consistency loss on target data (BEV features)
        """
        print(f"\n{'='*80}")
        print(f"[LOSS] Computing Mean-Teacher Loss")
        print(f"{'='*80}")
        
        # Load weights from config
        w_source = self.mean_teacher_cfg.get('source_loss_weight', 1.0)
        w_target = self.mean_teacher_cfg.get('target_loss_weight', 1.0)
        w_cont = self.mean_teacher_cfg.get('contrastive_weight', 0.1)
        
        print(f"[LOSS] Loss weights: Source={w_source:.2f}, Target={w_target:.2f}, "
            f"Contrastive={w_cont:.2f}")
        
        device = next(self.student.parameters()).device
        losses = {}
        
        # ========== TERM 1: Source Loss ==========
        print(f"\n{'─'*80}")
        print(f"[LOSS] TERM 1: Supervised Loss on Source Data")
        print(f"{'─'*80}")
        
        source_inputs = batch_inputs_dict["labeled"]
        source_samples = batch_data_samples["labeled"]
        
        print(f"  Source batch size: {len(source_samples)}")
        print(f"  Source input keys: {source_inputs.keys()}")
        
        # CRITICAL: Preprocess source data through student's data_preprocessor
        print(f"  Preprocessing source data...")
        print(f"  Input source_samples type: {type(source_samples)}, length: {len(source_samples)}")
        
        # data_preprocessor expects data dict with 'inputs' AND 'data_samples' keys
        result = self.student.data_preprocessor({
            'inputs': source_inputs,
            'data_samples': source_samples  # Include in the data dict!
        })
        
        print(f"  Data preprocessor returned: {type(result)}")
        
        # Extract processed data
        if isinstance(result, dict):
            source_inputs_processed = result.get('inputs', result)
            source_samples_processed = result.get('data_samples', source_samples)
        elif isinstance(result, tuple) and len(result) == 2:
            source_inputs_processed, source_samples_processed = result
        else:
            source_inputs_processed = result
            source_samples_processed = source_samples
        
        # Verify
        if source_samples_processed is None:
            raise ValueError("data_samples is None after preprocessing!")
        
        print(f"  ✓ Source data preprocessed (samples: {len(source_samples_processed)})")
        
        if w_source > 0:
            print(f"  Computing student loss on labeled data...")
            try:
                loss_source = self.student.loss(source_inputs_processed, source_samples_processed)
                print(f"  Source loss components:")
                for key, value in loss_source.items():
                    # Handle both tensor and list/tuple values
                    if isinstance(value, (list, tuple)):
                        # If it's a list/tuple of losses, sum them
                        if len(value) > 0 and isinstance(value[0], torch.Tensor):
                            value = sum(value)
                        else:
                            print(f"    WARNING: Skipping {key} - unexpected type: {type(value)}")
                            continue
                    
                    losses[f'{key}_source'] = value * w_source
                    print(f"    {key}: {value.item():.6f} (weighted: {(value * w_source).item():.6f})")
                print(f"✅ [LOSS] Source loss computed")
            except RuntimeError as e:
                print(f"  ERROR in student.loss(): {e}")
                print(f"  Debugging feature extraction...")
                
                # Debug: Check what shapes are produced
                try:
                    with torch.no_grad():
                        voxel_dict = source_inputs_processed['voxels']
                        print(f"  Voxel dict keys: {voxel_dict.keys()}")
                        print(f"  Voxels shape: {voxel_dict['voxels'].shape}")
                        print(f"  Coors shape: {voxel_dict['coors'].shape}")
                        
                        # Try extract_feat to see where it fails
                        print(f"  Testing extract_feat...")
                        from mmdet3d.models.detectors.voxelnet_bev import VoxelNetWithBEV
                        
                        # Check middle encoder output
                        voxel_features = self.student.voxel_encoder(voxel_dict['voxels'], 
                                                                    voxel_dict['num_points'],
                                                                    voxel_dict['coors'])
                        print(f"  Voxel features shape: {voxel_features.shape}")
                        
                        batch_size = voxel_dict['coors'][:, 0].max().item() + 1
                        x = self.student.middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
                        print(f"  Middle encoder output shape: {x.shape}")
                        
                        x = self.student.backbone(x)
                        print(f"  Backbone outputs: {[xi.shape for xi in x]}")
                        
                        # This is where it likely fails
                        x = self.student.neck(x)
                        print(f"  Neck output shape: {x.shape}")
                        
                except Exception as debug_e:
                    print(f"  Debug extraction failed: {debug_e}")
                    import traceback
                    traceback.print_exc()
                
                raise e
        else:
            print("⚠️  [LOSS] Source loss weight = 0, skipping")
        
        # ========== Teacher Prediction ==========
        print(f"\n{'─'*80}")
        print(f"[LOSS] Teacher Prediction on Unlabeled Data (Weak Aug)")
        print(f"{'─'*80}")
        
        target_weak = batch_inputs_dict["unlabeled"]["weak"]
        target_strong = batch_inputs_dict["unlabeled"]["strong"]
        target_samples_weak = batch_data_samples["unlabeled"]["weak"]
        target_samples_strong = batch_data_samples["unlabeled"]["strong"]
        
        print(f"  Target batch size: {len(target_samples_weak)}")
        print(f"  Weak aug keys: {target_weak.keys()}")
        print(f"  Strong aug keys: {target_strong.keys()}")
        
        # CRITICAL: Preprocess weak augmentation data
        print(f"  Preprocessing weak augmentation data...")
        result = self.student.data_preprocessor({
            'inputs': target_weak,
            'data_samples': target_samples_weak
        })
        if isinstance(result, dict):
            target_weak_processed = result.get('inputs', result)
            target_samples_weak_processed = result.get('data_samples', target_samples_weak)
        elif isinstance(result, tuple):
            target_weak_processed, target_samples_weak_processed = result
        else:
            target_weak_processed = result
            target_samples_weak_processed = target_samples_weak
        print(f"  ✓ Weak aug data preprocessed")
        
        with torch.no_grad():
            print(f"  Running teacher forward pass...")
            teacher_pred = self.teacher.predict(
                target_weak_processed,
                target_samples_weak_processed,
                return_bev_features=True
            )
            print(f"  Teacher predictions: {len(teacher_pred)} samples")
        
        # Filter predictions
        print(f"\n  Filtering teacher predictions...")
        filtered_teacher_preds = []
        for i in range(len(teacher_pred)):
            print(f"\n  ─── Sample {i+1}/{len(teacher_pred)} ───")
            filtered_pred = self.filter_teacher_predictions(teacher_pred[i])
            filtered_teacher_preds.append(filtered_pred)
        
        # ========== TERM 2: Pseudo-Label Loss ==========
        print(f"\n{'─'*80}")
        print(f"[LOSS] TERM 2: Pseudo-Label Loss on Target Data")
        print(f"{'─'*80}")
        
        # CRITICAL: Preprocess strong augmentation data
        print(f"  Preprocessing strong augmentation data...")
        result = self.student.data_preprocessor({
            'inputs': target_strong,
            'data_samples': target_samples_strong
        })
        if isinstance(result, dict):
            target_strong_processed = result.get('inputs', result)
            target_samples_strong_processed = result.get('data_samples', target_samples_strong)
        elif isinstance(result, tuple):
            target_strong_processed, target_samples_strong_processed = result
        else:
            target_strong_processed = result
            target_samples_strong_processed = target_samples_strong
        print(f"  ✓ Strong aug data preprocessed")
        
        pseudo_labeled_samples = self._create_pseudo_labels(
            filtered_teacher_preds,
            target_samples_weak_processed,  # Use processed samples
            target_samples_strong_processed
        )
        
        if w_target > 0:
            print(f"\n  Computing student loss on pseudo-labels...")

            loss_target = self.student.loss(target_strong_processed, pseudo_labeled_samples)

            print(f"  Target loss components:")
            for key, value in loss_target.items():
                # Handle both tensor and list/tuple values
                if isinstance(value, (list, tuple)):
                    losses[f'{key}_target'] = [v * w_target for v in value]

                    # Aggregate for printing
                    total_value = sum(v.sum() for v in value)
                    weighted_total = total_value * w_target
                    print(f"    {key}: {total_value.item():.6f} (weighted: {weighted_total.item():.6f})")

                else:
                    losses[f'{key}_target'] = value * w_target
                    print(f"    {key}: {value.item():.6f} (weighted: {(value * w_target).item():.6f})")

            print(f"✅ [LOSS] Target loss computed")
        else:
            print("⚠️  [LOSS] Target loss weight = 0, skipping")
        
        # ========== TERM 3: Contrastive Loss ==========
        print(f"\n{'─'*80}")
        print(f"[LOSS] TERM 3: Contrastive Consistency Loss")
        print(f"{'─'*80}")
        
        use_bev = self.mean_teacher_cfg.get("use_bev_consistency", False)
        print(f"  BEV consistency enabled: {use_bev}")
        
        if use_bev and w_cont > 0:
            print(f"  Running student forward pass...")
            student_pred = self.student.predict(
                target_strong_processed,
                target_samples_strong_processed,
                return_bev_features=True
            )
            
            loss_contrastive_total = torch.tensor(0., device=device)
            num_valid_samples = 0
            
            for i in range(len(student_pred)):
                print(f"\n  ─── Sample {i+1}/{len(student_pred)} ───")
                
                try:
                    # Extract BEV features
                    bev_s = student_pred[i].bev_features
                    bev_t = filtered_teacher_preds[i].bev_features
                    
                    if bev_s is None or bev_t is None:
                        print("⚠️  [CONTRASTIVE] BEV features not available, skipping")
                        continue
                    
                    boxes = filtered_teacher_preds[i].pred_instances_3d.bboxes_3d.tensor
                    labels = filtered_teacher_preds[i].pred_instances_3d.labels_3d
                    scores = filtered_teacher_preds[i].pred_instances_3d.scores_3d if hasattr(
                        filtered_teacher_preds[i].pred_instances_3d, 'scores_3d') else None
                    
                    if len(boxes) == 0:
                        print("⚠️  [CONTRASTIVE] No boxes after filtering, skipping")
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
                print(f"\n✅ [LOSS] Contrastive loss: {losses['loss_contrastive'].item():.6f} "
                    f"(avg: {avg_contrastive.item():.6f}, valid samples: {num_valid_samples})")
            else:
                losses['loss_contrastive'] = torch.tensor(0., device=device)
                print(f"⚠️  [LOSS] No valid samples for contrastive loss")
        else:
            losses['loss_contrastive'] = torch.tensor(0., device=device)
            print("⚠️  [LOSS] Contrastive loss disabled or weight = 0")
        
        # ========== Summary ==========
        print(f"\n{'='*80}")
        print(f"[LOSS] Loss Summary")
        print(f"{'='*80}")
        
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
        
        print(f"  Source loss:       {source_total.item():.6f}")
        print(f"  Target loss:       {target_total.item():.6f}")
        print(f"  Contrastive loss:  {contrastive_total.item():.6f}")
        print(f"  ───────────────────────────────")
        print(f"  TOTAL LOSS:        {total_loss.item():.6f}")
        
        # Add metadata for logging
        losses['_source_total'] = source_total
        losses['_target_total'] = target_total
        losses['_num_pseudo_labels'] = torch.tensor(num_valid_samples, device=device)
        losses['_total_loss'] = total_loss
        
        print(f"✅ [LOSS] Loss computation complete")
        print(f"{'='*80}\n")
        
        return losses


    def predict(self, batch_inputs, batch_data_samples,
                use_teacher=False, **kwargs):
        if use_teacher:
            return self.teacher.predict(batch_inputs, batch_data_samples, **kwargs)
        else:
            return self.student.predict(batch_inputs, batch_data_samples, **kwargs)

    def forward(self, *args, mode='tensor', **kwargs):

            if mode == 'loss':
                return self.loss(*args, **kwargs)
            elif mode == 'predict':
                return self.predict(*args, **kwargs)
            else:
                raise ValueError(f"Invalid mode: {mode}")
    
    def _forward(self, batch_inputs, batch_data_samples=None):
        return self.student._forward(batch_inputs, batch_data_samples)
        