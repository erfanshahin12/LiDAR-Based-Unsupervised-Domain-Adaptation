import copy
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.base import Base3DDetector


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

        # Build Student and Teacher Detectors
        self.student = MODELS.build(detector)
        self.teacher = MODELS.build(detector)

        # Teacher never receives gradients
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.mean_teacher_cfg = mean_teacher_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Initialize teacher with student weights
        for t_param, s_param in zip(self.teacher.parameters(),
                                    self.student.parameters()):
            t_param.data.copy_(s_param.data)

    # EMA Update — called by training hook every iteration
    @torch.no_grad()
    def ema_update(self):
        alpha = self.mean_teacher_cfg.get('ema_momentum', 0.999)
        for t_param, s_param in zip(self.teacher.parameters(),
                                    self.student.parameters()):
            t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data

    def extract_feat(self, model, inputs, data_samples):
        """
        Extract BEV features from the model.
        This is a placeholder - you need to implement this based on your detector.
        
        Different detectors have different ways to access BEV features:
        - BEVFormer: model.pts_bbox_head.transformer outputs BEV features
        - CenterPoint: model.pts_bbox_head.shared_conv outputs features
        - You may need to modify your base detector to expose these
        """
        pass

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
        scores = teacher_pred.get('scores_3d', None)
        
        if scores is None:
            # No scores available, return all predictions
            return teacher_pred
        
        # Get threshold
        conf_threshold = self.mean_teacher_cfg.get('conf_threshold', 0.6)
        use_class_specific = self.mean_teacher_cfg.get('use_class_specific_thresh', False)
        
        # Build mask: True if confidence > threshold for that class
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if use_class_specific:
            # Apply different thresholds per class
            class_thresholds = self.mean_teacher_cfg.get('class_thresholds', {})
            labels = teacher_pred['labels_3d']
            
            for class_id, thresh in class_thresholds.items():
                class_mask = (labels == class_id) & (scores >= thresh)
                mask |= class_mask
            
            # Add default threshold for classes not in class_thersholds dict
            default_mask = torch.ones_like(scores, dtype=torch.bool)
            for class_id in class_thresholds.keys():
                default_mask &= (labels != class_id)
            default_mask &= (scores >= conf_threshold)
            mask |= default_mask
        else:
            # Single global threshold
            mask = scores >= conf_threshold
        
        # Create filtered prediction (deep copy to avoid modifying original)
        filtered_pred = copy.deepcopy(teacher_pred)
        
        # Filter pred_instances_3d
        filtered_pred.pred_instances_3d.bboxes_3d = teacher_pred.pred_instances_3d.bboxes_3d[mask]
        filtered_pred.pred_instances_3d.scores_3d = teacher_pred.pred_instances_3d.scores_3d[mask]
        filtered_pred.pred_instances_3d.labels_3d = teacher_pred.pred_instances_3d.labels_3d[mask]
        
        # Preserve BEV features (they are not filtered, it's a spatial feature map)
        if hasattr(teacher_pred, 'bev_features'):
            filtered_pred.bev_features = teacher_pred.bev_features
        
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
        
        if len(boxes) == 0:
            return torch.tensor(0., device=bev_s.device)

        # Compute BEV centers of boxes (x,y in BEV pixel coords)
        # boxes[:, :2] = (x,y) center in meters
        # Convert to pixel indices
        bev_resolution = self.mean_teacher_cfg["voxel_size"]                    # assume square voxels in x,y
        xs = (boxes[:, 0] / bev_resolution).long().clamp(0, bev_s.shape[2]-1)
        ys = (boxes[:, 1] / bev_resolution).long().clamp(0, bev_s.shape[1]-1)

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

        # Avoid zero division
        pos_count = pos_mask.sum(dim=1)  # N

        # Compute contrastive loss
        # log_softmax over j dimension
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # For each i: average only over positive j’s
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-6)
   
        # Optional: weight by confidence scores
        if scores is not None:
            loss = loss * scores

        return lambda_weight * loss.mean()

    def _create_pseudo_labels(self, teacher_predictions, target_samples):
        """
        Create pseudo-labeled data samples from teacher predictions.
        
        Args:
            teacher_predictions: List of filtered teacher predictions
            target_samples: Original target data samples
        
        Returns:
            pseudo_labeled_samples: Data samples with teacher's predictions as ground truth
        """
        
        # Deep copy to avoid modifying original samples
        pseudo_labeled_samples = copy.deepcopy(target_samples)
        
        # Replace ground truth with teacher's pseudo-labels
        for i, (pred, sample) in enumerate(zip(teacher_predictions, pseudo_labeled_samples)):
            # Replace bboxes with teacher predictions
            sample.gt_instances_3d.bboxes_3d = pred['bboxes_3d']
            sample.gt_instances_3d.labels_3d = pred['labels_3d']

            # Optional: add confidence scores as weights
            if 'scores_3d' in pred:
                sample.gt_instances_3d.scores = pred['scores_3d']
        
        return pseudo_labeled_samples
    
    def loss(self, batch_inputs_dict, batch_data_samples):

        """
        Compute total loss with three components:
        1. Supervised loss on source/labeled data (regression + classification)
        2. Pseudo-label loss on target/unlabeled data (regression + classification)
        3. Contrastive consistency loss on target data (BEV features)
        
        Expected input format:
        batch_inputs_dict = {
            "labeled": {...},
            "unlabeled": {
                "weak": {...},
                "strong": {...}}} """

        # Load weights from config
        w_source = self.mean_teacher_cfg.get('source_loss_weight', 1.0)
        w_target = self.mean_teacher_cfg.get('target_loss_weight', 1.0)
        w_cont = self.mean_teacher_cfg.get('contrastive_weight', 0.1)

        losses = {}

        # TERM 1: Supervised Loss on Source Data
        source_inputs = batch_inputs_dict["labeled"]
        source_samples = batch_data_samples["labeled"]

        # Compute student loss on labeled source data
        if w_source > 0:
            loss_source = self.student.loss(source_inputs, source_samples)
            # Rename keys to avoid collision or confusion
            for key, value in loss_source.items():
                losses[f'{key}_source'] = value * w_source

        # TERM 2 & 3: Pseudo-Label Loss + Contrastive Loss on Target Data
        # Use weak augmentation for teacher, strong for student
        target_weak = batch_inputs_dict["unlabeled"]["weak"]
        target_strong = batch_inputs_dict["unlabeled"]["strong"]
        target_samples = batch_data_samples["unlabeled"]

        # Teacher forward on weakly augmented data (no grad)
        with torch.no_grad():
            teacher_pred = self.teacher.predict(
                target_weak,
                target_samples,
                return_bev_features=True)

        # Filter by confidence and prepare pseudo-labels
            filtered_teacher_preds = []
            for i in range(len(teacher_pred)):
                filtered_pred = self.filter_teacher_predictions(teacher_pred[i])
                filtered_teacher_preds.append(filtered_pred)

        # TERM 2: Pseudo-Label Loss (Regression + Classification)
        # Create pseudo-labeled data samples from teacher predictions
        pseudo_labeled_samples = self._create_pseudo_labels(
                                        filtered_teacher_preds, 
                                        target_samples)
        
        # Compute student loss on pseudo-labeled target data
        if w_target > 0:
            loss_target = self.student.loss(target_strong, pseudo_labeled_samples)
            # Apply target weight and add with prefix
            for key, value in loss_target.items():
                losses[f'{key}_target'] = value * w_target

        # TERM 3: Contrastive Consistency Loss

        # Student forward on strongly augmented data
        student_pred = self.student.predict(
            target_strong,
            target_samples,
            return_bev_features=True)

        # consistency loss (batch sum)
        loss_contrastive_total = torch.tensor(0., device=losses['loss_source'].device)
        num_valid_samples = 0

        if self.mean_teacher_cfg.get("use_bev_consistency", False) and w_cont > 0:
            for i in range(len(student_pred)):
                try:
                    # Extract BEV features
                    bev_s = student_pred[i].get("bev_features")
                    bev_t = filtered_teacher_preds[i].get("bev_features")
                    
                    if bev_s is None or bev_t is None:
                        continue
                    
                    boxes = filtered_teacher_preds[i]["bboxes_3d"].tensor
                    labels = filtered_teacher_preds[i]["labels_3d"]
                    scores = filtered_teacher_preds[i].get("scores_3d", None)
                    
                    # Skip if no high-confidence predictions
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
                    print(f"Warning: Could not compute contrastive loss for sample {i}: {e}")
                    continue

        # Add contrastive loss
        if num_valid_samples > 0:
            losses['loss_contrastive'] = (loss_contrastive_total / num_valid_samples) * w_cont
        else:
            losses['loss_contrastive'] = torch.tensor(0., device=losses['loss_source'].device)
        
        # Add aggregated losses for easier monitoring and debugging
        # These won't affect training (prefixed with underscore so MMDet ignores them)
        source_total = sum([v for k, v in losses.items() if k.startswith('source_')])
        target_total = sum([v for k, v in losses.items() if k.startswith('target_')])
        
        losses['_source_total'] = source_total      # For logging only
        losses['_target_total'] = target_total      # For logging only
        losses['_num_pseudo_labels'] = torch.tensor(num_valid_samples, 
                                                     device=losses['source_loss_cls'].device)
        
        return losses

    # PREDICTION — use student model for inference
    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        return self.student.predict(batch_inputs, batch_data_samples, **kwargs)

    def forward(self, *args, mode='tensor', **kwargs):

            if mode == 'loss':
                return self.loss(*args, **kwargs)
            elif mode == 'predict':
                return self.predict(*args, **kwargs)
            else:
                raise ValueError(f"Invalid mode: {mode}")