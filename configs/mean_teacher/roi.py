import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.structures import LiDARInstance3DBoxes


class ROIFeatureExtractor(nn.Module):
    """
    Extract per-object features from BEV feature maps for single-stage detectors.
    This simulates ROI head features that two-stage detectors naturally have.
    """
    
    def __init__(self, 
                 in_channels=256,
                 out_channels=256,
                 roi_size=7,  # ROI grid size (7x7)
                 voxel_size=0.16):
        super().__init__()
        
        self.roi_size = roi_size
        self.voxel_size = voxel_size
        
        # Feature extraction network (similar to ROI head)
        self.roi_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to get per-ROI feature
        )
        
        # Project to final feature dimension
        self.feature_proj = nn.Linear(out_channels, out_channels)
    
    def extract_roi_features(self, bev_features, boxes_3d):
        """
        Extract features for each detected box from BEV feature map.
        
        Args:
            bev_features: BEV feature map [C, H, W]
            boxes_3d: Detected 3D boxes [N, 7] (x, y, z, dx, dy, dz, heading)
            
        Returns:
            roi_features: Per-box features [N, C]
        """
        if len(boxes_3d) == 0:
            return torch.zeros((0, self.roi_encoder[-2].out_channels), 
                             device=bev_features.device)
        
        device = bev_features.device
        C, H, W = bev_features.shape
        N = len(boxes_3d)
        
        # Convert boxes to BEV pixel coordinates
        boxes_tensor = boxes_3d.tensor if isinstance(boxes_3d, LiDARInstance3DBoxes) else boxes_3d
        
        # Get box centers and dimensions in BEV
        centers = boxes_tensor[:, :2]  # [N, 2] (x, y)
        dims = boxes_tensor[:, 3:5]    # [N, 2] (dx, dy)
        
        # Convert to pixel coordinates
        centers_px = centers / self.voxel_size  # [N, 2]
        dims_px = dims / self.voxel_size        # [N, 2]
        
        roi_features_list = []
        
        for i in range(N):
            cx, cy = centers_px[i]
            dx, dy = dims_px[i]
            
            # Define ROI bounds (with padding)
            x_min = int(torch.clamp(cx - dx/2 - 2, 0, W-1))
            x_max = int(torch.clamp(cx + dx/2 + 2, 0, W-1))
            y_min = int(torch.clamp(cy - dy/2 - 2, 0, H-1))
            y_max = int(torch.clamp(cy + dy/2 + 2, 0, H-1))
            
            # Extract ROI from BEV features
            roi_feat = bev_features[:, y_min:y_max, x_min:x_max]  # [C, h, w]
            
            # Handle edge case: empty ROI
            if roi_feat.numel() == 0:
                roi_feat = torch.zeros((C, 1, 1), device=device)
            
            # Resize to fixed size and add batch dim
            roi_feat = F.interpolate(
                roi_feat.unsqueeze(0), 
                size=(self.roi_size, self.roi_size),
                mode='bilinear',
                align_corners=False
            )  # [1, C, roi_size, roi_size]
            
            roi_features_list.append(roi_feat)
        
        # Stack and process through encoder
        roi_features_batch = torch.cat(roi_features_list, dim=0)  # [N, C, roi_size, roi_size]
        encoded_features = self.roi_encoder(roi_features_batch)   # [N, C, 1, 1]
        encoded_features = encoded_features.squeeze(-1).squeeze(-1)  # [N, C]
        
        # Project to final dimension
        roi_features = self.feature_proj(encoded_features)  # [N, C]
        
        return roi_features


class HybridDomainContrastiveLoss(nn.Module):
    """
    Hybrid Domain Contrastive Loss using ROI-level features.
    
    Creates hybrid negative samples by combining:
    - High-confidence foreground (target domain objects)
    - Low-confidence background (hard negatives)
    
    This bridges source and target domains at the instance level.
    """
    
    def __init__(self,
                 tau=0.07,
                 neg_threshold=0.3,
                 pos_threshold=0.7,
                 use_reweighting=True,
                 score_type='hybrid_iou_cls',
                 iou_weight=0.5,
                 cls_weight=0.5):
        super().__init__()
        
        self.tau = tau
        self.neg_threshold = neg_threshold
        self.pos_threshold = pos_threshold
        self.use_reweighting = use_reweighting
        self.score_type = score_type
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
    
    def compute_confidence_scores(self, cls_scores, iou_scores=None):
        """Compute final confidence scores by fusing classification and IoU scores."""
        cls_scores = torch.sigmoid(cls_scores)
        
        if self.score_type == 'cls' or iou_scores is None:
            return cls_scores
        
        iou_scores = torch.sigmoid(iou_scores)
        
        if self.score_type == 'iou':
            return iou_scores
        elif self.score_type == 'hybrid_iou_cls':
            return self.iou_weight * iou_scores + self.cls_weight * cls_scores
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")
    
    def forward(self, 
                student_features,   # [N_s, C] from student on strong aug
                teacher_features,   # [N_t, C] from teacher on weak aug
                teacher_scores,     # [N_t] teacher confidence scores
                teacher_iou_scores=None):  # [N_t] teacher IoU scores (optional)
        """
        Compute hybrid domain contrastive loss.
        
        Args:
            student_features: ROI features from student model
            teacher_features: ROI features from teacher model
            teacher_scores: Teacher classification scores
            teacher_iou_scores: Teacher IoU prediction scores
            
        Returns:
            loss: Contrastive loss value
        """
        if student_features.shape[0] == 0 or teacher_features.shape[0] == 0:
            return torch.tensor(0., device=student_features.device)
        
        # Compute teacher confidence scores
        conf_scores = self.compute_confidence_scores(teacher_scores, teacher_iou_scores)
        
        # === Separate teacher predictions into foreground and background ===
        # Foreground: high confidence (positives)
        fg_mask = conf_scores > self.pos_threshold
        teacher_fg_features = teacher_features[fg_mask]
        teacher_fg_scores = conf_scores[fg_mask]
        
        # Background: low confidence (hard negatives)
        bg_mask = conf_scores <= self.neg_threshold
        teacher_bg_features = teacher_features[bg_mask]
        
        # Check if we have enough samples
        if teacher_fg_features.shape[0] == 0 or teacher_bg_features.shape[0] == 0:
            return torch.tensor(0., device=student_features.device)
        
        # === Feature Reweighting (optional) ===
        if self.use_reweighting:
            # Weight student features by their confidence (if available)
            # For simplicity, we don't reweight student here, only teacher
            teacher_fg_features = teacher_fg_features * teacher_fg_scores.unsqueeze(1)
        
        # === Normalize Features ===
        student_features = F.normalize(student_features, dim=1)
        teacher_fg_features = F.normalize(teacher_fg_features, dim=1)
        teacher_bg_features = F.normalize(teacher_bg_features, dim=1)
        
        # === Create Hybrid Negative Samples ===
        # Combine foreground + background as negatives
        negative_samples = torch.cat([teacher_fg_features, teacher_bg_features], dim=0)
        
        # === Compute Similarities ===
        # Positive similarity: student vs teacher foreground
        similarity_pos = torch.mm(student_features, teacher_fg_features.t()) / self.tau  # [N_s, N_fg]
        
        # Negative similarity: student vs hybrid negatives
        similarity_neg = torch.mm(student_features, negative_samples.t()) / self.tau  # [N_s, N_fg + N_bg]
        
        # === InfoNCE Loss ===
        # For each student feature, maximize agreement with positives while minimizing with all negatives
        logsumexp_neg = torch.logsumexp(similarity_neg, dim=1, keepdim=True)  # [N_s, 1]
        
        # Numerator: sum over positive samples
        # Denominator: included in logsumexp_neg
        losses = -torch.logsumexp(similarity_pos - logsumexp_neg, dim=1)  # [N_s]
        
        return losses.mean()


class ROIContrastiveMeanTeacher(nn.Module):
    """
    Integration layer for Mean-Teacher with ROI-level contrastive learning.
    This can be added to your MeanTeacherHybridDetector.
    """
    
    def __init__(self,
                 bev_channels=256,
                 roi_feature_channels=256,
                 contrastive_cfg=dict(
                     tau=0.07,
                     neg_threshold=0.3,
                     pos_threshold=0.7,
                     use_reweighting=True,
                     score_type='hybrid_iou_cls',
                     iou_weight=0.5,
                     cls_weight=0.5
                 ),
                 voxel_size=0.16):
        super().__init__()
        
        # ROI feature extractor
        self.roi_extractor = ROIFeatureExtractor(
            in_channels=bev_channels,
            out_channels=roi_feature_channels,
            voxel_size=voxel_size
        )
        
        # Contrastive loss module
        self.contrastive_loss = HybridDomainContrastiveLoss(**contrastive_cfg)
    
    def extract_roi_features_batch(self, bev_features_list, predictions_list):
        """
        Extract ROI features for a batch of samples.
        
        Args:
            bev_features_list: List of BEV feature maps [C, H, W]
            predictions_list: List of prediction results with bboxes_3d
            
        Returns:
            roi_features_list: List of ROI features [N_i, C] per sample
        """
        roi_features_list = []
        
        for bev_feat, pred in zip(bev_features_list, predictions_list):
            boxes = pred.pred_instances_3d.bboxes_3d
            roi_feat = self.roi_extractor.extract_roi_features(bev_feat, boxes)
            roi_features_list.append(roi_feat)
        
        return roi_features_list
    
    def compute_contrastive_loss(self,
                                 student_bev_features,
                                 student_predictions,
                                 teacher_bev_features,
                                 teacher_predictions):
        """
        Compute ROI-level contrastive loss for entire batch.
        
        Args:
            student_bev_features: List of student BEV features
            student_predictions: List of student predictions
            teacher_bev_features: List of teacher BEV features
            teacher_predictions: List of teacher predictions (filtered)
            
        Returns:
            loss: Average contrastive loss over batch
        """
        batch_loss = 0.0
        num_valid = 0
        
        for s_bev, s_pred, t_bev, t_pred in zip(
            student_bev_features, student_predictions,
            teacher_bev_features, teacher_predictions):
            
            # Extract ROI features
            student_roi_feat = self.roi_extractor.extract_roi_features(
                s_bev, s_pred.pred_instances_3d.bboxes_3d)
            teacher_roi_feat = self.roi_extractor.extract_roi_features(
                t_bev, t_pred.pred_instances_3d.bboxes_3d)
            
            if student_roi_feat.shape[0] == 0 or teacher_roi_feat.shape[0] == 0:
                continue
            
            # Get teacher scores
            teacher_scores = t_pred.pred_instances_3d.scores_3d
            teacher_iou = getattr(t_pred.pred_instances_3d, 'iou_scores_3d', None)
            
            # Compute loss
            loss = self.contrastive_loss(
                student_roi_feat,
                teacher_roi_feat,
                teacher_scores,
                teacher_iou
            )
            
            batch_loss += loss
            num_valid += 1
        
        return batch_loss / max(num_valid, 1)


# === Example Integration ===
def integrate_roi_contrastive_in_mean_teacher():
    """
    Example of how to integrate ROI-level contrastive loss into your
    MeanTeacherHybridDetector.
    """
    
    # In your MeanTeacherHybridDetector.__init__():
    roi_contrastive_cfg = dict(
        bev_channels=256,  # Match your detector's BEV channel count
        roi_feature_channels=256,
        contrastive_cfg=dict(
            tau=0.07,
            neg_threshold=0.3,
            pos_threshold=0.7,
            use_reweighting=True,
            score_type='hybrid_iou_cls',
            iou_weight=0.5,
            cls_weight=0.5
        ),
        voxel_size=0.16
    )
    
    # Add to your detector
    # self.roi_contrastive = ROIContrastiveMeanTeacher(**roi_contrastive_cfg)
    
    # In your loss() method, after getting student/teacher predictions:
    """
    # Extract BEV features (ensure return_bev=True in predict)
    student_bev_list = [pred.bev_features for pred in student_predictions]
    teacher_bev_list = [pred.bev_features for pred in teacher_predictions]
    
    # Compute ROI-level contrastive loss
    loss_roi_contrastive = self.roi_contrastive.compute_contrastive_loss(
        student_bev_list,
        student_predictions,
        teacher_bev_list,
        filtered_teacher_predictions
    )
    
    losses['loss_roi_contrastive'] = loss_roi_contrastive * w_roi_contrastive
    """
    
    pass