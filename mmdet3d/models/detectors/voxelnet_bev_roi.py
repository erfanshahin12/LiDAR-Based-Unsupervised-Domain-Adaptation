import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import VoxelNet
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.structures.ops.iou3d_calculator import bbox_overlaps_nearest_3d


class BEVRoIIoUHead(nn.Module):
    """Post-NMS RoI-level IoU quality head. Operates on RoI-pooled BEV features
    from decoded boxes, giving calibrated per-proposal scores unlike the
    single-stage per-anchor conv_iou.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, roi_feats: Tensor) -> Tensor:
        """Args: roi_feats [N, in_dim]. Returns: iou_scores [N] in (0, 1)."""
        return self.fc(roi_feats).squeeze(-1).sigmoid()


@MODELS.register_module()
class VoxelNetBEVRoI(VoxelNet):
    """
    Extended VoxelNet + BEV features + RoI features
    Features:
        - BEV feature extraction for spatial consistency
        - ROI feature extraction for instance-level consistency
    This is needed for Mean-Teacher consistency loss which operates on BEV and RoI features.
    """

    def __init__(self,
                 roi_extractor_cfg=None,
                 bev_roi_iou_head_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.return_bev_features = False
        self.return_roi_features = False

        if roi_extractor_cfg is not None:
            roi_out_channels = roi_extractor_cfg.get('out_channels', 256)
            self.roi_extractor = ROIFeatureExtractor(
                in_channels=roi_extractor_cfg.get('in_channels', 64),
                out_channels=roi_out_channels,
                roi_size=roi_extractor_cfg.get('roi_size', 7),
                voxel_size=roi_extractor_cfg.get('voxel_size', 0.2),
                point_cloud_range=roi_extractor_cfg.get('point_cloud_range', None),
            )
        else:
            self.roi_extractor = None
            roi_out_channels = 256

        if bev_roi_iou_head_cfg is not None and self.roi_extractor is not None:
            self.bev_roi_iou_head = BEVRoIIoUHead(
                in_dim=roi_out_channels,
                hidden_dim=bev_roi_iou_head_cfg.get('hidden_dim', 256),
            )
        else:
            self.bev_roi_iou_head = None
    
    def extract_feat(self, batch_inputs_dict: dict, 
                    return_bev: bool = False) -> Tuple[Tensor]:
        """
        Extract neck and BEV features from points.
        
        Args:
            batch_inputs_dict: Dict containing voxel information
            return_bev: If True, return (features, bev_features) tuple
        
        Returns:
            If return_bev=False: tuple of feature maps from neck (same as original VoxelNet)
            If return_bev=True: (neck_features, bev_feature_map)
        """
        voxel_dict = batch_inputs_dict['voxels']
        
        # Voxel encoding: sparse voxels → voxel features
        voxel_features = self.voxel_encoder(
            voxel_dict['voxels'],
            voxel_dict['num_points'],
            voxel_dict['coors'])
        
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        
        # Scatter to BEV: sparse voxel features → dense BEV map
        # Here is the BEV feature map
        bev_features = self.middle_encoder(
            voxel_features,
            voxel_dict['coors'],
            batch_size)
        
        # Backbone: extract multi-scale features
        x = self.backbone(bev_features)
        
        # Neck: fuse multi-scale features
        if self.with_neck:
            x = self.neck(x)
        
        if return_bev:
            # Return both neck features (for detection) and BEV features (for consistency)
            return x, bev_features
        else:
            return x

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: List[Det3DDataSample],
                return_bev_features: bool = False,
                                return_roi_features: bool = False,
                **kwargs) -> List[Det3DDataSample]:
        """
        Predict results from a batch of inputs and data samples.
        
        Args:
            batch_inputs_dict: Input point clouds
            batch_data_samples: Data samples
            return_bev_features: If True, add BEV features to results
            return_roi_features: If True, add ROI features to results
        
        Returns:
            List of Det3DDataSample containing:
            - pred_instances_3d (always)
            - bev_features (if return_bev_features=True)
            - roi_features (if return_roi_features=True)
        """

        # Always need BEV features if the RoI IoU head is active.
        need_bev = (return_bev_features or self.return_bev_features
                    or self.bev_roi_iou_head is not None)
        if need_bev:
            x, bev_features = self.extract_feat(batch_inputs_dict, return_bev=True)
        else:
            x = self.extract_feat(batch_inputs_dict, return_bev=False)
            bev_features = None

        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples, results_list)

        for i, data_sample in enumerate(predictions):
            if return_bev_features or self.return_bev_features:
                data_sample.bev_features = bev_features[i]

            # Post-NMS RoI IoU refinement: override the per-anchor iou_scores_3d with
            # calibrated scores from the two-stage BEV RoI head.
            if bev_features is not None and self.bev_roi_iou_head is not None:
                boxes_3d = data_sample.pred_instances_3d.bboxes_3d
                if len(boxes_3d) > 0:
                    roi_feats = self.roi_extractor.extract_roi_features(
                        bev_features[i], boxes_3d)
                    data_sample.pred_instances_3d.iou_scores_3d = (
                        self.bev_roi_iou_head(roi_feats))
                else:
                    data_sample.pred_instances_3d.iou_scores_3d = (
                        boxes_3d.tensor.new_zeros(0))

            if return_roi_features and self.roi_extractor is not None:
                boxes_3d = data_sample.pred_instances_3d.bboxes_3d
                roi_features = self.roi_extractor.extract_roi_features(
                    bev_features[i], boxes_3d)
                data_sample.roi_features = roi_features

        return predictions

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Standard anchor-head loss plus BEV RoI IoU head supervision (when active)."""
        if self.bev_roi_iou_head is not None:
            x, bev_features = self.extract_feat(batch_inputs_dict, return_bev=True)
        else:
            x = self.extract_feat(batch_inputs_dict)
            bev_features = None

        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)

        if bev_features is not None:
            losses['loss_bev_roi_iou'] = self._compute_bev_roi_iou_loss(
                x, bev_features, batch_data_samples)

        return losses

    def _compute_bev_roi_iou_loss(self, x, bev_features, batch_data_samples):
        """Compute BCE loss for the BEV RoI IoU head on post-NMS proposals.

        The backbone/neck/anchor-head are assumed frozen during the finetune phase,
        so we run bbox_head.predict() under no_grad and detach BEV features before
        RoI pooling. Gradients only flow through bev_roi_iou_head parameters.
        """
        device = bev_features.device

        with torch.no_grad():
            results_list = self.bbox_head.predict(x, batch_data_samples)

        total_loss = bev_features.new_zeros(1).squeeze()
        n_valid = 0

        for i, (inst, sample) in enumerate(zip(results_list, batch_data_samples)):
            if len(inst.bboxes_3d) == 0:
                continue

            proposal_boxes = inst.bboxes_3d.tensor[:, :7]

            gt = getattr(sample, 'gt_instances_3d', None)
            if gt is None or len(gt.bboxes_3d) == 0:
                iou_targets = proposal_boxes.new_zeros(len(proposal_boxes))
            else:
                gt_boxes = gt.bboxes_3d.tensor[:, :7].to(device)
                iou_mat = bbox_overlaps_nearest_3d(
                    proposal_boxes, gt_boxes, mode='iou', is_aligned=False)
                iou_targets = iou_mat.max(dim=1)[0]

            # RoI features from frozen extractor — detach so no grad flows to backbone.
            roi_feats = self.roi_extractor.extract_roi_features(
                bev_features[i].detach(), inst.bboxes_3d)

            # Use raw logits + BCEWithLogits to be AMP-safe.
            logits = self.bev_roi_iou_head.fc(roi_feats).squeeze(-1)
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                logits, iou_targets.clamp(0., 1.))
            n_valid += 1

        return total_loss / n_valid if n_valid > 0 else total_loss


class ROIFeatureExtractor(nn.Module):
    """
    Extract per-object features from BEV feature maps for single-stage detectors.
    Simulating ROI head features in two-stage detectors.
    """
    
    def __init__(self, 
                 in_channels=256,
                 out_channels=256,
                 roi_size=7,  # ROI grid size (7x7)
                 voxel_size=0.16,
                 point_cloud_range=None):
        super().__init__()
        
        self.roi_size = roi_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range or [-74.88, -74.88, -2, 74.88, 74.88, 4]
        
        # Feature extraction network (similar to ROI head)
        self.roi_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to get per-ROI feature
        )
        
        # Project to final feature dimension
        self.feature_proj = nn.Linear(out_channels, out_channels)
    
    def boxes_to_bev_pixels(self, boxes_3d, bev_h, bev_w):
        """
        Convert 3D boxes to BEV pixel coordinates.
        
        Args:
            boxes_3d: LiDARInstance3DBoxes or tensor [N, 7+]
            bev_h, bev_w: BEV feature map dimensions
            
        Returns:
            centers_px: [N, 2] pixel centers
            dims_px: [N, 2] pixel dimensions
        """
        # Extract box parameters
        if isinstance(boxes_3d, LiDARInstance3DBoxes):
            boxes_tensor = boxes_3d.tensor
        else:
            boxes_tensor = boxes_3d
        
        # Get box centers and dimensions in world coordinates
        centers_world = boxes_tensor[:, :2]  # [N, 2] (x, y)
        dims_world = boxes_tensor[:, 3:5]    # [N, 2] (dx, dy)
        
        # Convert from world coordinates to BEV pixel coordinates
        # Point cloud range: [x_min, y_min, z_min, x_max, y_max, z_max]
        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        x_max, y_max = self.point_cloud_range[3], self.point_cloud_range[4]
        
        # Normalize to [0, 1]
        centers_norm = torch.zeros_like(centers_world)
        centers_norm[:, 0] = (centers_world[:, 0] - x_min) / (x_max - x_min)  # x
        centers_norm[:, 1] = (centers_world[:, 1] - y_min) / (y_max - y_min)  # y
        
        # Scale to pixel coordinates
        centers_px = torch.zeros_like(centers_norm)
        centers_px[:, 0] = centers_norm[:, 0] * bev_w  # x -> width
        centers_px[:, 1] = centers_norm[:, 1] * bev_h  # y -> height
        
        # Convert dimensions to pixels
        dims_px = torch.zeros_like(dims_world)
        dims_px[:, 0] = dims_world[:, 0] / (x_max - x_min) * bev_w  # dx
        dims_px[:, 1] = dims_world[:, 1] / (y_max - y_min) * bev_h  # dy
        
        return centers_px, dims_px
    
    def extract_roi_features(self, bev_features, boxes_3d):
        """
        Extract features for each detected box from BEV feature map.
        
        Args:
            bev_features: BEV feature map [C, H, W]
            boxes_3d: Detected 3D boxes [N, 7] or LiDARInstance3DBoxes
            
        Returns:
            roi_features: Per-box features [N, C]
        """
        if isinstance(boxes_3d, LiDARInstance3DBoxes):
            num_boxes = len(boxes_3d)
        else:
            num_boxes = len(boxes_3d) if boxes_3d.dim() > 1 else 0
            
        if num_boxes == 0:
            out_channels = self.feature_proj.out_features
            return torch.zeros((0, out_channels), device=bev_features.device)
        
        device = bev_features.device
        C, H, W = bev_features.shape
        
        # Convert boxes to BEV pixel coordinates
        centers_px, dims_px = self.boxes_to_bev_pixels(boxes_3d, H, W)
        
        roi_features_list = []
        
        for i in range(num_boxes):
            cx, cy = centers_px[i]
            dx, dy = dims_px[i]
            
            # Define ROI bounds with padding (to capture context)
            padding = 2.0  # pixels
            x_min = int(torch.clamp(cx - dx/2 - padding, 0, W-1).item())
            x_max = int(torch.clamp(cx + dx/2 + padding, 1, W).item())
            y_min = int(torch.clamp(cy - dy/2 - padding, 0, H-1).item())
            y_max = int(torch.clamp(cy + dy/2 + padding, 1, H).item())
            
            # Ensure valid bounds
            if x_max <= x_min or y_max <= y_min:
                # Fallback to center point if invalid bounds
                x_min = max(0, int(cx.item()) - 1)
                x_max = min(W, int(cx.item()) + 1)
                y_min = max(0, int(cy.item()) - 1)
                y_max = min(H, int(cy.item()) + 1)
            
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
