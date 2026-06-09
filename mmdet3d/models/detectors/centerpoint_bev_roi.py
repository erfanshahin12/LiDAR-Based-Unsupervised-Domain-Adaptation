# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Sequence


from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


@MODELS.register_module()
class CenterPointBEVRoI(CenterPoint):
    """Extended CenterPoint + BEV features + RoI features
    Features:
        - BEV feature extraction for spatial consistency
        - ROI feature extraction for instance-level consistency
    This is needed for Mean-Teacher consistency loss which operates on BEV and RoI features.
    """

    def __init__(self,
                 roi_extractor_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize ROI feature extractor
        if roi_extractor_cfg is not None:
            # Get BEV channel count from middle_encoder output
            # Typically the same as backbone input channels
            bev_channels = roi_extractor_cfg.get('in_channels', 256)
            
            self.roi_extractor = ROIFeatureExtractor(
                in_channels=bev_channels,
                out_channels=roi_extractor_cfg.get('out_channels', 256),
                roi_size=roi_extractor_cfg.get('roi_size', 7),
                voxel_size=roi_extractor_cfg.get('voxel_size', 0.16),
                point_cloud_range=roi_extractor_cfg.get('point_cloud_range', None)
            )
        else:
            self.roi_extractor = None
        

    def extract_pts_feat(
            self,
            voxel_dict: Dict[str, Tensor],
            points: Optional[List[Tensor]] = None,
            img_feats: Optional[Sequence[Tensor]] = None,
            batch_input_metas: Optional[List[dict]] = None,
            return_bev: bool = False) -> Sequence[Tensor]:
        """
        Extract neck and BEV features from points.
        
        Args:
            batch_inputs_dict: Dict containing voxel information
            return_bev: If True, return (features, bev_features) tuple
        
        Returns:
            If return_bev=False: tuple of feature maps from neck (same as original CenterPoint)
            If return_bev=True: (neck_features, bev_feature_map)
        """
        if not self.with_pts_bbox:
            return None
        
        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'], img_feats,
                                                batch_input_metas)
        
        batch_size = voxel_dict['coors'][-1, 0] + 1

        bev_features = self.pts_middle_encoder(voxel_features,
                                               voxel_dict['coors'],
                                               batch_size)
        
        x = self.pts_backbone(bev_features)

        if self.with_pts_neck:
            x = self.pts_neck(x)

        if return_bev:
            return x, bev_features
        return x


    @property
    def bbox_head(self):
        """Alias so MeanTeacher3DDetector can call student.bbox_head uniformly."""
        return self.pts_bbox_head

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict] = None,
                     return_bev: bool = False) -> tuple:
        """Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        voxel_dict = batch_inputs_dict.get('voxels', None)
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        if return_bev:
            pts_feats, bev_features = self.extract_pts_feat(
                voxel_dict,
                points=points,
                img_feats=img_feats,
                batch_input_metas=batch_input_metas,
                return_bev=True)
            return img_feats, pts_feats, bev_features
        else:
            pts_feats = self.extract_pts_feat(
                voxel_dict,
                points=points,
                img_feats=img_feats,
                batch_input_metas=batch_input_metas,
                return_bev=False)
            return img_feats, pts_feats


    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
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

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        if return_bev_features:
            img_feats, pts_feats, bev_features = self.extract_feat(batch_inputs_dict,
                                                                   batch_input_metas,
                                                                   return_bev=True)
        else:
            img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                     batch_input_metas,
                                                     return_bev=False)
            bev_features = None

        if pts_feats and self.with_pts_bbox:
            results_list_3d = self.pts_bbox_head.predict(pts_feats, batch_data_samples, **kwargs)
        else:
            results_list_3d = None

        if img_feats and self.with_img_bbox:
            # TODO check this for camera modality
            results_list_2d = self.predict_imgs(img_feats, batch_data_samples, **kwargs)
        else:
            results_list_2d = None

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d,
                                                 results_list_2d)
        
        # Add BEV and RoI features to each sample
        for i, data_sample in enumerate(detsamples):
            if return_bev_features and bev_features is not None:
                # bev_features shape: [B, C, H, W]
                data_sample.bev_features = bev_features[i]  # [C, H, W]

            if return_roi_features and self.roi_extractor is not None:
                # Extract ROI features for this sample's predictions
                boxes_3d = data_sample.pred_instances_3d.bboxes_3d
                roi_features = self.roi_extractor.extract_roi_features(bev_features[i], boxes_3d)
                data_sample.roi_features = roi_features  # [N, C]

        return detsamples


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