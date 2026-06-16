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
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas,
                                                 return_bev=False)

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

        # Attach the NECK feature map (the detection representation the head
        # regresses from) as `bev_features`, for the Mean-Teacher contrastive /
        # RoI consistency loss. Previously this exposed the pre-backbone
        # SparseEncoder map, which the head never sees. SECONDFPN returns a
        # single-level list, so unwrap to a [B, C, H, W] tensor before indexing.
        if return_bev_features or return_roi_features:
            neck_map = pts_feats[0] if isinstance(pts_feats, (list, tuple)) else pts_feats
            for i, data_sample in enumerate(detsamples):
                if return_bev_features:
                    data_sample.bev_features = neck_map[i]  # [C, H, W]
                if return_roi_features and self.roi_extractor is not None:
                    boxes_3d = data_sample.pred_instances_3d.bboxes_3d
                    data_sample.roi_features = \
                        self.roi_extractor.extract_roi_features(neck_map[i], boxes_3d)

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
    
    def extract_roi_features(self, bev_features, boxes_3d):
        """Extract per-box features using rotation-aware affine RoI pooling.

        An affine grid aligned to each box's heading angle is sampled from the
        BEV feature map (``F.grid_sample``), giving the encoder a
        heading-canonical, differentiable view of the box interior. Replaces the
        former axis-aligned integer-crop loop (which ignored yaw and broke the
        gradient w.r.t. the box footprint). Mirrors ``VoxelNetBEVRoI`` and
        follows ST3D's ``roi_grid_pool``.

        Args:
            bev_features (Tensor): BEV feature map ``[C, H, W]`` for one sample.
            boxes_3d: ``[N, 7]`` tensor or ``LiDARInstance3DBoxes``.

        Returns:
            Tensor: Per-box features ``[N, out_channels]``.
        """
        if isinstance(boxes_3d, LiDARInstance3DBoxes):
            boxes_t = boxes_3d.tensor
            num_boxes = len(boxes_3d)
        else:
            boxes_t = boxes_3d
            num_boxes = boxes_3d.shape[0] if boxes_3d.dim() > 1 else 0

        if num_boxes == 0:
            return torch.zeros((0, self.feature_proj.out_features),
                               device=bev_features.device)

        device = bev_features.device
        C, H, W = bev_features.shape

        x_min_r = self.point_cloud_range[0]
        y_min_r = self.point_cloud_range[1]
        range_x = self.point_cloud_range[3] - x_min_r
        range_y = self.point_cloud_range[4] - y_min_r

        bt = boxes_t.float().to(device)
        cx, cy = bt[:, 0], bt[:, 1]
        dx, dy = bt[:, 3], bt[:, 4]
        angle = bt[:, 6]

        # Box corners in feature-map pixel space.
        x1 = (cx - dx / 2 - x_min_r) / range_x * W
        x2 = (cx + dx / 2 - x_min_r) / range_x * W
        y1 = (cy - dy / 2 - y_min_r) / range_y * H
        y2 = (cy + dy / 2 - y_min_r) / range_y * H

        cosa = torch.cos(angle)
        sina = torch.sin(angle)

        # 2x3 affine matrices in normalised [-1, 1] space (ST3D convention).
        theta = torch.stack([
            (x2 - x1) / (W - 1) * cosa,
            (x2 - x1) / (W - 1) * (-sina),
            (x1 + x2 - W + 1) / (W - 1),
            (y2 - y1) / (H - 1) * sina,
            (y2 - y1) / (H - 1) * cosa,
            (y1 + y2 - H + 1) / (H - 1),
        ], dim=1).view(num_boxes, 2, 3)

        grid = F.affine_grid(
            theta,
            torch.Size((num_boxes, C, self.roi_size, self.roi_size)),
            align_corners=True)
        # Expand the feature map to [N, C, H, W] without copying memory.
        bev_exp = bev_features.unsqueeze(0).expand(num_boxes, C, H, W)
        roi_patches = F.grid_sample(
            bev_exp, grid, align_corners=True)  # [N, C, roi_size, roi_size]

        encoded = self.roi_encoder(roi_patches)    # [N, out_ch, 1, 1]
        encoded = encoded.squeeze(-1).squeeze(-1)  # [N, out_ch]
        return self.feature_proj(encoded)          # [N, out_ch]