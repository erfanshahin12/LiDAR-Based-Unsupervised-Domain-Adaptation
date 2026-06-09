import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import VoxelNet
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


@MODELS.register_module()
class VoxelNetBEVRoI(VoxelNet):
    """VoxelNet extended with BEV and RoI feature extraction.

    Adds:
      * ``extract_feat(return_bev=True)`` — returns both neck features and
        the pre-backbone BEV scatter map; consumed by MeanTeacher's
        contrastive/BEV-consistency loss.
      * ``ROIFeatureExtractor`` — per-object BEV crop features; also used by
        the MeanTeacher contrastive loss.
      * ``predict(return_bev_features=True, return_roi_features=True)`` — lets
        the teacher forward expose BEV and RoI features alongside detections.

    The quality/IoU score for each detected box (``iou_scores_3d``) is now
    produced post-NMS by the rotation-aware ``iou_mlp`` in ``Anchor3DHead``
    (when ``predict_iou=True`` and ``roi_extractor_cfg`` is set).
    """

    def __init__(self,
                 roi_extractor_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.return_bev_features = False
        self.return_roi_features = False

        if roi_extractor_cfg is not None:
            self.roi_extractor = ROIFeatureExtractor(
                in_channels=roi_extractor_cfg.get('in_channels', 64),
                out_channels=roi_extractor_cfg.get('out_channels', 256),
                roi_size=roi_extractor_cfg.get('roi_size', 7),
                voxel_size=roi_extractor_cfg.get('voxel_size', 0.2),
                point_cloud_range=roi_extractor_cfg.get(
                    'point_cloud_range', None),
            )
        else:
            self.roi_extractor = None

    def extract_feat(self, batch_inputs_dict: dict,
                     return_bev: bool = False) -> Tuple[Tensor]:
        """Extract neck and (optionally) BEV features from points.

        Args:
            batch_inputs_dict (dict): Dict containing voxel information.
            return_bev (bool): If True, return ``(neck_features, bev_features)``
                where ``bev_features`` is the raw PointPillarsScatter output
                (pre-backbone dense BEV map).  Required by MeanTeacher's
                contrastive loss.

        Returns:
            Tuple or Tensor: neck features, optionally with the BEV map.
        """
        voxel_dict = batch_inputs_dict['voxels']

        voxel_features = self.voxel_encoder(
            voxel_dict['voxels'],
            voxel_dict['num_points'],
            voxel_dict['coors'])

        batch_size = voxel_dict['coors'][-1, 0].item() + 1

        # Dense BEV map (pre-backbone scatter output).
        bev_features = self.middle_encoder(
            voxel_features,
            voxel_dict['coors'],
            batch_size)

        x = self.backbone(bev_features)
        if self.with_neck:
            x = self.neck(x)

        if return_bev:
            return x, bev_features
        return x

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: List[Det3DDataSample],
                return_bev_features: bool = False,
                return_roi_features: bool = False,
                **kwargs) -> List[Det3DDataSample]:
        """Run detection and optionally attach BEV / RoI features.

        Args:
            batch_inputs_dict: Input point clouds.
            batch_data_samples: Data samples.
            return_bev_features (bool): Attach the BEV map to each sample.
            return_roi_features (bool): Attach per-box RoI features to each
                sample (requires ``roi_extractor`` to be configured).

        Returns:
            List[Det3DDataSample]: Predictions.  Each sample's
            ``pred_instances_3d`` carries ``bboxes_3d``, ``scores_3d``,
            ``labels_3d``, and (when ``predict_iou=True`` in the head)
            ``iou_scores_3d`` produced by the per-anchor quality head.
        """
        need_bev = (return_bev_features or self.return_bev_features
                    or return_roi_features or self.return_roi_features
                    or getattr(self.bbox_head, 'has_iou_mlp', False))
        if need_bev:
            x, bev_features = self.extract_feat(
                batch_inputs_dict, return_bev=True)
        else:
            x = self.extract_feat(batch_inputs_dict, return_bev=False)
            bev_features = None

        if getattr(self.bbox_head, 'has_iou_mlp', False) \
                and bev_features is not None:
            kwargs['bev_features'] = [
                bev_features[i] for i in range(bev_features.shape[0])]

        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        for i, data_sample in enumerate(predictions):
            if (return_bev_features or self.return_bev_features) \
                    and bev_features is not None:
                data_sample.bev_features = bev_features[i]

            if (return_roi_features or self.return_roi_features) \
                    and self.roi_extractor is not None \
                    and bev_features is not None:
                boxes_3d = data_sample.pred_instances_3d.bboxes_3d
                data_sample.roi_features = \
                    self.roi_extractor.extract_roi_features(
                        bev_features[i], boxes_3d)

        return predictions

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Standard anchor-head detection loss.

        When the bbox_head has an ``iou_mlp`` (``has_iou_mlp=True``), the raw
        BEV feature map is extracted alongside the neck features and passed to
        ``bbox_head.loss`` as ``bev_features`` so the RoI IoU head can be
        trained.  The BEV map comes from ``PointPillarsScatter`` and is already
        computed as part of the normal forward pass — no extra cost.
        """
        if getattr(self.bbox_head, 'has_iou_mlp', False):
            x, bev = self.extract_feat(batch_inputs_dict, return_bev=True)
            # Split [B, C, H, W] into a per-image list for loss_by_feat
            kwargs['bev_features'] = [bev[i] for i in range(bev.shape[0])]
        else:
            x = self.extract_feat(batch_inputs_dict)
        return self.bbox_head.loss(x, batch_data_samples, **kwargs)


class ROIFeatureExtractor(nn.Module):
    """Extract per-object features from BEV feature maps for single-stage
    detectors, simulating the RoI-pooling stage of two-stage detectors.

    Used by MeanTeacher's contrastive/BEV-consistency loss.
    """

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 roi_size=7,
                 voxel_size=0.16,
                 point_cloud_range=None):
        super().__init__()

        self.roi_size = roi_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range or \
            [-74.88, -74.88, -2, 74.88, 74.88, 4]

        # Two-layer conv + global average pool → per-RoI descriptor.
        self.roi_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_proj = nn.Linear(out_channels, out_channels)

    def extract_roi_features(self, bev_features, boxes_3d):
        """Extract per-box features using rotation-aware affine RoI pooling.

        Replaces the former axis-aligned integer-crop loop.  An affine grid
        aligned to each box's heading angle is sampled from the BEV feature
        map, giving the encoder a heading-canonical view of the box interior.
        Follows the design of ST3D's ``SECONDHead.roi_grid_pool``.

        Args:
            bev_features (Tensor): BEV feature map ``[C, H, W]`` for one image.
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
            return torch.zeros(
                (0, self.feature_proj.out_features),
                device=bev_features.device)

        device = bev_features.device
        C, H, W = bev_features.shape

        x_min_r = self.point_cloud_range[0]
        y_min_r = self.point_cloud_range[1]
        range_x  = self.point_cloud_range[3] - x_min_r
        range_y  = self.point_cloud_range[4] - y_min_r

        bt = boxes_t.float().to(device)
        cx, cy = bt[:, 0], bt[:, 1]
        dx, dy = bt[:, 3], bt[:, 4]
        angle   = bt[:, 6]

        # Box corners in feature-map pixel space
        x1 = (cx - dx / 2 - x_min_r) / range_x * W
        x2 = (cx + dx / 2 - x_min_r) / range_x * W
        y1 = (cy - dy / 2 - y_min_r) / range_y * H
        y2 = (cy + dy / 2 - y_min_r) / range_y * H

        cosa = torch.cos(angle)
        sina = torch.sin(angle)

        # 2×3 affine matrices in normalised [-1,1] space (ST3D convention)
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
            align_corners=True,
        )
        # Expand feature map to [N, C, H, W] without copying memory
        bev_exp = bev_features.unsqueeze(0).expand(num_boxes, C, H, W)
        roi_patches = F.grid_sample(
            bev_exp, grid, align_corners=True)  # [N, C, roi_size, roi_size]

        encoded = self.roi_encoder(roi_patches)   # [N, out_ch, 1, 1]
        encoded = encoded.squeeze(-1).squeeze(-1) # [N, out_ch]
        return self.feature_proj(encoded)         # [N, out_ch]
