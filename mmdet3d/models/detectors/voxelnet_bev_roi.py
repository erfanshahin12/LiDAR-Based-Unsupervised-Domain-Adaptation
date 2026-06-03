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
    produced entirely by the per-anchor ``conv_iou`` branch in
    ``Anchor3DHead`` (when ``predict_iou=True``).  The old post-NMS
    ``BEVRoIIoUHead`` has been removed.
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
                    or return_roi_features or self.return_roi_features)
        if need_bev:
            x, bev_features = self.extract_feat(
                batch_inputs_dict, return_bev=True)
        else:
            x = self.extract_feat(batch_inputs_dict, return_bev=False)
            bev_features = None

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

        BEV features are extracted only when the contrastive/consistency
        loss in an outer wrapper (e.g. MeanTeacher) requests them via
        ``return_bev=True``.  In standalone pretrain the BEV extraction path
        is skipped entirely.
        """
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

    def boxes_to_bev_pixels(self, boxes_3d, bev_h, bev_w):
        """Convert 3D boxes to BEV pixel coordinates.

        Args:
            boxes_3d: LiDARInstance3DBoxes or tensor [N, 7+].
            bev_h, bev_w: BEV feature map dimensions.

        Returns:
            centers_px (Tensor): [N, 2] pixel centers.
            dims_px (Tensor): [N, 2] pixel dimensions.
        """
        if isinstance(boxes_3d, LiDARInstance3DBoxes):
            boxes_tensor = boxes_3d.tensor
        else:
            boxes_tensor = boxes_3d

        centers_world = boxes_tensor[:, :2]
        dims_world = boxes_tensor[:, 3:5]

        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        x_max, y_max = self.point_cloud_range[3], self.point_cloud_range[4]

        centers_norm = torch.zeros_like(centers_world)
        centers_norm[:, 0] = (centers_world[:, 0] - x_min) / (x_max - x_min)
        centers_norm[:, 1] = (centers_world[:, 1] - y_min) / (y_max - y_min)

        centers_px = torch.zeros_like(centers_norm)
        centers_px[:, 0] = centers_norm[:, 0] * bev_w
        centers_px[:, 1] = centers_norm[:, 1] * bev_h

        dims_px = torch.zeros_like(dims_world)
        dims_px[:, 0] = dims_world[:, 0] / (x_max - x_min) * bev_w
        dims_px[:, 1] = dims_world[:, 1] / (y_max - y_min) * bev_h

        return centers_px, dims_px

    def extract_roi_features(self, bev_features, boxes_3d):
        """Extract features for each detected box from the BEV feature map.

        Args:
            bev_features (Tensor): BEV feature map [C, H, W].
            boxes_3d: Detected 3D boxes [N, 7] or LiDARInstance3DBoxes.

        Returns:
            Tensor: Per-box features [N, out_channels].
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
        centers_px, dims_px = self.boxes_to_bev_pixels(boxes_3d, H, W)

        roi_features_list = []
        for i in range(num_boxes):
            cx, cy = centers_px[i]
            dx, dy = dims_px[i]

            padding = 2.0
            x_min = int(torch.clamp(cx - dx / 2 - padding, 0, W - 1).item())
            x_max = int(torch.clamp(cx + dx / 2 + padding, 1, W).item())
            y_min = int(torch.clamp(cy - dy / 2 - padding, 0, H - 1).item())
            y_max = int(torch.clamp(cy + dy / 2 + padding, 1, H).item())

            if x_max <= x_min or y_max <= y_min:
                x_min = max(0, int(cx.item()) - 1)
                x_max = min(W, int(cx.item()) + 1)
                y_min = max(0, int(cy.item()) - 1)
                y_max = min(H, int(cy.item()) + 1)

            roi_feat = bev_features[:, y_min:y_max, x_min:x_max]

            if roi_feat.numel() == 0:
                roi_feat = torch.zeros((C, 1, 1), device=device)

            roi_feat = F.interpolate(
                roi_feat.unsqueeze(0),
                size=(self.roi_size, self.roi_size),
                mode='bilinear',
                align_corners=False,
            )
            roi_features_list.append(roi_feat)

        roi_features_batch = torch.cat(roi_features_list, dim=0)
        encoded_features = self.roi_encoder(roi_features_batch)
        encoded_features = encoded_features.squeeze(-1).squeeze(-1)
        return self.feature_proj(encoded_features)
