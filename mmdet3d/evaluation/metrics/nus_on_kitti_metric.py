from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes

from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric


@METRICS.register_module()
class NusOnKittiMetric(KittiMetric):
    """Evaluate a nuScenes-trained model on KITTI data using KITTI IoU AP.

    The test pipeline applies KittiToNuscenes to rotate input points into the
    nuScenes LiDAR frame so a nuScenes-trained model sees familiar data.
    Predicted boxes therefore come out in nuScenes LiDAR frame; this metric
    undoes that rotation before passing them to the standard KittiMetric
    pipeline, which expects KITTI LiDAR frame.

    GT loading, camera projection, range filtering, and kitti_eval invocation
    are all inherited unchanged from KittiMetric.

    Args:
        label_mapping (dict[int, int] | None): Maps nuScenes output label
            indices to KITTI dataset class indices.  Predictions whose label
            is absent from the mapping are dropped silently.  Example:
            ``{0: 2}`` maps nuScenes 'car' (label 0 with num_classes=1) to
            KITTI 'Car' (index 2 in ``['Pedestrian', 'Cyclist', 'Car']``).
            Defaults to None (no remapping or dropping).
    """

    def __init__(
            self,
            ann_file: str,
            metric: Union[str, List[str]] = 'bbox',
            pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
            prefix: Optional[str] = None,
            pklfile_prefix: Optional[str] = None,
            default_cam_key: str = 'CAM2',
            format_only: bool = False,
            submission_prefix: Optional[str] = None,
            label_mapping: Optional[Dict[int, int]] = None,
            collect_device: str = 'cpu',
            backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'NusOnKitti metric'
        super().__init__(
            ann_file=ann_file,
            metric=metric,
            pcd_limit_range=pcd_limit_range,
            prefix=prefix,
            pklfile_prefix=pklfile_prefix,
            default_cam_key=default_cam_key,
            format_only=format_only,
            submission_prefix=submission_prefix,
            collect_device=collect_device,
            backend_args=backend_args)
        self.label_mapping = label_mapping or {}

    @staticmethod
    def _nus_to_kitti_boxes(
            bboxes_3d: LiDARInstance3DBoxes) -> LiDARInstance3DBoxes:
        """Inverse of KittiToNuscenes: rotate boxes back to KITTI LiDAR frame.

        KittiToNuscenes applies:
            x_nus = -y_kitti,  y_nus = x_kitti
            yaw_nus = yaw_kitti + π/2
            l, w unchanged (box-local dims are frame-independent)
            z_nus = z_kitti - 0.11 + h/2  (bottom-center → gravity-center + sensor alignment, GT only)

        Model predictions use the standard mmdet3d bottom-center convention
        (origin=(0.5, 0.5, 0)), so no z-shift is applied here.

        Inverse:
            x_kitti =  y_nus
            y_kitti = -x_nus
            yaw_kitti = yaw_nus - π/2
            l, w unchanged
        """
        t = bboxes_3d.tensor.clone()
        x = t[:, 0].clone()
        y = t[:, 1].clone()
        t[:, 0] = y             # x_kitti = y_nus
        t[:, 1] = -x            # y_kitti = -x_nus
        t[:, 2] += 0.11         # undo sensor height alignment: nuScenes z → KITTI z
        # l, w (indices 3, 4) unchanged
        t[:, 6] = t[:, 6] - np.pi / 2  # yaw_kitti = yaw_nus - π/2
        return LiDARInstance3DBoxes(t, box_dim=7, origin=(0.5, 0.5, 0))

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Inverse-rotate predicted boxes to KITTI frame, remap labels, then
        delegate to KittiMetric.process for accumulation."""
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']

            if len(pred_3d['bboxes_3d']) > 0:
                pred_3d['bboxes_3d'] = self._nus_to_kitti_boxes(
                    pred_3d['bboxes_3d'])

            if self.label_mapping and len(pred_3d['labels_3d']) > 0:
                labels = pred_3d['labels_3d']
                keep = torch.tensor(
                    [l.item() in self.label_mapping for l in labels],
                    dtype=torch.bool)
                pred_3d['bboxes_3d'] = pred_3d['bboxes_3d'][keep]
                pred_3d['scores_3d'] = pred_3d['scores_3d'][keep]
                pred_3d['labels_3d'] = torch.tensor(
                    [self.label_mapping[l.item()] for l in labels[keep]],
                    dtype=labels.dtype)

        super().process(data_batch, data_samples)
