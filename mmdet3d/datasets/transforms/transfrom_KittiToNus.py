import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes

@TRANSFORMS.register_module()
class KittiToNuscenes(BaseTransform):
    """
    Convert KITTI LiDAR coordinates to nuScenes LiDAR coordinates.

    KITTI:     +X forward, +Y left, +Z up
    nuScenes:  +X right,  +Y forward, +Z up

    Transform:
        x_nus = -y_kitti
        y_nus =  x_kitti
        z_nus =  z_kitti
        yaw_nus = yaw_kitti + π/2
        (swap l ↔ w)
    """
    def transform(self, results: dict) -> dict:
        # --- Transform point cloud ---
        if 'points' in results:
            rot_mat = torch.tensor([
                [0, -1, 0, 0],
                [1,  0, 0, 0],
                [0,  0, 1, 0],
                [0,  0, 0, 1]
            ], dtype=results['points'].tensor.dtype, device=results['points'].tensor.device)
            results['points'].tensor = results['points'].tensor @ rot_mat.T

        # --- Transform 3D boxes ---
        if 'gt_bboxes_3d' in results:
            boxes = results['gt_bboxes_3d'].tensor.clone()

            # Coordinate transform
            x, y, z, l, w, h, yaw = [boxes[:, i] for i in range(7)]
            boxes[:, 0], boxes[:, 1], boxes[:, 2] = -y, x, z
            boxes[:, 3], boxes[:, 4], boxes[:, 5] = w, l, h
            boxes[:, 6] = torch.atan2(torch.sin(yaw + np.pi / 2), torch.cos(yaw + np.pi / 2))

            results['gt_bboxes_3d'] = LiDARInstance3DBoxes(boxes, box_dim=7, origin=(0.5, 0.5, 0))

        return results
