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
        yaw_nus = yaw_kitti + π/2
        l, w unchanged (box-local dimensions are frame-independent)
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
            boxes = results['gt_bboxes_3d'].tensor.numpy()

            # Coordinate transform — use .copy() to avoid numpy view aliasing:
            # boxes[:, i] returns a view; assigning to one column would corrupt
            # another variable that is a view of the same column.
            x, y, z, l, w, h, yaw = [boxes[:, i].copy() for i in range(7)]
            boxes[:, 0] = -y
            boxes[:, 1] = x
            boxes[:, 2] = z + h / 2.0        # bottom-center → gravity-center
            # l, w (indices 3, 4) unchanged: box-local dims don't change with coord rotation
            boxes[:, 5] = h
            boxes[:, 6] = np.arctan2(np.sin(yaw + np.pi / 2), np.cos(yaw + np.pi / 2))

            results['gt_bboxes_3d'] = LiDARInstance3DBoxes(torch.from_numpy(boxes),
                                                           box_dim=7,
                                                           origin=(0.5, 0.5, 0.5))    # Nuscenes boxes' origin

        return results
