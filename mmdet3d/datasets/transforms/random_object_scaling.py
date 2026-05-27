# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.ops import box_np_ops


@TRANSFORMS.register_module()
class RandomObjectScaling(BaseTransform):
    """Random Object Scaling (ROS).

    For each GT box, sample a scalar ``r`` uniformly from ``scale_range``
    and scale the box's length/width/height by ``r``. Points inside the
    box are transformed to the box-local frame, scaled by ``r`` along
    all three axes, and transformed back.

    Up to ``num_try`` candidates are sampled per box; the first one whose
    scaled box has zero BEV IoU (axis-aligned AABB of the rotated
    footprint) with every other GT box is accepted. If all candidates
    conflict, the box is left unchanged.

    When ``r > 1`` the box grows, so background points that fall newly
    inside the enlarged box are dropped. The box center ``z`` is shifted
    by ``(new_h - old_h) / 2`` so the object stays grounded.

    Port of ST3D's ``scale_pre_object``
    (``pcdet/datasets/augmentor/augmentor_utils.py``).

    Required Keys: ``points``, ``gt_bboxes_3d``, ``gt_labels_3d``
    Modified Keys: ``points``, ``gt_bboxes_3d``
    """

    def __init__(self,
                 scale_range: Sequence[float] = (0.75, 1.0),
                 num_try: int = 50,
                 class_names: Optional[List[str]] = None) -> None:
        assert len(scale_range) == 2 and scale_range[0] <= scale_range[1]
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.num_try = int(num_try)
        self.class_names = list(class_names) if class_names is not None else None

    def transform(self, input_dict: dict) -> dict:
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']

        if len(gt_bboxes_3d) == 0:
            return input_dict

        boxes_np = gt_bboxes_3d.tensor.numpy().copy()
        points_np = points.tensor.numpy().copy()

        eligible = self._eligible_mask(input_dict, n=boxes_np.shape[0])
        if not eligible.any():
            return input_dict

        points_np, boxes_np = self._scale_per_object(
            boxes_np, points_np, eligible)

        input_dict['points'] = points.new_point(points_np)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(boxes_np)
        return input_dict

    def _eligible_mask(self, input_dict: dict, n: int) -> np.ndarray:
        if self.class_names is None:
            return np.ones(n, dtype=bool)
        gt_labels = np.asarray(input_dict['gt_labels_3d'])
        # Resolve label index -> name through dataset metainfo if available
        all_classes = None
        if 'box_type_3d' in input_dict and 'metainfo' in input_dict:
            all_classes = input_dict['metainfo'].get('classes', None)
        if all_classes is None:
            # Fall back: assume label index already aligns with class_names.
            # This is enough when ClassRemapWithLabel has already filtered
            # boxes down to self.class_names.
            return gt_labels >= 0
        keep_ids = {
            i for i, c in enumerate(all_classes) if c in self.class_names
        }
        return np.array([int(l) in keep_ids for l in gt_labels], dtype=bool)

    def _scale_per_object(self, boxes: np.ndarray, points: np.ndarray,
                          eligible: np.ndarray):
        """In-place scale of points; returns (points, boxes).

        ``boxes`` is (N, 7+) with columns [x, y, z, l, w, h, yaw, ...].
        ``points`` is (M, C) with xyz in the first three columns.
        """
        n = boxes.shape[0]
        lo, hi = self.scale_range
        scales = np.random.uniform(lo, hi, size=(n, self.num_try))

        for k in range(n):
            if not eligible[k]:
                continue

            r = self._pick_scale(boxes, k, scales[k])
            if r is None:
                continue

            center = boxes[k, 0:3].copy()
            lwh = boxes[k, 3:6].copy()
            yaw = float(boxes[k, 6])

            # Interior point mask using the original (pre-scale) box.
            pre_mask = box_np_ops.points_in_rbbox(
                points[:, :3], boxes[k:k + 1]).reshape(-1)
            obj_pts = points[pre_mask].copy()

            if obj_pts.shape[0] > 0:
                # Local frame: translate to origin, rotate by -yaw.
                obj_pts[:, :3] -= center
                rotated, _ = box_np_ops.rotation_points_single_angle(
                    obj_pts[:, :3], -yaw, axis=2)
                rotated *= r
                # Back to scene frame.
                back, _ = box_np_ops.rotation_points_single_angle(
                    rotated, yaw, axis=2)
                obj_pts[:, :3] = back

            # New box dims + grounded z shift.
            new_lwh = lwh * r
            new_center = center.copy()
            new_center[2] += (new_lwh[2] - lwh[2]) / 2.0

            if obj_pts.shape[0] > 0:
                obj_pts[:, :3] += new_center
                points[pre_mask] = obj_pts

            boxes[k, 0:3] = new_center
            boxes[k, 3:6] = new_lwh

            # Enlarging: drop background points that became interior.
            if r > 1.0:
                post_mask = box_np_ops.points_in_rbbox(
                    points[:, :3], boxes[k:k + 1]).reshape(-1)
                # Keep points that did NOT change membership.
                keep = ~np.logical_xor(pre_mask, post_mask)
                points = points[keep]

        return points, boxes

    def _pick_scale(self, boxes: np.ndarray, k: int,
                    candidates: np.ndarray) -> Optional[float]:
        """Return the first candidate scale that produces no BEV
        collision with other GT boxes, or ``None`` if all collide.

        Uses axis-aligned bounding boxes of each rotated footprint as a
        conservative collision proxy: AABB IoU of zero implies the
        rotated boxes definitely do not overlap.
        """
        n = boxes.shape[0]
        if n <= 1:
            return float(candidates[0])

        others_idx = np.arange(n) != k
        other_boxes = boxes[others_idx]
        other_aabb = self._bev_aabb(other_boxes)  # (n-1, 4) xyxy

        x, y, _ = boxes[k, 0:3]
        l, w, _ = boxes[k, 3:6]
        yaw = float(boxes[k, 6])
        for r in candidates:
            cand = np.array([[x, y, 0.0, l * r, w * r, 1.0, yaw]],
                            dtype=boxes.dtype)
            cand_aabb = self._bev_aabb(cand)
            ious = box_np_ops.iou_jit(cand_aabb, other_aabb)
            if float(ious.max()) == 0.0:
                return float(r)
        return None

    @staticmethod
    def _bev_aabb(boxes: np.ndarray) -> np.ndarray:
        """Axis-aligned BEV bbox of each rotated box. (N, 4) xyxy."""
        # center_to_corner_box3d wants (centers, dims, angles); use
        # bottom-center origin (mmdet3d LiDAR convention).
        corners = box_np_ops.center_to_corner_box3d(
            boxes[:, 0:3], boxes[:, 3:6], boxes[:, 6],
            origin=(0.5, 0.5, 0), axis=2)  # (N, 8, 3)
        xy = corners[:, :, :2]
        mins = xy.min(axis=1)
        maxs = xy.max(axis=1)
        return np.concatenate([mins, maxs], axis=1).astype(boxes.dtype)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'scale_range={self.scale_range}, '
                f'num_try={self.num_try}, '
                f'class_names={self.class_names})')
