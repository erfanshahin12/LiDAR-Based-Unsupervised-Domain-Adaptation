# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.ops import box_np_ops


# ── Module-level per-object scaling core ────────────────────────────────────────
# Shared by RandomObjectScaling, NormalizeObjectSize, and the Mean-Teacher
# detector's target-side ROS. Operates on plain numpy arrays so it can be called
# without a transform instance (e.g. on raw points/boxes inside a detector).


def _bev_aabb(boxes: np.ndarray) -> np.ndarray:
    """Axis-aligned BEV bbox of each rotated box. (N, 4) xyxy."""
    corners = box_np_ops.center_to_corner_box3d(
        boxes[:, 0:3], boxes[:, 3:6], boxes[:, 6],
        origin=(0.5, 0.5, 0), axis=2)  # (N, 8, 3)
    xy = corners[:, :, :2]
    mins = xy.min(axis=1)
    maxs = xy.max(axis=1)
    return np.concatenate([mins, maxs], axis=1).astype(boxes.dtype)


def pick_isotropic_scale(boxes: np.ndarray, k: int,
                         candidates: np.ndarray) -> Optional[float]:
    """First candidate scale that produces no BEV collision with other boxes.

    Uses axis-aligned bounding boxes of each rotated footprint as a conservative
    collision proxy (AABB IoU of zero ⇒ rotated boxes definitely do not overlap).
    Returns ``None`` if every candidate collides.
    """
    n = boxes.shape[0]
    if n <= 1:
        return float(candidates[0])

    others_idx = np.arange(n) != k
    other_aabb = _bev_aabb(boxes[others_idx])  # (n-1, 4) xyxy

    x, y, _ = boxes[k, 0:3]
    l, w, _ = boxes[k, 3:6]
    yaw = float(boxes[k, 6])
    for r in candidates:
        cand = np.array([[x, y, 0.0, l * r, w * r, 1.0, yaw]], dtype=boxes.dtype)
        cand_aabb = _bev_aabb(cand)
        ious = box_np_ops.iou_jit(cand_aabb, other_aabb)
        if float(ious.max()) == 0.0:
            return float(r)
    return None


def scale_objects_in_place(boxes: np.ndarray, points: np.ndarray,
                           eligible: np.ndarray,
                           get_scale: Callable[[np.ndarray, int],
                                               Optional[np.ndarray]]):
    """Scale per-object points (box-local, anisotropic) and box dims.

    Args:
        boxes (np.ndarray): ``(N, 7+)`` ``[x, y, z, l, w, h, yaw, ...]``.
        points (np.ndarray): ``(M, C)`` with xyz in the first three columns.
        eligible (np.ndarray): ``(N,)`` bool mask of boxes to consider.
        get_scale: ``f(boxes, k) -> [s_l, s_w, s_h] | None``. Returns the
            per-axis scale in the box-local frame (x→length, y→width, z→height),
            or ``None`` to leave box ``k`` unchanged.

    Returns:
        tuple: ``(points, boxes)``. ``points`` may be a new array (rows dropped
        when a box is enlarged and swallows background points).
    """
    n = boxes.shape[0]
    for k in range(n):
        if not eligible[k]:
            continue

        s = get_scale(boxes, k)
        if s is None:
            continue
        s = np.asarray(s, dtype=boxes.dtype).reshape(3)

        center = boxes[k, 0:3].copy()
        lwh = boxes[k, 3:6].copy()
        yaw = float(boxes[k, 6])

        # Interior points using the original (pre-scale) box.
        pre_mask = box_np_ops.points_in_rbbox(
            points[:, :3], boxes[k:k + 1]).reshape(-1)
        obj_pts = points[pre_mask].copy()

        if obj_pts.shape[0] > 0:
            # Box-local frame: translate to origin, rotate by -yaw, scale per axis.
            obj_pts[:, :3] -= center
            rotated, _ = box_np_ops.rotation_points_single_angle(
                obj_pts[:, :3], -yaw, axis=2)
            rotated[:, 0] *= s[0]   # length axis
            rotated[:, 1] *= s[1]   # width axis
            rotated[:, 2] *= s[2]   # height axis
            back, _ = box_np_ops.rotation_points_single_angle(
                rotated, yaw, axis=2)
            obj_pts[:, :3] = back

        new_lwh = lwh * s
        new_center = center.copy()
        new_center[2] += (new_lwh[2] - lwh[2]) / 2.0   # keep grounded

        if obj_pts.shape[0] > 0:
            obj_pts[:, :3] += new_center
            points[pre_mask] = obj_pts

        boxes[k, 0:3] = new_center
        boxes[k, 3:6] = new_lwh

        # Enlarging along any axis: drop background points that became interior.
        if bool((new_lwh > lwh).any()):
            post_mask = box_np_ops.points_in_rbbox(
                points[:, :3], boxes[k:k + 1]).reshape(-1)
            keep = ~np.logical_xor(pre_mask, post_mask)
            points = points[keep]

    return points, boxes


def random_object_scale(boxes: np.ndarray, points: np.ndarray,
                        eligible: np.ndarray,
                        scale_range: Sequence[float],
                        num_try: int = 50):
    """Random Object Scaling (ROS): isotropic per-object scale with collision
    avoidance. Samples ``num_try`` candidate scalars per box from
    ``scale_range`` and applies the first that does not collide (BEV AABB) with
    any other box. Returns ``(points, boxes)``."""
    lo, hi = float(scale_range[0]), float(scale_range[1])
    cand = np.random.uniform(lo, hi, size=(boxes.shape[0], num_try))

    def get_scale(bx: np.ndarray, k: int) -> Optional[np.ndarray]:
        r = pick_isotropic_scale(bx, k, cand[k])
        if r is None:
            return None
        return np.array([r, r, r], dtype=bx.dtype)

    return scale_objects_in_place(boxes, points, eligible, get_scale)


@TRANSFORMS.register_module()
class RandomObjectScaling(BaseTransform):
    """Random Object Scaling (ROS).

    For each GT box, sample a scalar ``r`` uniformly from ``scale_range`` and
    scale the box's length/width/height by ``r`` (points inside the box are
    scaled with it). Up to ``num_try`` candidates are tried per box; the first
    whose scaled footprint has zero BEV AABB IoU with every other GT box is
    accepted. When ``r > 1`` the box grows and newly-interior background points
    are dropped; ``z`` is shifted by ``(new_h - old_h) / 2`` to stay grounded.

    Port of ST3D's ``scale_pre_object``.

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

        points_np, boxes_np = random_object_scale(
            boxes_np, points_np, eligible, self.scale_range, self.num_try)

        input_dict['points'] = points.new_point(points_np)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(boxes_np)
        return input_dict

    def _eligible_mask(self, input_dict: dict, n: int) -> np.ndarray:
        if self.class_names is None:
            return np.ones(n, dtype=bool)
        gt_labels = np.asarray(input_dict['gt_labels_3d'])
        all_classes = None
        if 'box_type_3d' in input_dict and 'metainfo' in input_dict:
            all_classes = input_dict['metainfo'].get('classes', None)
        if all_classes is None:
            # Fall back: assume label index already aligns with class_names
            # (true once ClassRemapWithLabel has filtered to self.class_names).
            return gt_labels >= 0
        keep_ids = {
            i for i, c in enumerate(all_classes) if c in self.class_names
        }
        return np.array([int(l) in keep_ids for l in gt_labels], dtype=bool)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'scale_range={self.scale_range}, '
                f'num_try={self.num_try}, '
                f'class_names={self.class_names})')
