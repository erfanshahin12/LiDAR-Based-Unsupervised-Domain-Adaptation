# Copyright (c) OpenMMLab. All rights reserved.
"""Ground-snap: align each box's bottom to the locally-estimated ground.

Association-free, label-free vertical refinement for LiDAR boxes. For each box,
the ground is estimated from a robust low percentile of the LiDAR points in the
box's own (slightly widened) BEV footprint, and the box is rigidly shifted in z
so its bottom sits at that ground (height and footprint unchanged). Because it
uses only the point cloud — never ground-truth — and both the points and the box
live in the same working frame, it removes per-box vertical-localization scatter
without leaking labels or touching any global z datum.

Shared by ``PseudoLabelRefreshHook`` (training pseudo-labels) and
``NusOnKittiMetric`` (eval predictions) so train and test stay consistent.
"""
import numpy as np

from mmdet3d.structures.ops.box_np_ops import points_in_rbbox


def _footprint_column_mask(box, points, margin):
    """Bool mask of ``points`` inside ``box``'s BEV footprint (+margin), any z."""
    col = box.copy()
    col[3] += margin  # widen length
    col[4] += margin  # widen width
    col[2] = -5.0     # bottom far below ground
    col[5] = 10.0     # tall column → capture the full vertical extent
    return points_in_rbbox(points[:, :3], col[None], z_axis=2,
                           origin=(0.5, 0.5, 0))[:, 0]


def ground_snap_boxes(boxes, points, pctl=2.0, min_pts=25, margin=0.3,
                      max_disp=0.0, car_label=None, labels=None):
    """Snap each box bottom to the local ground estimated from its footprint.

    Args:
        boxes (np.ndarray): (N, 7) ``[x, y, z_bottom, l, w, h, yaw]`` in the
            working LiDAR frame (z is bottom-center, mmdet3d convention).
        points (np.ndarray): (M, >=3) point cloud in the same frame.
        pctl (float): Percentile of in-footprint point z used as the ground
            estimate (low = closer to wheel/road contact). Default 2.0.
        min_pts (int): Minimum interior points required to snap a box; sparser
            boxes (far/occluded) are left unchanged. Default 25.
        margin (float): Footprint widening (m) added to l and w to catch road
            points at the car's edge. Default 0.3.
        max_disp (float): If > 0, clip the z shift to +/- this (m). The model's
            mean z is roughly right, so this rejects noisy far-box estimates.
            0 disables the cap (the validated default). Default 0.0.
        car_label (int | None): If given with ``labels``, only snap boxes whose
            label equals this value.
        labels (np.ndarray | None): (N,) per-box labels for ``car_label`` gating.

    Returns:
        tuple[np.ndarray, int]: (snapped boxes copy, number of boxes snapped).
    """
    out = boxes.astype(np.float32, copy=True)
    if len(out) == 0 or len(points) == 0:
        return out, 0
    eligible = np.ones(len(out), dtype=bool)
    if car_label is not None and labels is not None:
        eligible &= (np.asarray(labels) == car_label)

    n_snapped = 0
    for i in np.nonzero(eligible)[0]:
        z = points[_footprint_column_mask(out[i], points, margin), 2]
        if z.size < min_pts:
            continue
        ground = float(np.percentile(z, pctl))
        if max_disp > 0:
            ground = float(np.clip(ground, out[i, 2] - max_disp,
                                   out[i, 2] + max_disp))
        out[i, 2] = ground
        n_snapped += 1
    return out, n_snapped


def size_debias_boxes(boxes, ratios, car_label=None, labels=None):
    """Remove a systematic box-size offset by dividing l/w/h by fixed ratios.

    Multiplicative shrink toward the target-domain size prior: it corrects the
    constant source->target scale offset (e.g. nuScenes boxes are larger than
    KITTI) while *preserving* per-object size variation — unlike a clamp-to-prior.
    x/y/yaw and the bottom z are unchanged, so a height shrink lowers the top and
    keeps the (ground-snapped) base in place. Apply AFTER ``ground_snap_boxes``.

    Args:
        boxes (np.ndarray): (N, 7) ``[x, y, z_bottom, l, w, h, yaw]``.
        ratios (sequence[float]): ``[l_ratio, w_ratio, h_ratio]`` divisors
            (each = predicted/GT median for that dimension).
        car_label (int | None): with ``labels``, only de-bias this class.
        labels (np.ndarray | None): (N,) per-box labels for ``car_label`` gating.

    Returns:
        tuple[np.ndarray, int]: (de-biased boxes copy, number of boxes affected).
    """
    out = boxes.astype(np.float32, copy=True)
    if len(out) == 0:
        return out, 0
    eligible = np.ones(len(out), dtype=bool)
    if car_label is not None and labels is not None:
        eligible &= (np.asarray(labels) == car_label)
    rl, rw, rh = (float(r) for r in ratios)
    out[eligible, 3] /= rl   # length
    out[eligible, 4] /= rw   # width
    out[eligible, 5] /= rh   # height (bottom z fixed → top lowers)
    return out, int(eligible.sum())
