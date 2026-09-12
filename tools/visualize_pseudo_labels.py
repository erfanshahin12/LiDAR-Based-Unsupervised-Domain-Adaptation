#!/usr/bin/env python3
"""
Visualize pseudo-labels vs KITTI GT vs baseline predictions (Bird's-Eye-View PNG).

All boxes are brought into the nuScenes coordinate frame — the same frame the
teacher operates in:
  - Points:        KITTI velodyne_reduced bin loaded and manually rotated to nuScenes
  - GT:            KITTI camera boxes → KITTI LiDAR (via lidar2cam) → nuScenes
                   (via KittiToNuscenes), mirroring the val_pipeline in the MT config
  - Pseudo-labels: used as-is (stored in nuScenes frame by PseudoLabelRefreshHook)
  - Baseline:      init_model on the pretrain checkpoint, inference on nuScenes-frame
                   points using the same data_preprocessor → predict pattern as the
                   PseudoLabelRefreshHook

NOTE: faithful pseudo-label overlay assumes the weak target pipeline in the MT config
has RandomFlip3D / GlobalRotScaleTrans commented out.  If those are active the script
will warn and overlay will be incorrect.

Box z convention: LiDARInstance3DBoxes always stores z at bottom-center internally.
The side-view panel uses box[2] as the box bottom edge on the z-axis.

Usage (from ~/mmdetection3d/):
    python tools/visualize_pseudo_labels.py \\
        --ps-label-pkl work_dirs/mt_pointpillars_20may/20260520_072924/ps_labels/ps_label_e0.pkl \\
        --baseline-config work_dirs/baseline_pointpillars_5may/mt_pretrain_pointpillars_config.py \\
        --baseline-ckpt   work_dirs/baseline_pointpillars_5may/epoch_24.pth \\
        --num-scenes 6 --out-dir vis_ps_labels

    # GT vs pseudo only (no GPU required):
    python tools/visualize_pseudo_labels.py --ps-label-pkl ps_label_e0.pkl --no-baseline --num-scenes 10

    # Numerical stats only (no PNGs):
    python tools/visualize_pseudo_labels.py --ps-label-pkl ps_label_e0.pkl \\
        --baseline-config ... --baseline-ckpt ... --stats-only --num-scenes 100
"""

import argparse
import os
import pickle
import random
import sys

import matplotlib
matplotlib.use('Agg')  # headless — must be set before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

# ── MMDet3D imports ────────────────────────────────────────────────────────────
from mmengine.config import Config
from mmdet3d.structures import (
    Box3DMode,
    CameraInstance3DBoxes,
    Det3DDataSample,
    LiDARInstance3DBoxes,
)
from mmcv.ops import box_iou_rotated
from mmdet3d.datasets.transforms.transform_KittiToNus import KittiToNuscenes

KITTI_CAT = {
    0: 'Pedestrian', 1: 'Cyclist', 2: 'Car',
    3: 'Van', 4: 'Truck', 5: 'Person_sitting',
    6: 'Tram', 7: 'Misc', -1: 'DontCare',
}


# ── BEV IoU helpers ────────────────────────────────────────────────────────────

def gt_iou_per_pred(pred_boxes: np.ndarray, gt_boxes: np.ndarray,
                    mode: str = '3d') -> np.ndarray:
    """Best IoU of each prediction against any GT box.

    Args:
        pred_boxes: (K, 7) float32 predicted boxes in nuScenes frame.
        gt_boxes:   (M, 7) float32 GT Car boxes in nuScenes frame.
        mode:       '3d' → full 3D IoU via LiDARInstance3DBoxes.overlaps;
                    'bev' → BEV IoU via bev_iou_matrix.

    Returns:
        (K,) float32 — max IoU of each prediction vs any GT box; 0 when no GT.
    """
    if len(pred_boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(gt_boxes) == 0:
        return np.zeros(len(pred_boxes), dtype=np.float32)
    if mode == 'bev':
        iou = bev_iou_matrix(pred_boxes, gt_boxes)           # (K, M)
    else:
        p = LiDARInstance3DBoxes(torch.from_numpy(pred_boxes))
        g = LiDARInstance3DBoxes(torch.from_numpy(gt_boxes))
        iou = LiDARInstance3DBoxes.overlaps(p, g).cpu().numpy()  # (K, M)
    return iou.max(axis=1).astype(np.float32)


def bev_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """BEV IoU between every pair of boxes in a (M,7) and b (N,7).
    Returns (M, N) float32; zeros if either array is empty."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    boxes_a = LiDARInstance3DBoxes(torch.from_numpy(a))
    boxes_b = LiDARInstance3DBoxes(torch.from_numpy(b))

    boxes1_bev, boxes2_bev = boxes_a.bev, boxes_b.bev
    boxes1_bev[:, 2:4] = boxes1_bev[:, 2:4].clamp(min=1e-4)
    boxes2_bev[:, 2:4] = boxes2_bev[:, 2:4].clamp(min=1e-4)

    # bev overlap
    iou2d = box_iou_rotated(boxes1_bev, boxes2_bev).cpu().numpy()
    return iou2d


def count_points_in_boxes(pts_nus: np.ndarray, ps_boxes: np.ndarray) -> np.ndarray:
    """Count LiDAR points inside each pseudo-label box (CPU, pure numpy).

    Args:
        pts_nus: (M, 4) float32 point cloud in nuScenes frame [x,y,z,intensity].
        ps_boxes: (K, 7) float32 boxes in nuScenes frame [x,y,z,l,w,h,yaw],
                  z is bottom-center (LiDARInstance3DBoxes internal convention).

    Returns:
        (K,) int32 array — number of points inside each box.
    """
    if len(ps_boxes) == 0 or len(pts_nus) == 0:
        return np.zeros(len(ps_boxes), dtype=np.int32)
    from mmdet3d.structures.ops.box_np_ops import points_in_rbbox
    # points_in_rbbox: (N_pts, K_boxes) bool; origin=(0.5,0.5,0) matches
    # LiDARInstance3DBoxes convention where z is stored at bottom-center.
    in_box = points_in_rbbox(pts_nus[:, :3], ps_boxes, z_axis=2,
                              origin=(0.5, 0.5, 0))
    return in_box.sum(axis=0).astype(np.int32)


def select_scene_keep_mask(scores: np.ndarray,
                           keep_frac: float | None = None,
                           min_floor: float = 0.0,
                           fallback_k: int = 0,
                           floor_before: bool = False) -> np.ndarray:
    """Per-scene keep-fraction selection. Returns a boolean mask over ``scores``.

    The three-step scheme:
      1. **Keep-fraction (primary):** keep the top ``ceil(keep_frac × N)``
         highest-scoring boxes, where N is the candidate pool size (all boxes,
         or above-floor boxes when ``floor_before=True``).
         When ``keep_frac`` is None or ≥ 1 this step is skipped.
      2. **Floor (guard):** drop surviving boxes with ``score < min_floor``.
         Skipped when ``min_floor <= 0``.
      3. **Fallback (coverage):** if fewer than ``fallback_k`` boxes remain,
         restore the scene's top-``min(fallback_k, N)`` raw boxes by score,
         ignoring ``min_floor``.  ``fallback_k=0`` disables the fallback.

    The default ordering is fraction-first, floor-as-guard (``floor_before=False``).
    Set ``floor_before=True`` to apply the floor before computing the fraction
    (the fraction denominator then excludes below-floor boxes).

    Args:
        scores:      1-D float array of per-box ranking scores (higher = better).
        keep_frac:   Fraction in (0, 1) of boxes to keep.  None / ≥1 = no cut.
        min_floor:   Absolute score floor.  0.0 = off.
        fallback_k:  Minimum surviving boxes per scene.  0 = off.
        floor_before: Order flag — see docstring above.

    Returns:
        Boolean keep-mask of shape ``(len(scores),)``.
    """
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=bool)

    mask = np.ones(n, dtype=bool)            # start: keep everything
    order = np.argsort(scores)[::-1]         # indices sorted best → worst

    use_frac = keep_frac is not None and keep_frac < 1.0

    if floor_before:
        # Order A: floor first, then fraction over the surviving pool
        if min_floor > 0:
            mask = scores >= min_floor
        if use_frac:
            n_pool = int(mask.sum())
            keep_n = int(np.ceil(keep_frac * n_pool)) if n_pool > 0 else 0
            # Among the above-floor candidates, keep the top keep_n
            above_floor_ordered = [i for i in order if mask[i]]
            new_mask = np.zeros(n, dtype=bool)
            for i in above_floor_ordered[:keep_n]:
                new_mask[i] = True
            mask = new_mask
    else:
        # Order B (default): fraction first, then floor as guard
        if use_frac:
            keep_n = int(np.ceil(keep_frac * n))
            mask = np.zeros(n, dtype=bool)
            for i in order[:keep_n]:
                mask[i] = True
        if min_floor > 0:
            mask = mask & (scores >= min_floor)

    # Fallback: restore top-k raw boxes if coverage is lost
    if fallback_k > 0 and int(mask.sum()) < fallback_k:
        restore_n = min(fallback_k, n)
        mask = np.zeros(n, dtype=bool)
        for i in order[:restore_n]:
            mask[i] = True

    return mask


def match_boxes(pred: np.ndarray, gt: np.ndarray,
                iou_thr: float) -> tuple[int, int, int]:
    """Greedy 3D IoU matching. Returns (TP, FP, FN)."""
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    iou = LiDARInstance3DBoxes.overlaps(
        LiDARInstance3DBoxes(torch.from_numpy(pred)),
        LiDARInstance3DBoxes(torch.from_numpy(gt))).cpu().numpy()
    matched_pred: set = set()
    matched_gt: set = set()
    flat_idx = np.argsort(iou.ravel())[::-1]
    for idx in flat_idx:
        i, j = divmod(int(idx), iou.shape[1])
        if iou[i, j] < iou_thr:
            break
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
    tp = len(matched_pred)
    return tp, len(pred) - tp, len(gt) - tp


# ── Ground-truth helpers ───────────────────────────────────────────────────────

def build_gt_lookup(info_path: str) -> tuple[dict, int]:
    """Return ({scene_id → dict}, gt_car_label) from kitti_infos_train.pkl.

    scene_id is the stem of the lidar_path filename, e.g. '000042'.
    Each value: {'cam_boxes': (N,7) float32, 'labels': (N,) int64,
                 'lidar2cam': (4,4) float64}
    gt_car_label is the integer label for 'Car' read from the pkl's metainfo.
    """
    with open(info_path, 'rb') as f:
        d = pickle.load(f)
    categories = d.get('metainfo', {}).get('categories', {})
    gt_car_label = categories.get('Car', 2)  # 2 is the standard MMDet3D KITTI value
    print(f'GT Car label = {gt_car_label}  (from {info_path} metainfo)')
    lookup = {}
    for info in d['data_list']:
        basename = os.path.splitext(info['lidar_points']['lidar_path'])[0]
        lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])
        instances = info.get('instances', [])
        if instances:
            cam_boxes = np.array(
                [inst['bbox_3d'] for inst in instances], dtype=np.float32)
            labels = np.array(
                [inst['bbox_label_3d'] for inst in instances], dtype=np.int64)
        else:
            cam_boxes = np.zeros((0, 7), dtype=np.float32)
            labels = np.zeros(0, dtype=np.int64)
        lookup[basename] = {
            'cam_boxes': cam_boxes,
            'labels': labels,
            'lidar2cam': lidar2cam,
        }
    return lookup, gt_car_label


def gt_boxes_to_nus(cam_boxes: np.ndarray, lidar2cam: np.ndarray) -> np.ndarray:
    """Convert GT camera boxes to nuScenes frame.

    Mirrors KittiDataset.parse_ann_info + val_pipeline KittiToNuscenes step.
    Returns (N, 7) array; tensor z is bottom-center (LiDARInstance3DBoxes internal).
    """
    if len(cam_boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32)
    gt_lidar = CameraInstance3DBoxes(cam_boxes).convert_to(
        Box3DMode.LIDAR, np.linalg.inv(lidar2cam))
    results = {'gt_bboxes_3d': gt_lidar}
    KittiToNuscenes().transform(results)
    return results['gt_bboxes_3d'].tensor.numpy()


# ── Point-cloud helpers ────────────────────────────────────────────────────────

def load_points_nus(bin_path: str) -> np.ndarray:
    """Load KITTI velodyne_reduced bin and rotate to nuScenes frame.

    KITTI: +X forward, +Y left, +Z up
    nuScenes: +X right, +Y forward, +Z up  →  x_n = -y_k, y_n = x_k

    Also subtracts 0.11 m from z to align KITTI ground (−1.73 m, sensor 1.73 m
    above ground) to nuScenes ground (−1.84 m, sensor 1.84 m above ground),
    mirroring KittiToNuscenes.transform applied in the training pipeline.

    Intensity is also scaled [0,1] → [0,255] to match nuScenes convention.
    Returns (M, 4) float32 [x, y, z, intensity] in nuScenes frame.
    """
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pts_nus = np.empty_like(pts)
    pts_nus[:, 0] = -pts[:, 1]          # x_nus = -y_kitti
    pts_nus[:, 1] = pts[:, 0]           # y_nus =  x_kitti
    pts_nus[:, 2] = pts[:, 2] - 0.29    # align KITTI ground (−1.73) to nuScenes (−1.84)
    pts_nus[:, 3] = pts[:, 3] * 255.0
    return pts_nus


# ── Baseline inference ─────────────────────────────────────────────────────────

def _resolve_bbox_head(model):
    """Return the bbox_head used during predict().

    VoxelNet/VoxelNetBEVRoI expose it at the top level as bbox_head;
    CenterPoint / CenterPointBEVRoI use pts_bbox_head;
    MeanTeacher3DDetector nests it under .teacher (used by predict() by default).
    """
    head = getattr(model, 'bbox_head', None)
    if head is None:
        head = getattr(model, 'pts_bbox_head', None)   # CenterPoint family
    if head is None:
        # MeanTeacher3DDetector: predict() uses teacher by default
        head = model.teacher.bbox_head
    return head


def _apply_kitti_test_cfg(head, overrides: dict) -> dict:
    """Override test_cfg keys and return originals for later restoration.

    CenterHead does not have 'score_type' in its test_cfg (it uses a different
    key layout: nms_type, post_max_size, min_radius, …).  In that case we leave
    the test_cfg completely untouched — the loaded KITTI test config already
    contains the correct CenterPoint settings.  An empty saved dict is returned
    so _restore_test_cfg is a no-op.
    """
    if 'score_type' not in head.test_cfg:
        # CenterHead or other non-VoxelNet head — do not apply VoxelNet overrides
        return {}
    saved = {k: head.test_cfg.get(k) for k in overrides}
    for k, v in overrides.items():
        head.test_cfg[k] = v
    return saved


def _restore_test_cfg(head, saved: dict) -> None:
    for k, v in saved.items():
        if v is None:
            head.test_cfg.pop(k, None)
        else:
            head.test_cfg[k] = v


# KITTI test-config inference settings (test_kitti_nuspretrained_pointpillars.py).
# Applied to every baseline inference call so the box pool is identical to
# what tools/test.py would produce on KITTI.
_KITTI_TEST_CFG = dict(
    score_type='cls',                        # scores_3d = raw sigmoid CLS
    score_weights=dict(iou=0.0, cls=1.0),    # consistent with score_type='cls'
    nms_thr=0.01,                            # tight — suppress duplicate FPs
    score_thr=0.05,
    nms_pre=1000,
    max_num=200,
)


def run_baseline(model, pts_nus: np.ndarray, device: str,
                 keep_frac: float | None = None,
                 min_floor: float = 0.0,
                 fallback_k: int = 1,
                 floor_before: bool = False,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray | None, np.ndarray]:
    """Run baseline on a single scene and apply per-scene keep-fraction filtering.

    The test_cfg is overridden with _KITTI_TEST_CFG (matching
    test_kitti_nuspretrained_pointpillars.py) so that inference is consistent
    with KITTI evaluation regardless of which training config was loaded.
    ``score_type='cls'`` guarantees ``scores_3d`` is the raw classification
    sigmoid, which is the score used for keep_frac ranking and the floor.

    Filtering: keep the top ``keep_frac`` of boxes per scene by CLS score
    (None = keep all), then drop survivors below ``min_floor`` (see
    ``select_scene_keep_mask``).

    Returns (boxes (K,7), cls (K,), labels (K,), iou_scores (K,)|None,
             cls_scores (K,)) in nuScenes frame.
    """
    pts_tensor = torch.from_numpy(pts_nus).float()
    sample = Det3DDataSample()
    sample.set_metainfo({
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR,
    })

    head = _resolve_bbox_head(model)
    saved = _apply_kitti_test_cfg(head, _KITTI_TEST_CFG)
    try:
        with torch.no_grad():
            data = model.data_preprocessor(
                {'inputs': {'points': [pts_tensor]},
                 'data_samples': [sample]},
                training=False)
            pred_list = model.predict(data['inputs'], data['data_samples'])
    finally:
        _restore_test_cfg(head, saved)

    pred = pred_list[0]
    inst = pred.pred_instances_3d
    # score_type='cls' guarantees scores_3d is raw CLS.
    cls_scores = inst.scores_3d.cpu().numpy()
    boxes  = inst.bboxes_3d.tensor.cpu().numpy()[:, :7]
    labels = inst.labels_3d.cpu().numpy()
    iou_sc = getattr(inst, 'iou_scores_3d', None)
    # iou_scores_3d is the raw sigmoid of conv_iou (always separate from scores_3d).
    iou_scores = iou_sc.cpu().numpy() if iou_sc is not None else None

    return apply_score_filter(
        boxes, cls_scores, iou_scores, labels,
        keep_frac=keep_frac, min_floor=min_floor,
        fallback_k=fallback_k, floor_before=floor_before)


def apply_score_filter(boxes, cls_scores, iou_scores, labels, *,
                       keep_frac=None, min_floor=0.0,
                       fallback_k=0, floor_before=False):
    """Apply the per-scene keep-fraction + floor filter to a raw prediction pool.

    Shared by ``run_baseline`` (inference path) and the ``--floor-percentile``
    pre-pass (cached-prediction path) so both apply identical masking. Ranking
    and floor use the raw CLS score. ``min_floor`` is an ABSOLUTE score value;
    the percentile→floor conversion is done by the caller. Returns
    (boxes, cls, labels, iou_scores|None, cls_scores) filtered.
    """
    mask = select_scene_keep_mask(cls_scores,
                                  keep_frac=keep_frac,
                                  min_floor=min_floor,
                                  fallback_k=fallback_k,
                                  floor_before=floor_before)

    return (boxes[mask], cls_scores[mask], labels[mask],
            iou_scores[mask] if iou_scores is not None else None,
            cls_scores[mask])


def run_baseline_raw(model, pts_nus: np.ndarray, device: str,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Return all post-NMS boxes with no additional threshold filtering.

    Applies the same _KITTI_TEST_CFG overrides as run_baseline so the box pool
    is consistent (score_type='cls', nms_thr=0.01, etc.).
    Returns (boxes (K,7), cls_scores (K,), iou_scores (K,)|None, labels (K,)).
    Used by --dump-raw-preds to build a cache for threshold sweeping.
    """
    pts_tensor = torch.from_numpy(pts_nus).float()
    sample = Det3DDataSample()
    sample.set_metainfo({
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR,
    })

    head = _resolve_bbox_head(model)
    saved = _apply_kitti_test_cfg(head, _KITTI_TEST_CFG)
    try:
        with torch.no_grad():
            data = model.data_preprocessor(
                {'inputs': {'points': [pts_tensor]},
                 'data_samples': [sample]},
                training=False)
            pred_list = model.predict(data['inputs'], data['data_samples'])
    finally:
        _restore_test_cfg(head, saved)

    pred = pred_list[0]
    inst = pred.pred_instances_3d
    cls_s = inst.scores_3d.cpu().numpy()   # raw CLS (score_type='cls')
    boxes  = inst.bboxes_3d.tensor.cpu().numpy()[:, :7]
    labels = inst.labels_3d.cpu().numpy()
    iou_sc = getattr(inst, 'iou_scores_3d', None)
    iou_s  = iou_sc.cpu().numpy() if iou_sc is not None else None
    return boxes, cls_s, iou_s, labels


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(',')]


# ── BEV / side-view rendering helpers ─────────────────────────────────────────

def _draw_box_bev(ax, box: np.ndarray, color: str,
                  linestyle: str, linewidth: float, zorder: int) -> None:
    """Draw one [x, y, z, l, w, h, yaw] box as a rotated rectangle in BEV."""
    x, y, _, l, w, _, yaw = box
    hl, hw = l / 2.0, w / 2.0
    local = np.array([
        [-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    world = local @ R.T + np.array([x, y])
    ax.plot(world[:, 0], world[:, 1],
            color=color, linestyle=linestyle, linewidth=linewidth,
            zorder=zorder)
    # Heading indicator: center → front-center
    front = np.array([hl, 0.0]) @ R.T + np.array([x, y])
    ax.plot([x, front[0]], [y, front[1]],
            color=color, linestyle=linestyle, linewidth=linewidth,
            zorder=zorder)


def _draw_box_side(ax_side, box: np.ndarray, color: str,
                   linestyle: str, linewidth: float, zorder: int) -> None:
    """Draw one box in the depth-height side view.

    x-axis = y_nus (forward depth), y-axis = z (height).
    LiDARInstance3DBoxes stores z at bottom-center, so the rectangle starts at z.
    box[3]=l is used as the depth extent (approximate; ignores yaw projection).
    """
    y_center, z_bottom, l, h = box[1], box[2], box[3], box[5]
    rect = mpatches.Rectangle(
        (y_center - l / 2, z_bottom), l, h,
        linewidth=linewidth, edgecolor=color, facecolor='none',
        linestyle=linestyle, zorder=zorder)
    ax_side.add_patch(rect)


# ── Combined render ────────────────────────────────────────────────────────────

def render_bev(pts_nus: np.ndarray,
               gt_boxes: np.ndarray, gt_labels: np.ndarray,
               ps_boxes: np.ndarray, ps_scores: np.ndarray,
               bl_boxes, bl_scores,
               scene_id: str, out_path: str,
               gt_cars_only: bool = False,
               gt_car_label: int = 2) -> None:
    """Render and save a two-panel PNG (BEV left + side-view right) for one scene.

    BEV annotations show 'score/BEV-IoU' per box, colour-coded by IoU quality
    (green ≥0.5, orange ≥0.25, red <0.25).  Side-view shows depth vs z-height
    to expose vertical placement errors.
    """
    fig, (ax_bev, ax_side) = plt.subplots(1, 2, figsize=(28, 12))
    fig.patch.set_facecolor('#0a0a0a')
    for ax in (ax_bev, ax_side):
        ax.set_facecolor('#0a0a0a')

    # ── Points (coloured by height) ──
    z_vals = pts_nus[:, 2]
    z_rng = z_vals.max() - z_vals.min() + 1e-6
    z_norm = (z_vals - z_vals.min()) / z_rng
    ax_bev.scatter(pts_nus[:, 0], pts_nus[:, 1],
                   s=0.2, c=z_norm, cmap='viridis', alpha=0.35,
                   linewidths=0, zorder=1)
    ax_side.scatter(pts_nus[:, 1], pts_nus[:, 2],
                    s=0.2, c=z_norm, cmap='viridis', alpha=0.3,
                    linewidths=0, zorder=1)

    # IoU reference: GT Car boxes only (single-class model comparison)
    gt_car_boxes = (gt_boxes[gt_labels == gt_car_label]
                    if len(gt_labels) > 0 else np.zeros((0, 7), dtype=np.float32))
    ps_iou = bev_iou_matrix(ps_boxes, gt_car_boxes)
    bl_arr = bl_boxes if bl_boxes is not None else np.zeros((0, 7), dtype=np.float32)
    bl_iou = bev_iou_matrix(bl_arr, gt_car_boxes)

    legend_items = []

    # ── GT boxes ──
    n_gt_car = int((gt_labels == gt_car_label).sum()) if len(gt_labels) else 0
    n_gt_other = len(gt_labels) - n_gt_car
    for box, label in zip(gt_boxes, gt_labels):
        is_car = (label == gt_car_label)
        if gt_cars_only and not is_car:
            continue
        color = 'limegreen' if is_car else '#888888'
        _draw_box_bev(ax_bev, box, color=color, linestyle='--',
                      linewidth=1.8, zorder=3)
        if is_car:
            _draw_box_side(ax_side, box, color='limegreen', linestyle='--',
                           linewidth=1.5, zorder=3)
    if n_gt_car > 0 or (not gt_cars_only and n_gt_other > 0):
        legend_items.append(mpatches.Patch(
            facecolor='none', edgecolor='limegreen', linestyle='--',
            label=f'GT Car ({n_gt_car})'))
    if not gt_cars_only and n_gt_other > 0:
        legend_items.append(mpatches.Patch(
            facecolor='none', edgecolor='#888888', linestyle='--',
            label=f'GT other ({n_gt_other})'))

    # ── Pseudo-labels ──
    # Score annotated in BEV at the heading-tip (beyond front face), so boxes
    # facing different directions naturally spread their labels.
    # BEV IoU annotated in the side-view above each rectangle.
    for i, (box, score) in enumerate(zip(ps_boxes, ps_scores)):
        _draw_box_bev(ax_bev, box, color='tomato', linestyle='-',
                      linewidth=2.0, zorder=4)
        best_iou = float(ps_iou[i].max()) if ps_iou.shape[1] > 0 else 0.0
        iou_color = ('limegreen' if best_iou >= 0.5
                     else 'orange' if best_iou >= 0.25 else 'tomato')
        # Score at heading tip (box front + 0.9 m offset along heading)
        x, y, _, l, _, _, yaw = box
        tip_x = x + (l / 2 + 0.9) * np.cos(yaw)
        tip_y = y + (l / 2 + 0.9) * np.sin(yaw)
        ax_bev.text(tip_x, tip_y, f'{score:.2f}',
                    color='tomato', fontsize=6, ha='center', va='center',
                    zorder=5,
                    bbox=dict(facecolor='black', alpha=0.45, pad=1,
                              linewidth=0, boxstyle='round,pad=0.15'))
        _draw_box_side(ax_side, box, color='tomato', linestyle='-',
                       linewidth=1.5, zorder=4)
        # IoU above each rectangle in side-view
        ax_side.text(box[1], box[2] + box[5] + 0.15, f'{best_iou:.2f}',
                     color=iou_color, fontsize=6, ha='center', va='bottom',
                     zorder=5,
                     bbox=dict(facecolor='black', alpha=0.45, pad=1,
                               linewidth=0, boxstyle='round,pad=0.15'))
    legend_items.append(mpatches.Patch(
        facecolor='none', edgecolor='tomato',
        label=f'Pseudo ({len(ps_boxes)})  BEV score@tip | IoU@side'))

    # ── Baseline predictions ──
    if bl_boxes is not None and len(bl_boxes) > 0:
        for i, (box, score) in enumerate(zip(bl_boxes, bl_scores)):
            _draw_box_bev(ax_bev, box, color='dodgerblue', linestyle=':',
                          linewidth=1.8, zorder=4)
            best_iou = float(bl_iou[i].max()) if bl_iou.shape[1] > 0 else 0.0
            iou_color = ('limegreen' if best_iou >= 0.5
                         else 'orange' if best_iou >= 0.25 else 'dodgerblue')
            x, y, _, l, _, _, yaw = box
            tip_x = x + (l / 2 + 0.9) * np.cos(yaw)
            tip_y = y + (l / 2 + 0.9) * np.sin(yaw)
            ax_bev.text(tip_x, tip_y, f'{score:.2f}',
                        color='dodgerblue', fontsize=6, ha='center', va='center',
                        zorder=5,
                        bbox=dict(facecolor='black', alpha=0.45, pad=1,
                                  linewidth=0, boxstyle='round,pad=0.15'))
            _draw_box_side(ax_side, box, color='dodgerblue', linestyle=':',
                           linewidth=1.5, zorder=4)
            ax_side.text(box[1], box[2] + box[5] + 0.15, f'{best_iou:.2f}',
                         color=iou_color, fontsize=6, ha='center', va='bottom',
                         zorder=5,
                         bbox=dict(facecolor='black', alpha=0.45, pad=1,
                                   linewidth=0, boxstyle='round,pad=0.15'))
        legend_items.append(mpatches.Patch(
            facecolor='none', edgecolor='dodgerblue', linestyle=':',
            label=f'Baseline ({len(bl_boxes)})  BEV score@tip | IoU@side'))
    elif bl_boxes is not None:
        legend_items.append(mpatches.Patch(
            facecolor='none', edgecolor='dodgerblue', linestyle=':',
            label='Baseline (0)'))

    # ── BEV panel styling ──
    ax_bev.legend(handles=legend_items, loc='upper right', fontsize=9,
                  facecolor='#222222', edgecolor='white', labelcolor='white')
    bl_n = len(bl_boxes) if bl_boxes is not None else '–'
    ax_bev.set_title(
        f'BEV  Scene {scene_id}  |  GT Car={n_gt_car}  '
        f'Pseudo={len(ps_boxes)}  Baseline={bl_n}',
        color='white', fontsize=11, pad=8)
    ax_bev.set_xlabel('X  (nuScenes — rightward)', color='white', fontsize=9)
    ax_bev.set_ylabel('Y  (nuScenes — forward)', color='white', fontsize=9)
    ax_bev.tick_params(colors='white')
    for spine in ax_bev.spines.values():
        spine.set_edgecolor('#444444')
    ax_bev.grid(True, alpha=0.12, color='white', linewidth=0.5)
    ax_bev.set_aspect('equal')
    margin = 5.0
    ax_bev.set_xlim(pts_nus[:, 0].min() - margin, pts_nus[:, 0].max() + margin)
    ax_bev.set_ylim(pts_nus[:, 1].min() - margin, pts_nus[:, 1].max() + margin)

    # ── Side-view panel styling ──
    # Ground reference: after KittiToNuscenes shifts KITTI z by −0.11 m,
    # ground sits at z ≈ −1.84 m (nuScenes sensor height convention).
    # A correctly placed car has its bottom edge at this line; boxes well
    # below indicate a residual z-axis domain gap.
    ax_side.axhline(-1.84, color='white', linestyle=':', linewidth=0.8,
                    alpha=0.5, label='ground ref z≈−1.84 m')
    ax_side.legend(loc='upper right', fontsize=8,
                   facecolor='#222222', edgecolor='white', labelcolor='white')
    ax_side.set_title(
        f'Side View  Scene {scene_id}  (box z = bottom-center)',
        color='white', fontsize=11, pad=8)
    ax_side.set_xlabel('Depth (Y nuScenes — forward, m)', color='white', fontsize=9)
    ax_side.set_ylabel('Height (Z, m)', color='white', fontsize=9)
    ax_side.tick_params(colors='white')
    for spine in ax_side.spines.values():
        spine.set_edgecolor('#444444')
    ax_side.grid(True, alpha=0.12, color='white', linewidth=0.5)
    margin_s = 2.0
    ax_side.set_xlim(pts_nus[:, 1].min() - margin_s, pts_nus[:, 1].max() + margin_s)
    ax_side.set_ylim(pts_nus[:, 2].min() - margin_s, pts_nus[:, 2].max() + margin_s)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a')
    plt.close(fig)
    print(f'  Saved → {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Visualise pseudo-labels vs GT vs baseline (BEV PNG)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Data sources ──────────────────────────────────────────────────────────
    g = p.add_argument_group('Data sources')
    g.add_argument('--ps-label-pkl',
                   help='Path to ps_label_e*.pkl from PseudoLabelRefreshHook '
                        '(omit to run in baseline-only mode)')
    g.add_argument('--kitti-info', default='data/kitti/kitti_infos_train.pkl',
                   help='KITTI train info pkl (GT source)')

    # ── Baseline model ────────────────────────────────────────────────────────
    g = p.add_argument_group('Baseline model')
    g.add_argument('--baseline-config', default=None,
                   help='Config .py for the baseline model (required unless --no-baseline)')
    g.add_argument('--baseline-ckpt', default=None,
                   help='Checkpoint .pth for the baseline model (required unless --no-baseline)')
    g.add_argument('--no-baseline', action='store_true',
                   help='Skip baseline inference (no GPU needed; pkl-only mode)')
    g.add_argument('--device', default='cuda:0',
                   help='Torch device for baseline inference')

    # ── Thresholds & scoring ──────────────────────────────────────────────────
    # The score used for ranking (keep_frac) and the floor is the raw CLS score.
    g = p.add_argument_group('Thresholds & scoring')
    g.add_argument('--sweep-top-n', type=int, default=20,
                   help='Number of top results to display in sweep table.')
    # ── Keep-fraction sweep axes ──────────────────────────────────────────────
    g.add_argument('--sweep-keep-fracs', default=None,
                   help='Comma-separated keep_frac values for --sweep-from-cache '
                        '(e.g. "0.2,0.3,0.4,0.5"). Activates the keep-fraction sweep.')
    g.add_argument('--sweep-floor-percentiles', default='0',
                   help='Comma-separated GLOBAL score percentiles [0-100] for the '
                        'adaptive floor in the keep-frac sweep. For each value P the '
                        'floor is set to the P-th percentile of the pooled box-score '
                        'distribution across ALL scenes (keep_frac stays per-scene). '
                        '0 = no floor. Replaces the old absolute --sweep-min-floors so '
                        'the floor tracks teacher-confidence drift instead of a fixed cut.')
    g.add_argument('--sweep-fallback-ks', default='0,1,3',
                   help='Comma-separated fallback_k integer values for the keep-frac sweep.')
    g.add_argument('--sweep-abs-floors', default=None,
                   help='Comma-separated ABSOLUTE score floors (e.g. "0.25,0.3,0.35,0.4") '
                        'for the keep-frac sweep. Swept in ADDITION to '
                        '--sweep-floor-percentiles, so a single run compares fixed floors '
                        'against the adaptive percentile floor. In the table the '
                        'floor_pctl column shows the percentile (e.g. 60) for adaptive '
                        'rows and the fixed value (e.g. 0.3) for absolute rows; the '
                        '→floor column always shows the absolute cut applied.')
    g.add_argument('--sweep-min-pt-counts', default='0',
                   help='Comma-separated minimum interior point count values for the '
                        'keep-frac sweep (e.g. "0,3,5"). 0 = no filter. Requires '
                        '--kitti-info so the velodyne_reduced bin files can be located.')
    # ── Per-scene keep-fraction thresholding ─────────────────────────────────
    g.add_argument('--keep-frac', type=float, default=None,
                   help='Per-scene keep-fraction: keep the top ceil(keep_frac × N) '
                        'highest-CLS boxes per scene. None = keep all (floor still '
                        'applies). Combine with --min-floor / --floor-percentile.')
    g.add_argument('--floor-percentile', type=float, default=0.0,
                   help='Adaptive score floor used with --keep-frac: drop surviving '
                        'boxes whose score is below the P-th percentile of the GLOBAL '
                        'pooled box-score distribution (computed across all scenes in a '
                        'pre-pass; keep_frac stays per-scene). 0 = off. Adaptive '
                        'alternative to the fixed --min-floor; --min-floor takes precedence.')
    g.add_argument('--min-floor', type=float, default=0.0,
                   help='FIXED absolute score floor used with --keep-frac: drop survivors '
                        'below this value (floor acts as a guard after the per-scene '
                        'fraction, unless --floor-before). 0 = off. Takes precedence over '
                        '--floor-percentile when both are set (no pooling pre-pass needed).')
    g.add_argument('--fallback-k', type=int, default=0,
                   help='If fewer than this many boxes survive --keep-frac + '
                        'the floor, restore the top-k raw boxes (ignoring the '
                        'floor) to prevent empty scenes. 0 = off. Default 1.')
    g.add_argument('--floor-before', action='store_true',
                   help='Apply the floor before computing the keep-fraction '
                        '(denominator = above-floor count). Default is '
                        'fraction-first (floor acts as a guard afterwards).')
    g.add_argument('--baseline-min-pts', type=int, default=0,
                   help='Fixed interior-point filter on baseline predictions: '
                        'drop baseline boxes containing fewer than this many '
                        'LiDAR points (counted in the nuScenes frame, same as '
                        '--sweep-min-pt-counts but a single fixed value applied '
                        'to the baseline). 0 = off.')

    g.add_argument('--gt-iou', action='store_true',
                   help='Replace the model IoU-head axis with actual GT-box IoU '
                        '(best overlap of each prediction vs GT Car boxes). '
                        'Required for models without an IoU head, e.g. CenterPoint.')
    g.add_argument('--gt-iou-mode', choices=['3d', 'bev', 'both'], default='both',
                   help='IoU type for --gt-iou: 3d (full volume), bev (bird\'s-eye-view), '
                        'or both (two extra panels).')

    # ── Output & sampling ─────────────────────────────────────────────────────
    g = p.add_argument_group('Output & sampling')
    g.add_argument('--out-dir', default='vis_ps_labels',
                   help='Directory for output PNGs / histograms')
    g.add_argument('--num-scenes', type=int, default=10,
                   help='Number of random scenes to visualise')
    g.add_argument('--seed', type=int, default=0,
                   help='RNG seed for scene sampling')
    g.add_argument('--stats-only', action='store_true',
                   help='Print per-scene TP/FP/FN stats; skip PNG generation')
    g.add_argument('--gt-cars-only', action='store_true',
                   help='Show only Car GT boxes (hide Pedestrian/Cyclist)')
    g.add_argument('--score-hist', action='store_true',
                   help='Save score distribution histogram to <out-dir>/score_distributions.png')
    g.add_argument('--points-analysis', action='store_true',
                   help='Compute interior point counts for every PS-label box and save '
                        'points_histogram.png + points_iou_cls_scatter.html to --out-dir. '
                        'Requires --ps-label-pkl. --no-baseline is fine (no GPU needed). '
                        'Iterates all pkl entries regardless of --num-scenes.')
    g.add_argument('--dump-raw-preds', metavar='PATH', default=None,
                   help='Run inference on all scenes and save raw (unfiltered) predictions '
                        'to PATH as a pkl cache for use with --sweep-from-cache. '
                        'Requires --baseline-config/--baseline-ckpt. Early exit after dump.')
    g.add_argument('--sweep-from-cache', metavar='PATH', default=None,
                   help='Load raw-prediction cache from PATH (created by --dump-raw-preds) '
                        'and sweep threshold combinations to find optimal operating points. '
                        'Early exit after printing the sweep table.')

    return p.parse_args()


def plot_points_histogram(pt_counts: np.ndarray, out_path: str,
                          title: str = 'Interior point count per PS-box') -> None:
    """Save a dark-theme histogram of per-box interior point counts.

    Uses log-spaced bins so the heavy tail is visible without squashing the
    sparse end.  Vertical lines mark mean and median; text annotations give
    the fraction of boxes in key sparsity bands.
    """
    counts = np.asarray(pt_counts, dtype=np.float32)
    n = len(counts)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111111')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    # Log-spaced bins: 0 placed at leftmost edge, rest log-spaced
    max_val = max(int(counts.max()), 1)
    log_edges = np.unique(np.concatenate([
        [0, 1],
        np.logspace(0, np.log10(max_val + 1), 50).astype(int),
    ])).astype(float)
    ax.hist(counts, bins=log_edges, color='#4db8ff', alpha=0.85, edgecolor='none')
    ax.set_xscale('symlog', linthresh=1)

    med = float(np.median(counts))
    mean = float(counts.mean())
    ax.axvline(med,  color='white',  linestyle='--', linewidth=1.2,
               label=f'median={med:.0f}')
    ax.axvline(mean, color='yellow', linestyle=':',  linewidth=1.2,
               label=f'mean={mean:.1f}')

    f0   = 100.0 * (counts == 0).mean()
    f5   = 100.0 * (counts <= 5).mean()
    f20  = 100.0 * (counts <= 20).mean()
    stats_str = (f'n={n}\n'
                 f'0 pts:  {f0:.1f}%\n'
                 f'≤5 pts: {f5:.1f}%\n'
                 f'≤20 pts:{f20:.1f}%')
    ax.text(0.98, 0.97, stats_str, transform=ax.transAxes,
            color='white', fontsize=9, va='top', ha='right',
            bbox=dict(facecolor='#222222', alpha=0.8, pad=4, linewidth=0))

    ax.set_title(title, color='white', fontsize=12, pad=8)
    ax.set_xlabel('Points inside box (log scale)', color='white', fontsize=10)
    ax.set_ylabel('Box count', color='white', fontsize=10)
    ax.legend(fontsize=9, facecolor='#222222', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close(fig)
    print(f'Points histogram saved → {out_path}')


def plot_iou_cls_pts_3d_scatter(pt_counts: np.ndarray,
                                 gt_ious: np.ndarray,
                                 cls_scores: np.ndarray,
                                 out_path: str,
                                 title: str = 'PS-box quality: CLS × GT-IoU × Point count'
                                 ) -> None:
    """Save an interactive 3D scatter plot as a standalone HTML file (Plotly).

    Axes:
        x = CLS score (teacher classification confidence)
        y = GT 3D IoU  (actual overlap with nearest GT Car box; 0 = FP)
        z = log₁₀(pt_count + 1)  (log-normalised interior point count)

    Colour encodes raw pt_count (Viridis scale).  The HTML uses CDN-hosted
    Plotly.js so the file is small enough to download and open locally in a
    browser.
    """
    if not _PLOTLY_OK:
        print('plot_iou_cls_pts_3d_scatter: plotly not installed — skipping HTML. '
              'Install with: pip install plotly')
        return

    pt  = np.asarray(pt_counts,  dtype=np.float32)
    iou = np.asarray(gt_ious,    dtype=np.float32)
    cls = np.asarray(cls_scores, dtype=np.float32)
    z   = np.log10(pt + 1)

    fig = go.Figure(data=[go.Scatter3d(
        x=cls, y=iou, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=pt,
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(
                title=dict(text='Points in box', font=dict(color='white')),
                tickfont=dict(color='white'),
            ),
        ),
        hovertemplate=(
            'CLS: %{x:.3f}<br>'
            'GT IoU: %{y:.3f}<br>'
            'log₁₀(pts+1): %{z:.2f}<br>'
            '<extra></extra>'
        ),
    )])

    axis_style = dict(
        backgroundcolor='#111111',
        gridcolor='#333333',
        showbackground=True,
        tickfont=dict(color='white'),
        title_font=dict(color='white'),
        zerolinecolor='#555555',
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=14)),
        paper_bgcolor='#0a0a0a',
        scene=dict(
            xaxis=dict(**axis_style, title='CLS score'),
            yaxis=dict(**axis_style, title='GT 3D IoU'),
            zaxis=dict(**axis_style, title='log₁₀(pt_count+1)'),
            bgcolor='#111111',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        font=dict(color='white'),
    )

    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'3D scatter saved → {out_path}')

    # Pearson correlations
    n = len(pt)
    if n > 1:
        r_cls_iou = float(np.corrcoef(cls, iou)[0, 1])
        r_cls_pt  = float(np.corrcoef(cls, z)[0, 1])
        r_iou_pt  = float(np.corrcoef(iou, z)[0, 1])
        print(f'  Pearson r(CLS, GT-IoU)         = {r_cls_iou:.4f}')
        print(f'  Pearson r(CLS, log-pt_count)   = {r_cls_pt:.4f}')
        print(f'  Pearson r(GT-IoU, log-pt_count)= {r_iou_pt:.4f}')


def _check_augmentation_warning(mt_config_path: str) -> None:
    """Warn if the weak pipeline has random augmentation transforms enabled."""
    try:
        cfg = Config.fromfile(mt_config_path)
        weak_pipe = (cfg.train_dataloader.dataset
                     .unlabeled_weak_dataset.pipeline)
        aug_types = {'RandomFlip3D', 'GlobalRotScaleTrans'}
        active = [getattr(t, 'type', None) for t in weak_pipe
                  if getattr(t, 'type', None) in aug_types]
        if active:
            print(
                f'\n⚠  WARNING: weak target pipeline has active augmentations '
                f'{active}.\n'
                '   Random per-sample transforms are not stored in the pkl, so '
                'pseudo-label\n'
                '   boxes and the loaded point cloud are in DIFFERENT augmented '
                'frames.\n'
                '   Overlay will be INCORRECT. Disable these transforms to get '
                'faithful visualisation.\n')
    except Exception as e:
        print(f'  (Could not check MT config for aug warning: {e})')


def _find_mt_config(pkl_path: str) -> str | None:
    """Derive the MT config saved by MMEngine alongside the checkpoints.

    MMEngine writes <config_name>.py to <work_dir>/ at training start.
    The pkl lives at <work_dir>/<timestamp>/ps_labels/ps_label_e*.pkl,
    so the work_dir root is three parent directories above the pkl.
    """
    work_dir = os.path.dirname(          # <work_dir>/
        os.path.dirname(                 # <work_dir>/<timestamp>/
            os.path.dirname(             # <work_dir>/<timestamp>/ps_labels/
                os.path.abspath(pkl_path))))
    py_files = [f for f in os.listdir(work_dir) if f.endswith('.py')]
    if not py_files:
        return None
    mt = [f for f in py_files if 'mean_teacher' in f]
    chosen = mt[0] if mt else py_files[0]
    return os.path.join(work_dir, chosen)


def _empty_stats() -> dict:
    return {'0.7': {'tp': 0, 'fp': 0, 'fn': 0},
            '0.50': {'tp': 0, 'fp': 0, 'fn': 0}}


def _pred_car_label(baseline_model, ps_labels: dict) -> int:
    """Return the integer label index that means 'Car' in prediction space.

    Prefers looking up 'Car' by name in the baseline model's dataset_meta.
    Falls back to scanning pseudo-label gt_labels: a single-class model always
    emits label 0, so if only one unique label exists it must be Car.
    """
    if baseline_model is not None:
        classes = list(baseline_model.dataset_meta.get('classes', []))
        if 'Car' in classes:
            idx = classes.index('Car')
            print(f'Prediction Car label = {idx}  (from model classes {classes})')
            return idx

    # No model available — infer from ps_labels
    all_labels = []
    for v in ps_labels.values():
        all_labels.extend(v.get('gt_labels', []).tolist())
    unique = sorted(set(all_labels))
    if len(unique) == 1:
        print(f'Prediction Car label = {unique[0]}  '
              f'(single unique label in ps_labels — assumed Car)')
        return unique[0]

    print('WARNING: cannot determine Car label index automatically; defaulting to 0. '
          'Pass a baseline model or ensure ps_labels contain only Car boxes.')
    return 0


def plot_score_distributions(cls_scores, iou_scores, out_path: str,
                             title: str = 'Pseudo-label score distributions',
                             score1_label: str = 'CLS score',
                             iou_label: str = 'IoU score',
                             iou2_scores=None,
                             iou2_label: str = 'IoU2 score',
                             thresholds: dict | None = None,
                             scene_stats: tuple[int, int] = (0, 0)) -> None:
    """Save a histogram of score distributions.

    Supports up to two IoU axes (e.g. 3D IoU and BEV IoU simultaneously).
    Each IoU axis generates one histogram panel + one CLS-vs-IoU hexbin panel.

    Args:
        cls_scores:   list or array of the first score (CLS or hybrid).
        iou_scores:   list or array for the primary IoU axis (may be empty).
        out_path:     path for the output PNG.
        title:        figure suptitle.
        score1_label: label for the CLS axis.
        iou_label:    label for the primary IoU axis.
        iou2_scores:  optional second IoU array (e.g. BEV when iou_scores is 3D).
        iou2_label:   label for the second IoU axis.
        thresholds:   optional dict with key min_cls (the score floor). When
            provided, draws the floor line on the CLS histogram and hexbin.
        scene_stats: (n_with_boxes, n_total) for scene coverage reporting.
    """
    cls = np.asarray(cls_scores, dtype=np.float32)

    # Build list of (array, label) for each IoU axis that has data
    iou_panels = []
    if len(iou_scores) > 0:
        iou_panels.append((np.asarray(iou_scores, dtype=np.float32), iou_label))
    if iou2_scores is not None and len(iou2_scores) > 0:
        iou_panels.append((np.asarray(iou2_scores, dtype=np.float32), iou2_label))

    # Layout: [CLS hist] + per IoU axis: [IoU hist] [CLS vs IoU hexbin]
    n_plots = 1 + 2 * len(iou_panels)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    fig.patch.set_facecolor('#0a0a0a')
    if n_plots == 1:
        axes = [axes]
    for ax in axes:
        ax.set_facecolor('#111111')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    bins = np.linspace(0, 1, 41)  # 40 bins of width 0.025

    def _hist(ax, data, color, panel_title, vline=None):
        ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor='none')
        ax.axvline(np.median(data), color='white', linestyle='--',
                   linewidth=1.2, label=f'median={np.median(data):.3f}')
        ax.axvline(np.mean(data), color='yellow', linestyle=':',
                   linewidth=1.2, label=f'mean={np.mean(data):.3f}')
        if vline is not None and vline > 0:
            ax.axvline(vline, color='red', linestyle='--', linewidth=1.2,
                       alpha=0.85, label=f'floor={vline:.2f}')
        ax.set_title(f'{panel_title}  (n={len(data)})', color='white', fontsize=11)
        ax.set_xlabel('Score', color='white', fontsize=9)
        ax.set_ylabel('Box count', color='white', fontsize=9)
        ax.legend(fontsize=8, facecolor='#222222',
                  edgecolor='white', labelcolor='white')
        stats = (f'min={data.min():.3f}  max={data.max():.3f}\n'
                 f'std={data.std():.3f}')
        ax.text(0.02, 0.97, stats, transform=ax.transAxes,
                color='white', fontsize=8, va='top',
                bbox=dict(facecolor='#222222', alpha=0.7, pad=3, linewidth=0))

    def _hexbin(ax, x, y, y_label, thr=None):
        """CLS (x) vs IoU (y) joint hexbin panel."""
        thr = thr or {}
        hb = ax.hexbin(x, y, gridsize=30, cmap='plasma',
                       extent=[0, 1, 0, 1], mincnt=1)
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.ax.tick_params(colors='white', labelsize=7)
        cb.outline.set_edgecolor('#444444')
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.set_title(f'{score1_label} vs {y_label}  (r={corr:.3f})',
                     color='white', fontsize=11)
        ax.set_xlabel(score1_label, color='white', fontsize=9)
        ax.set_ylabel(y_label, color='white', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], color='white', linestyle=':', linewidth=0.8,
                alpha=0.5, label=f'{score1_label} = {y_label}')
        # Floor line on the CLS axis.
        min_cls = thr.get('min_cls', 0.0)
        if min_cls > 0:
            ax.axvline(min_cls, color='cyan', linestyle='--',
                       linewidth=1.0, alpha=0.85, label=f'floor={min_cls:.2f}')
        ax.legend(fontsize=7, facecolor='#222222',
                  edgecolor='white', labelcolor='white')
        return corr

    thr = thresholds or {}
    _hist(axes[0], cls, '#4db8ff', f'{score1_label} distribution',
          vline=thr.get('min_cls'))

    # IoU panel colours — first axis coral, second axis green
    iou_colours = ['#ff7f50', '#7fff7f']
    corr_list = []
    for panel_idx, (iou_arr, lbl) in enumerate(iou_panels):
        ax_hist = axes[1 + panel_idx * 2]
        ax_hex  = axes[2 + panel_idx * 2]
        _hist(ax_hist, iou_arr, iou_colours[panel_idx % 2], f'{lbl} distribution',
              vline=thr.get('min_iou') if panel_idx == 0 else None)
        corr = _hexbin(ax_hex, cls, iou_arr, lbl, thr=thr if panel_idx == 0 else {})
        corr_list.append((lbl, corr))

    fig.suptitle(title, color='white', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close(fig)
    print(f'Score distribution plot saved → {out_path}')

    # Text summary
    print(f'\n=== Score distributions ({len(cls)} boxes) ===')
    all_series = [(score1_label, cls)] + [(lbl, arr) for arr, lbl in iou_panels]
    for name, data in all_series:
        print(f'\n  {name}:  min={data.min():.3f}  max={data.max():.3f}  '
              f'mean={data.mean():.3f}  median={np.median(data):.3f}  std={data.std():.3f}')
        bins_e = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        counts, _ = np.histogram(data, bins=bins_e)
        for lo, hi, c in zip(bins_e, bins_e[1:], counts):
            bar = '█' * int(c / max(counts) * 30)
            print(f'    [{lo:.1f}-{hi:.1f}]: {c:6d}  {bar}')
    if corr_list:
        for lbl, r in corr_list:
            print(f'\n  Pearson r({score1_label}, {lbl}) = {r:.4f}')
    else:
        print('\n  IoU scores not available — joint plot skipped.')
    n_with, n_total = scene_stats
    if n_total > 0:
        pct = 100.0 * n_with / n_total
        print(f'\n  Scene coverage: {n_with}/{n_total} ({pct:.1f}%) scenes have ≥1 box')


def _dump_raw_predictions(args) -> None:
    """Run inference on all selected scenes and save raw predictions to a pkl cache."""
    for name, val in [('--baseline-config', args.baseline_config),
                      ('--baseline-ckpt', args.baseline_ckpt)]:
        if val is None:
            print(f'ERROR: {name} is required for --dump-raw-preds.')
            sys.exit(1)

    from mmdet3d.apis import init_model
    print(f'Loading baseline model: {args.baseline_ckpt}')
    _cfg = Config.fromfile(args.baseline_config)
    if not hasattr(_cfg, 'class_names'):
        try:
            _cfg.class_names = list(
                _cfg.train_dataloader.dataset.metainfo.get('classes', ['Car']))
        except Exception:
            _cfg.class_names = ['Car']
    model = init_model(_cfg, args.baseline_ckpt, device=args.device)
    model.eval()

    print(f'Loading KITTI GT from {args.kitti_info}...')
    gt_lookup, gt_car_label = build_gt_lookup(args.kitti_info)

    classes = list(model.dataset_meta.get('classes', []))
    pred_car_label = classes.index('Car') if 'Car' in classes else 0
    print(f'pred_car_label={pred_car_label}  gt_car_label={gt_car_label}')

    velodyne_dir = os.path.join(os.path.dirname(args.kitti_info),
                                'training', 'velodyne_reduced')
    available = [os.path.join(velodyne_dir, k + '.bin')
                 for k in sorted(gt_lookup.keys())]
    n = min(args.num_scenes, len(available))
    selected = random.Random(args.seed).sample(available, n)
    print(f'Dumping raw predictions for {n} scene(s) → {args.dump_raw_preds}')

    scenes = []
    for key in selected:
        scene_id = os.path.splitext(os.path.basename(key))[0]
        if not os.path.isfile(key):
            print(f'  WARN: {key!r} not found, skipping.')
            continue
        pts_nus = load_points_nus(key)
        boxes, cls_s, iou_s, labels = run_baseline_raw(model, pts_nus, args.device)
        gt_info = gt_lookup.get(scene_id)
        if gt_info is not None and len(gt_info['cam_boxes']) > 0:
            gt_all = gt_boxes_to_nus(gt_info['cam_boxes'], gt_info['lidar2cam'])
            gt_lbl = gt_info['labels']
            gt_car = gt_all[gt_lbl == gt_car_label]
        else:
            gt_car = np.zeros((0, 7), dtype=np.float32)
        scenes.append({
            'scene_id': scene_id,
            'boxes': boxes, 'cls_scores': cls_s,
            'iou_scores': iou_s, 'labels': labels,
            'gt_car_boxes': gt_car,
        })
        print(f'  {scene_id}: {len(boxes)} raw boxes, {len(gt_car)} GT cars')

    cache = {'scenes': scenes, 'pred_car_label': pred_car_label,
             'n_scenes': len(scenes)}
    os.makedirs(os.path.dirname(os.path.abspath(args.dump_raw_preds)), exist_ok=True)
    with open(args.dump_raw_preds, 'wb') as f:
        pickle.dump(cache, f)
    print(f'\nCache saved → {args.dump_raw_preds}  ({len(scenes)} scenes)')


def _f05(p: float, r: float) -> float:
    """F-score with β=0.5 (weights precision twice as much as recall)."""
    denom = 0.25 * p + r
    return 1.25 * p * r / denom if denom > 0 else 0.0


def _sweep_thresholds(args) -> None:
    """Load a raw-prediction cache and sweep keep-fraction filtering combinations.

    Sweeps ``keep_frac × floor × fallback_k × min_pt_count`` using
    ``select_scene_keep_mask`` per scene, ranking/flooring on the CLS score.
    The floor axis combines ``--sweep-floor-percentiles`` (adaptive, pooled)
    and ``--sweep-abs-floors`` (fixed). Requires ``--sweep-keep-fracs``.
    CSV columns: ``keep_frac, floor_percentile, floor_value, fallback_k,
    min_pt_count, precision, recall, f05, scene_coverage``.
    """
    print(f'Loading cache: {args.sweep_from_cache}')
    with open(args.sweep_from_cache, 'rb') as f:
        cache = pickle.load(f)

    if 'scenes' in cache:
        # raw_preds cache written by --dump-raw-preds
        scenes         = cache['scenes']
        pred_car_label = cache['pred_car_label']
    else:
        # PS-label pkl (ps_label_eN.pkl) — convert to the common scene-list format
        kitti_info = getattr(args, 'kitti_info', None)
        if kitti_info is None:
            print('ERROR: --kitti-info is required when sweeping a PS-label pkl '
                  '(needed to load GT car boxes).')
            return
        print(f'  PS-label pkl detected — loading GT from {kitti_info}...')
        gt_lookup, gt_car_label = build_gt_lookup(kitti_info)
        pred_car_label = 0  # single-class Car model
        scenes = []
        for key, entry in cache.items():
            if not isinstance(entry, dict) or 'gt_boxes' not in entry:
                continue
            scene_id = os.path.splitext(os.path.basename(key))[0]
            gt_info  = gt_lookup.get(scene_id)
            if gt_info is not None and len(gt_info['cam_boxes']) > 0:
                gt_all  = gt_boxes_to_nus(gt_info['cam_boxes'], gt_info['lidar2cam'])
                gt_car  = gt_all[gt_info['labels'] == gt_car_label]
            else:
                gt_car = np.zeros((0, 7), dtype=np.float32)
            scenes.append({
                'scene_id':    scene_id,
                '_bin_path':   key,   # full path for point-count loading
                'boxes':       entry['gt_boxes'],
                'cls_scores':  entry.get('cls_scores', entry['scores']),
                'iou_scores':  entry.get('iou_scores'),
                'labels':      entry.get('gt_labels',
                                         np.zeros(len(entry['gt_boxes']),
                                                  dtype=np.int64)),
                'gt_car_boxes': gt_car,
            })
        print(f'  Converted: {len(scenes)} scenes with PS boxes')

    n_total = len(scenes)
    print(f'  {n_total} scenes, pred_car_label={pred_car_label}')

    csv_path = args.sweep_from_cache.replace('.pkl', '_sweep.csv')

    # ── Keep-fraction sweep (new mode) ────────────────────────────────────────
    if args.sweep_keep_fracs is not None:
        keep_fracs   = _parse_floats(args.sweep_keep_fracs)
        percentiles  = _parse_floats(args.sweep_floor_percentiles)
        fallback_ks  = [int(x) for x in args.sweep_fallback_ks.split(',')]
        min_pt_counts = [int(x) for x in
                         getattr(args, 'sweep_min_pt_counts', '0').split(',')]

        # ── Global score pool → percentile-to-absolute-floor map ──────────────
        # keep_frac is per-scene; the floor is the P-th percentile of the CLS
        # scores pooled across ALL scenes, so it drifts with the distribution.
        pool = np.concatenate([
            sc['cls_scores']
            for sc in scenes if len(sc['boxes']) > 0
        ]) if any(len(sc['boxes']) > 0 for sc in scenes) else np.zeros(0)
        floor_for = {
            P: (float(np.percentile(pool, P)) if (P > 0 and len(pool) > 0) else 0.0)
            for P in percentiles
        }
        print('  Adaptive floors (global percentile → score):  '
              + '  '.join(f'P{P:g}={floor_for[P]:.3f}' for P in percentiles))

        # Floor axis = adaptive percentile floors + (optionally) fixed absolute
        # floors. Each entry is (display, abs_floor): display is the percentile
        # for adaptive rows and the value itself for absolute rows.
        floor_specs = [(P, floor_for[P]) for P in percentiles]
        if args.sweep_abs_floors:
            abs_floors = _parse_floats(args.sweep_abs_floors)
            floor_specs += [(fl, fl) for fl in abs_floors]
            print('  Absolute floors:  '
                  + '  '.join(f'{fl:.3f}' for fl in abs_floors))

        # ── Pre-compute per-box interior point counts (one pass) ──────────────
        need_pt = any(m > 0 for m in min_pt_counts)
        if need_pt:
            kitti_info = getattr(args, 'kitti_info', None)
            if kitti_info is None:
                print('WARNING: --kitti-info required for --sweep-min-pt-counts > 0. '
                      'Dropping point count filter.')
                min_pt_counts = [0]
                need_pt = False
            else:
                velodyne_dir = os.path.join(os.path.dirname(kitti_info),
                                            'training', 'velodyne_reduced')
                print(f'Pre-computing point counts for {n_total} scenes '
                      f'(velodyne_dir={velodyne_dir})...')
                for sc in scenes:
                    bin_path = sc.get('_bin_path') or os.path.join(
                        velodyne_dir, sc['scene_id'] + '.bin')
                    if os.path.isfile(bin_path):
                        pts_nus = load_points_nus(bin_path)
                        boxes_f32 = sc['boxes'].astype(np.float32)
                        sc['pt_counts'] = count_points_in_boxes(pts_nus, boxes_f32)
                    else:
                        sc['pt_counts'] = np.zeros(len(sc['boxes']), dtype=np.int32)
                print('  Done.\n')

        n_combos = (len(keep_fracs) * len(floor_specs)
                    * len(fallback_ks) * len(min_pt_counts))
        print(f'  Keep-fraction sweep: {n_combos} combinations '
              f'({len(keep_fracs)} keep_frac × {len(floor_specs)} floor × '
              f'{len(fallback_ks)} fallback_k × '
              f'{len(min_pt_counts)} min_pt_count)...\n')

        results = []   # (f05, p50, r50, cov, kf, pctl, fl, fk, min_pt)
        for kf in keep_fracs:
            for pctl, fl in floor_specs:
                for fk in fallback_ks:
                    for min_pt in min_pt_counts:
                        tp50 = fp50 = fn50 = 0
                        n_cov = 0
                        for sc in scenes:
                            cls_s  = sc['cls_scores']
                            boxes  = sc['boxes']
                            labels = sc['labels']
                            gt_car = sc['gt_car_boxes']

                            mask = select_scene_keep_mask(cls_s,
                                                          keep_frac=kf,
                                                          min_floor=fl,
                                                          fallback_k=fk)
                            if min_pt > 0 and 'pt_counts' in sc:
                                mask = mask & (sc['pt_counts'] >= min_pt)

                            car_mask = mask & (labels == pred_car_label)
                            pred_car = boxes[car_mask]
                            if len(pred_car) > 0:
                                n_cov += 1

                            t, f_p, f_n = match_boxes(pred_car, gt_car, 0.50)
                            tp50 += t; fp50 += f_p; fn50 += f_n

                        p50 = tp50 / (tp50 + fp50) if (tp50 + fp50) > 0 else 0.0
                        r50 = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
                        cov = n_cov / n_total if n_total > 0 else 0.0
                        results.append((_f05(p50, r50), p50, r50, cov,
                                        kf, pctl, fl, fk, min_pt))

        results.sort(reverse=True)
        top_n = args.sweep_top_n
        hdr = (f'{"Rank":>4}  {"keep_frac":>9}  {"floor_pctl":>10}  {"→floor":>7}  '
               f'{"fallback_k":>10}  {"min_pts":>7}  '
               f'{"P@0.5":>6}  {"R@0.5":>6}  {"F_β0.5":>6}  {"cov%":>6}')
        sep = '─' * len(hdr)
        print(f'=== Keep-fraction sweep — top {top_n} of {n_combos} by F_β=0.5 ===')
        print(sep)
        print(hdr)
        print(sep)
        for rank, (f, p, r, cov, kf, pctl, fl, fk, mp) in enumerate(results[:top_n], 1):
            print(f'{rank:>4}  {kf:>9.3f}  {pctl:>10.3g}  {fl:>7.3f}  {fk:>10d}  '
                  f'{mp:>7d}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {100*cov:>6.1f}')
        print(sep)

        with open(csv_path, 'w') as fout:
            fout.write('keep_frac,floor_percentile,floor_value,fallback_k,'
                       'min_pt_count,precision,recall,f05,scene_coverage\n')
            for f, p, r, cov, kf, pctl, fl, fk, mp in results:
                fout.write(f'{kf},{pctl},{fl:.4f},{fk},{mp},'
                           f'{p:.4f},{r:.4f},{f:.4f},{cov:.4f}\n')
        print(f'\nFull sweep results saved → {csv_path}')
        return

    # Keep-fraction is the only sweep mode.
    print('ERROR: --sweep-keep-fracs is required for --sweep-from-cache '
          '(e.g. --sweep-keep-fracs 0.4,0.6).')


def main():
    args = parse_args()

    # ── Early exit: dump raw predictions ──────────────────────────────────────
    if args.dump_raw_preds:
        _dump_raw_predictions(args)
        return

    # ── Early exit: sweep thresholds from cache ───────────────────────────────
    if args.sweep_from_cache:
        _sweep_thresholds(args)
        return

    if not args.stats_only:
        os.makedirs(args.out_dir, exist_ok=True)

    # ── Validate baseline args ──
    if not args.no_baseline:
        missing = [name for name, val in [
            ('--baseline-config', args.baseline_config),
            ('--baseline-ckpt',   args.baseline_ckpt),
        ] if val is None]
        if missing:
            print(f'ERROR: {", ".join(missing)} must be provided '
                  'when not using --no-baseline.')
            sys.exit(1)

    # ── Load pseudo-labels (optional) ──
    ps_labels = {}
    if args.ps_label_pkl:
        with open(args.ps_label_pkl, 'rb') as f:
            ps_labels = pickle.load(f)
        print(f'Loaded {len(ps_labels)} pseudo-label entries '
              f'from {args.ps_label_pkl}')

    # ── Score distribution histogram (from pkl if available) ──
    # pkl structure (written by PseudoLabelRefreshHook):
    #   'scores'     → hybrid quality score (w*iou + (1-w)*cls) used for filtering
    #   'cls_scores' → raw CLS: the teacher's scores_3d before hybrid composition;
    #                  equals sigmoid(CLS logit) when teacher uses score_type='cls'
    #   'iou_scores' → raw sigmoid of conv_iou (iou_scores_3d from bbox_head.predict)
    if args.score_hist and ps_labels:
        os.makedirs(args.out_dir, exist_ok=True)
        _cls_s, _iou_s = [], []
        _has_stored_cls = False
        for v in ps_labels.values():
            if len(v['scores']) == 0:
                continue
            raw_cls = v.get('cls_scores')
            if raw_cls is not None:
                # Raw CLS stored directly — the standard path with the new model.
                _cls_s.extend(raw_cls.tolist())
                _has_stored_cls = True
            else:
                # Fallback: pkl predates cls_scores key; use the hybrid score.
                _cls_s.extend(v['scores'].tolist())
            raw_iou = v.get('iou_scores')
            if raw_iou is not None:
                _iou_s.extend(raw_iou.tolist())

        _score1_label = 'CLS score' if _has_stored_cls else 'Hybrid score'
        _n_pkl_with = sum(1 for v in ps_labels.values() if len(v['scores']) > 0)
        plot_score_distributions(
            _cls_s, _iou_s,
            os.path.join(args.out_dir, 'score_distributions.png'),
            score1_label=_score1_label,
            scene_stats=(_n_pkl_with, len(ps_labels)))

    # ── Augmentation sanity check (only when pkl was provided) ──
    if args.ps_label_pkl:
        mt_config_path = _find_mt_config(args.ps_label_pkl)
        if mt_config_path and os.path.isfile(mt_config_path):
            print(f'MT config (derived from pkl path): {mt_config_path}')
            _check_augmentation_warning(mt_config_path)

    # ── GT lookup ──
    print(f'Loading KITTI GT from {args.kitti_info}...')
    gt_lookup, gt_car_label = build_gt_lookup(args.kitti_info)

    # ── Points analysis: interior point counts per PS-box ─────────────────────
    if getattr(args, 'points_analysis', False) and ps_labels:
        os.makedirs(args.out_dir, exist_ok=True)
        all_pt_counts: list = []
        all_gt_ious:   list = []
        all_cls_scores: list = []
        # Validate pkl format — must be a PS-label store (ps_label_eN.pkl),
        # not a raw-preds cache written by --dump-raw-preds.
        _sample = next(iter(ps_labels.values()))
        if not isinstance(_sample, dict) or 'gt_boxes' not in _sample:
            print('ERROR: --points-analysis requires a PS-label pkl (ps_label_eN.pkl) '
                  'with per-scene gt_boxes entries.\n'
                  '  Got a different format — possibly a --dump-raw-preds cache.\n'
                  '  Pass a ps_label_e*.pkl from <work_dir>/<timestamp>/ps_labels/ instead.')
            return

        n_scenes_done = 0
        all_ps_keys = [k for k, v in ps_labels.items() if len(v.get('gt_boxes', [])) > 0]
        print(f'Points analysis: processing {len(all_ps_keys)} scenes with PS boxes...')
        for key in all_ps_keys:
            ps_entry = ps_labels[key]
            ps_boxes  = ps_entry['gt_boxes']          # (K, 7)
            cls_s     = ps_entry.get('cls_scores',
                                     ps_entry['scores'])  # (K,) fallback
            if len(ps_boxes) == 0:
                continue
            if not os.path.isfile(key):
                continue
            pts_nus = load_points_nus(key)
            pt_counts = count_points_in_boxes(pts_nus, ps_boxes)

            # GT IoU per PS-box
            gt_info = gt_lookup.get(os.path.splitext(os.path.basename(key))[0])
            if gt_info is not None and len(gt_info['cam_boxes']) > 0:
                gt_all = gt_boxes_to_nus(gt_info['cam_boxes'], gt_info['lidar2cam'])
                gt_car = gt_all[gt_info['labels'] == gt_car_label]
            else:
                gt_car = np.zeros((0, 7), dtype=np.float32)
            ious = gt_iou_per_pred(ps_boxes, gt_car, mode='3d')

            all_pt_counts.append(pt_counts)
            all_gt_ious.append(ious)
            all_cls_scores.append(cls_s)
            n_scenes_done += 1
            if n_scenes_done % 500 == 0:
                print(f'  ... {n_scenes_done}/{len(all_ps_keys)} scenes done')

        if all_pt_counts:
            pt_arr  = np.concatenate(all_pt_counts)
            iou_arr = np.concatenate(all_gt_ious)
            cls_arr = np.concatenate(all_cls_scores)

            print(f'\n=== Points-in-box summary ({n_scenes_done} scenes, '
                  f'{len(pt_arr)} boxes) ===')
            print(f'  pt_count:  min={pt_arr.min()}  max={pt_arr.max()}  '
                  f'mean={pt_arr.mean():.1f}  median={np.median(pt_arr):.0f}')
            print(f'  0 pts:   {100*(pt_arr==0).mean():.1f}%  '
                  f'≤5 pts: {100*(pt_arr<=5).mean():.1f}%  '
                  f'≤20 pts: {100*(pt_arr<=20).mean():.1f}%')
            print(f'  gt_iou:    mean={iou_arr.mean():.3f}  '
                  f'median={np.median(iou_arr):.3f}  '
                  f'>0.5: {100*(iou_arr>0.5).mean():.1f}%  '
                  f'>0.25: {100*(iou_arr>0.25).mean():.1f}%')

            pkl_stem = os.path.splitext(os.path.basename(args.ps_label_pkl))[0]
            plot_points_histogram(
                pt_arr,
                os.path.join(args.out_dir, f'points_histogram_{pkl_stem}.png'),
                title=f'Interior point count per PS-box  [{pkl_stem}]  '
                      f'({n_scenes_done} scenes, {len(pt_arr)} boxes)')
            plot_iou_cls_pts_3d_scatter(
                pt_arr, iou_arr, cls_arr,
                os.path.join(args.out_dir, f'points_iou_cls_scatter_{pkl_stem}.html'),
                title=f'PS-box quality: CLS × GT-IoU × Point count  [{pkl_stem}]')
        else:
            print('Points analysis: no scenes with valid data found.')
        return  # early exit — don't run the BEV rendering loop

    # ── Scene selection: from pkl keys, or bin paths derived from gt_lookup ──
    if ps_labels:
        available = list(ps_labels.keys())
    else:
        velodyne_dir = os.path.join(os.path.dirname(args.kitti_info),
                                    'training', 'velodyne_reduced')
        available = [os.path.join(velodyne_dir, k + '.bin')
                     for k in sorted(gt_lookup.keys())]
    n = min(args.num_scenes, len(available))
    selected = random.Random(args.seed).sample(available, n)
    print(f'Selected {n} scene(s) (seed={args.seed})')

    # ── Baseline model ──
    baseline_model = None
    if not args.no_baseline:
        from mmdet3d.apis import init_model
        print(f'Loading baseline model: {args.baseline_ckpt}')
        # init_model falls back to config.class_names when the checkpoint meta
        # lacks 'dataset_meta' (common for custom MT checkpoints).  Inject it
        # from the train dataset metainfo so init_model doesn't crash.
        _cfg = Config.fromfile(args.baseline_config)
        if not hasattr(_cfg, 'class_names'):
            try:
                _cfg.class_names = list(
                    _cfg.train_dataloader.dataset.metainfo.get('classes', ['Car']))
            except Exception:
                _cfg.class_names = ['Car']
        baseline_model = init_model(_cfg, args.baseline_ckpt, device=args.device)
        baseline_model.eval()
        print('Baseline model ready.')

    # Label index for 'Car' in the prediction space (model output).
    # Looked up by name so it works regardless of how many classes the model has.
    pred_car_label = _pred_car_label(baseline_model, ps_labels)

    # ── Stats accumulators ──
    stats = {
        'pseudo':   _empty_stats(),
        'baseline': _empty_stats(),
    }
    bl_all_scores: list = []
    bl_all_iou: list = []
    bl_all_cls: list = []
    bl_all_giou_3d: list = []   # actual GT 3D IoU per prediction
    bl_all_giou_bev: list = []  # actual GT BEV IoU per prediction
    bl_scenes_with_boxes: int = 0

    # ── Adaptive-floor pre-pass ───────────────────────────────────────────────
    # When --floor-percentile is set, pool baseline scores across ALL scenes to
    # turn the percentile into one absolute floor (keep_frac stays per-scene).
    # Raw predictions are cached so the main loop reuses them (no re-inference).
    raw_pred_cache: dict = {}
    global_floor: float = 0.0
    # A fixed absolute --min-floor takes precedence over the adaptive percentile
    # floor (which needs a pooling pre-pass). Set it directly and skip the pre-pass.
    if args.min_floor > 0:
        if args.floor_percentile > 0:
            print(f'Both --min-floor ({args.min_floor:.3f}) and --floor-percentile '
                  f'({args.floor_percentile:g}) set; using the fixed --min-floor.')
            args.floor_percentile = 0.0
        global_floor = args.min_floor
        print(f'Using fixed absolute floor = {global_floor:.3f} (--min-floor)')
    if baseline_model is not None and args.floor_percentile > 0:
        print(f'Adaptive-floor pre-pass: pooling baseline scores over {len(selected)} '
              f'scene(s) for P{args.floor_percentile:g}...')
        pool_list = []
        for key in selected:
            if not os.path.isfile(key):
                continue
            pts_nus = load_points_nus(key)
            rb, rcls, riou, rlab = run_baseline_raw(baseline_model, pts_nus, args.device)
            raw_pred_cache[key] = (rb, rcls, riou, rlab)
            if len(rcls) > 0:
                pool_list.append(rcls)   # CLS score is the ranking/floor score
        pool = np.concatenate(pool_list) if pool_list else np.zeros(0)
        global_floor = (float(np.percentile(pool, args.floor_percentile))
                        if len(pool) > 0 else 0.0)
        print(f'  Global floor = P{args.floor_percentile:g} of {len(pool)} boxes '
              f'= {global_floor:.3f}\n')

    # ── Per-scene loop ──
    for key in selected:
        scene_id = os.path.splitext(os.path.basename(key))[0]
        # print(f'\nScene {scene_id}')

        if not os.path.isfile(key):
            print(f'  WARN: bin file not found at {key!r}, skipping.')
            continue
        pts_nus = load_points_nus(key)

        # GT
        gt_info = gt_lookup.get(scene_id)
        if gt_info is None:
            print(f'  WARN: no GT entry for scene {scene_id}')
            gt_boxes_nus = np.zeros((0, 7), dtype=np.float32)
            gt_labels = np.zeros(0, dtype=np.int64)
        else:
            gt_boxes_nus = gt_boxes_to_nus(
                gt_info['cam_boxes'], gt_info['lidar2cam'])
            gt_labels = gt_info['labels']
            # print(f'  GT:     {len(gt_boxes_nus)} boxes '
            #       f'({(gt_labels==gt_car_label).sum()} Car)')

        # Pseudo-labels (empty when no pkl was loaded)
        ps_entry = ps_labels.get(key, {'gt_boxes': np.zeros((0, 7), dtype=np.float32),
                                       'gt_labels': np.zeros(0, dtype=np.int64),
                                       'scores': np.zeros(0, dtype=np.float32)})
        ps_boxes = ps_entry['gt_boxes']
        ps_scores = ps_entry['scores']
        # if ps_labels:
            # print(f'  Pseudo: {len(ps_boxes)} boxes')

        # Baseline
        bl_boxes, bl_scores, bl_labels, bl_iou_scores, bl_cls_scores = (
            None, None, None, None, None)
        if baseline_model is not None:
            if args.floor_percentile > 0 and key in raw_pred_cache:
                # Reuse cached raw preds + the global percentile floor.
                rb, rcls, riou, rlab = raw_pred_cache[key]
                bl_boxes, bl_scores, bl_labels, bl_iou_scores, bl_cls_scores = (
                    apply_score_filter(
                        rb, rcls, riou, rlab,
                        keep_frac=args.keep_frac, min_floor=global_floor,
                        fallback_k=args.fallback_k, floor_before=args.floor_before))
            else:
                bl_boxes, bl_scores, bl_labels, bl_iou_scores, bl_cls_scores = run_baseline(
                    baseline_model, pts_nus, args.device,
                    keep_frac=args.keep_frac,
                    min_floor=global_floor,
                    fallback_k=args.fallback_k,
                    floor_before=args.floor_before)
            # Fixed interior-point filter on baseline predictions. Applied here
            # so all downstream consumers (score hist, GT-IoU, stats, render)
            # see the same filtered box pool.
            if args.baseline_min_pts > 0 and bl_boxes is not None and len(bl_boxes) > 0:
                bl_pt_counts = count_points_in_boxes(
                    pts_nus, bl_boxes[:, :7].astype(np.float32))
                keep = bl_pt_counts >= args.baseline_min_pts
                bl_boxes = bl_boxes[keep]
                bl_scores = bl_scores[keep]
                bl_labels = bl_labels[keep]
                if bl_iou_scores is not None:
                    bl_iou_scores = bl_iou_scores[keep]
                if bl_cls_scores is not None:
                    bl_cls_scores = bl_cls_scores[keep]
            if len(bl_scores) > 0:
                bl_all_scores.extend(bl_scores.tolist())
                bl_all_cls.extend(bl_cls_scores.tolist())
                if bl_iou_scores is not None:
                    bl_all_iou.extend(bl_iou_scores.tolist())
                bl_scenes_with_boxes += 1
                # Actual GT IoU per prediction (for --gt-iou mode)
                if getattr(args, 'gt_iou', False):
                    # gt_car is built a few lines below; compute it here early
                    _gt_car = (gt_boxes_nus[gt_labels == gt_car_label]
                               if len(gt_labels) > 0
                               else np.zeros((0, 7), dtype=np.float32))
                    _pred_boxes = bl_boxes  # already filtered by threshold
                    mode = getattr(args, 'gt_iou_mode', 'both')
                    if mode in ('3d', 'both'):
                        bl_all_giou_3d.extend(
                            gt_iou_per_pred(_pred_boxes, _gt_car, mode='3d').tolist())
                    if mode in ('bev', 'both'):
                        bl_all_giou_bev.extend(
                            gt_iou_per_pred(_pred_boxes, _gt_car, mode='bev').tolist())

        # Stats or render
        gt_car = (gt_boxes_nus[gt_labels == gt_car_label]
                  if len(gt_labels) > 0 else np.zeros((0, 7), dtype=np.float32))

        if args.stats_only:
            ps_labels_arr = ps_entry.get('gt_labels', np.zeros(len(ps_boxes), dtype=np.int64))
            ps_car_boxes = ps_boxes[ps_labels_arr == pred_car_label]
            # print(f'  GT Car={len(gt_car)}  Pseudo Car={len(ps_car_boxes)}/{len(ps_boxes)}')
            for thr_key, thr_val in (('0.7', 0.7), ('0.50', 0.50)):
                ps_tp, ps_fp, ps_fn = match_boxes(ps_car_boxes, gt_car, thr_val)
                stats['pseudo'][thr_key]['tp'] += ps_tp
                stats['pseudo'][thr_key]['fp'] += ps_fp
                stats['pseudo'][thr_key]['fn'] += ps_fn
                if bl_boxes is not None:
                    bl_car_boxes = bl_boxes[bl_labels == pred_car_label]
                    bl_tp, bl_fp, bl_fn = match_boxes(bl_car_boxes, gt_car, thr_val)
                    stats['baseline'][thr_key]['tp'] += bl_tp
                    stats['baseline'][thr_key]['fp'] += bl_fp
                    stats['baseline'][thr_key]['fn'] += bl_fn
        else:
            out_path = os.path.join(args.out_dir, f'{scene_id}_bev.png')
            render_bev(
                pts_nus,
                gt_boxes_nus, gt_labels,
                ps_boxes, ps_scores,
                bl_boxes, bl_scores,
                scene_id, out_path,
                gt_cars_only=args.gt_cars_only,
                gt_car_label=gt_car_label)

    # ── Final output ──
    if args.stats_only:
        def _row(s, thr):
            tp, fp, fn = s[thr]['tp'], s[thr]['fp'], s[thr]['fn']
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            return f'{tp:5d} {fp:5d} {fn:5d}  {r:.3f}  {p:.3f}'

        print(f'\n=== Stats summary ({n} scenes) ===')
        hdr1 = f'{"":12s}  {"── 3D-IoU@0.7 ──":^30s}  {"── 3D-IoU@0.50 ──":^30s}'
        hdr2 = (f'{"":12s}  {"TP":>5s} {"FP":>5s} {"FN":>5s}  {"R":>5s}  {"P":>5s}'
                f'  {"TP":>5s} {"FP":>5s} {"FN":>5s}  {"R":>5s}  {"P":>5s}')
        print(hdr1)
        print(hdr2)
        for name in ('pseudo', 'baseline'):
            if name == 'baseline' and args.no_baseline:
                continue
            row = (f'{name:12s}  {_row(stats[name], "0.7")}'
                   f'  {_row(stats[name], "0.50")}')
            print(row)
    else:
        print(f'\nDone. {n} PNG(s) written to {os.path.abspath(args.out_dir)}/')

    # ── Baseline score histogram (baseline-only mode, no pkl) ──
    if args.score_hist and not ps_labels:
        os.makedirs(args.out_dir, exist_ok=True)
        if bl_all_cls:
            _use_gt_iou = getattr(args, 'gt_iou', False)
            _gt_iou_mode = getattr(args, 'gt_iou_mode', 'both')
            # Build a human-readable title that names the active filtering scheme.
            _scheme_str = f'keep_frac={args.keep_frac}'
            if args.floor_percentile > 0:
                _scheme_str += f', floor≥{global_floor:.3f} (P{args.floor_percentile:g})'
            elif global_floor > 0:
                _scheme_str += f', floor≥{global_floor:.3f} (fixed)'
            if args.fallback_k > 0:
                _scheme_str += f', fallback_k={args.fallback_k}'
            if args.floor_before:
                _scheme_str += ' [floor-before]'
            if args.baseline_min_pts > 0:
                _scheme_str += f', min_pts={args.baseline_min_pts}'
            _hist_title_base = f'Baseline: per-scene keep-fraction ({_scheme_str})'
            # Threshold overlay: the floor line only.
            _thr_dict = dict(min_cls=global_floor) if global_floor > 0 else None
            if _use_gt_iou:
                # First IoU axis: 3D GT IoU (or BEV if mode='bev')
                _iou1       = bl_all_giou_3d if _gt_iou_mode in ('3d', 'both') else bl_all_giou_bev
                _iou1_label = 'GT 3D IoU'    if _gt_iou_mode in ('3d', 'both') else 'GT BEV IoU'
                # Second IoU axis: BEV GT IoU (only when mode='both')
                _iou2       = bl_all_giou_bev if _gt_iou_mode == 'both' else None
                _iou2_label = 'GT BEV IoU'   if _gt_iou_mode == 'both' else ''
                plot_score_distributions(
                    bl_all_cls, _iou1,
                    os.path.join(args.out_dir, 'score_distributions.png'),
                    title=f'{_hist_title_base} — CLS vs GT IoU',
                    score1_label='CLS score',
                    iou_label=_iou1_label,
                    iou2_scores=_iou2,
                    iou2_label=_iou2_label,
                    thresholds=_thr_dict,
                    scene_stats=(bl_scenes_with_boxes, n))
            else:
                plot_score_distributions(
                    bl_all_cls, bl_all_iou,
                    os.path.join(args.out_dir, 'score_distributions.png'),
                    title=_hist_title_base,
                    score1_label='CLS score',
                    thresholds=_thr_dict,
                    scene_stats=(bl_scenes_with_boxes, n))
        else:
            print('--score-hist: no baseline scores collected '
                  '(add --baseline-config/--baseline-ckpt, or relax '
                  '--keep-frac / --min-floor).')


if __name__ == '__main__':
    main()
