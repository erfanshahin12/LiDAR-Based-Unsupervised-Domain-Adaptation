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


def match_boxes(pred: np.ndarray, gt: np.ndarray,
                iou_thr: float) -> tuple[int, int, int]:
    """Greedy BEV IoU matching. Returns (TP, FP, FN)."""
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    iou = bev_iou_matrix(pred, gt)
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
    pts_nus[:, 2] = pts[:, 2] - 0.11    # align KITTI ground (−1.73) to nuScenes (−1.84)
    pts_nus[:, 3] = pts[:, 3] * 255.0
    return pts_nus


# ── Baseline inference ─────────────────────────────────────────────────────────

def _apply_kitti_test_cfg(head, overrides: dict) -> dict:
    """Override test_cfg keys and return originals for later restoration."""
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
                 hybrid_thr: float, iou_weight: float,
                 min_cls_thr: float = 0.0,
                 min_iou_thr: float = 0.0,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray | None, np.ndarray]:
    """Run baseline on a single scene and apply threshold filtering.

    The test_cfg is overridden with _KITTI_TEST_CFG (matching
    test_kitti_nuspretrained_pointpillars.py) so that inference is consistent
    with KITTI evaluation regardless of which training config was loaded.
    Specifically, score_type='cls' guarantees that scores_3d is the raw
    classification sigmoid — not the hybrid blend — so hybrid can be computed
    explicitly from raw components below.

    Three independent constraints:
      hybrid = iou_weight*iou_scores + (1-iou_weight)*cls_scores >= hybrid_thr
      cls_scores >= min_cls_thr   (0.0 = off)
      iou_scores >= min_iou_thr   (0.0 = off)

    Returns (boxes (K,7), hybrid (K,), labels (K,), iou_scores (K,)|None,
             cls_scores (K,)) in nuScenes frame.
    """
    pts_tensor = torch.from_numpy(pts_nus).float()
    sample = Det3DDataSample()
    sample.set_metainfo({
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR,
    })

    head = model.bbox_head
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

    # Hybrid computed explicitly from raw components — matches filter_teacher_predictions.
    if iou_scores is not None and iou_weight > 0:
        hybrid = iou_weight * iou_scores + (1.0 - iou_weight) * cls_scores
    else:
        hybrid = cls_scores.copy()

    mask = hybrid >= hybrid_thr
    if iou_scores is not None and min_iou_thr > 0:
        mask = mask & (iou_scores >= min_iou_thr)
    if min_cls_thr > 0:
        mask = mask & (cls_scores >= min_cls_thr)

    return (boxes[mask], hybrid[mask], labels[mask],
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

    head = model.bbox_head
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
    g = p.add_argument_group('Thresholds & scoring')
    g.add_argument('--hybrid-thr', type=float, default=0.3,
                   help='Minimum hybrid score: iou_weight*IoU + (1-iou_weight)*CLS')
    g.add_argument('--iou-weight', type=float, default=0.5,
                   help='IoU weight in hybrid score [0,1]. 0 = CLS-only.')
    g.add_argument('--min-cls-thr', type=float, default=0.0,
                   help='Minimum raw CLS score floor applied independently of hybrid '
                        '(0.0 = off)')
    g.add_argument('--min-iou-thr', type=float, default=0.0,
                   help='Minimum raw IoU head score floor applied independently of hybrid '
                        '(0.0 = off)')
    g.add_argument('--sweep-iou-weights', default='0.5,0.6,0.7,0.8',
                   help='Comma-separated iou_weight values for --sweep-from-cache.')
    g.add_argument('--sweep-hybrid-thrs', default='0.30,0.35,0.40,0.45,0.50',
                   help='Comma-separated hybrid_thr values for --sweep-from-cache.')
    g.add_argument('--sweep-floors', default='0.10,0.15,0.20,0.25,0.30',
                   help='Comma-separated symmetric floor values (min_cls=min_iou) '
                        'for --sweep-from-cache.')
    g.add_argument('--sweep-cov-target', type=float, default=0.85,
                   help='Coverage target for penalised score in sweep (default 0.85).')
    g.add_argument('--sweep-top-n', type=int, default=20,
                   help='Number of top results to display in sweep table.')

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
    g.add_argument('--dump-raw-preds', metavar='PATH', default=None,
                   help='Run inference on all scenes and save raw (unfiltered) predictions '
                        'to PATH as a pkl cache for use with --sweep-from-cache. '
                        'Requires --baseline-config/--baseline-ckpt. Early exit after dump.')
    g.add_argument('--sweep-from-cache', metavar='PATH', default=None,
                   help='Load raw-prediction cache from PATH (created by --dump-raw-preds) '
                        'and sweep threshold combinations to find optimal operating points. '
                        'Early exit after printing the sweep table.')

    return p.parse_args()


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
    return {'0.25': {'tp': 0, 'fp': 0, 'fn': 0},
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
                             thresholds: dict | None = None,
                             scene_stats: tuple[int, int] = (0, 0)) -> None:
    """Save a histogram of score distributions.

    Args:
        cls_scores: list or array of the first score (CLS or hybrid)
        iou_scores: list or array of IoU scores (may be empty)
        out_path: path for the output PNG
        title: figure suptitle
        score1_label: label for the first score axis
        thresholds: optional dict with keys min_cls, min_iou, hybrid_thr, iou_weight.
            When provided, draws constraint lines on the histograms and hexbin.
        scene_stats: (n_with_boxes, n_total) for scene coverage reporting.
    """
    cls = np.asarray(cls_scores, dtype=np.float32)
    has_iou = len(iou_scores) > 0
    iou = np.asarray(iou_scores, dtype=np.float32) if has_iou else None

    # Layout: [CLS hist] [IoU hist] [CLS vs IoU joint] when both scores exist,
    # otherwise just [CLS hist].
    n_plots = 3 if has_iou else 1
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

    thr = thresholds or {}
    _hist(axes[0], cls, '#4db8ff', f'{score1_label} distribution',
          vline=thr.get('min_cls'))
    if has_iou:
        _hist(axes[1], iou, '#ff7f50', 'IoU score distribution',
              vline=thr.get('min_iou'))

        # Joint hexbin — meaningful only when score1 is raw CLS, not hybrid
        hb = axes[2].hexbin(cls, iou, gridsize=30, cmap='plasma',
                            extent=[0, 1, 0, 1], mincnt=1)
        cb = fig.colorbar(hb, ax=axes[2], pad=0.02)
        cb.ax.tick_params(colors='white', labelsize=7)
        cb.outline.set_edgecolor('#444444')
        corr = float(np.corrcoef(cls, iou)[0, 1])
        axes[2].set_title(f'{score1_label} vs IoU  (r={corr:.3f})',
                          color='white', fontsize=11)
        axes[2].set_xlabel(score1_label, color='white', fontsize=9)
        axes[2].set_ylabel('IoU score', color='white', fontsize=9)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        # Diagonal reference: perfect CLS=IoU agreement
        axes[2].plot([0, 1], [0, 1], color='white', linestyle=':', linewidth=0.8,
                     alpha=0.5, label='CLS = IoU')
        # Constraint overlays — draw the elbow boundary when thresholds are set
        if thr:
            min_cls = thr.get('min_cls', 0.0)
            min_iou = thr.get('min_iou', 0.0)
            ht = thr.get('hybrid_thr', 0.0)
            w = thr.get('iou_weight', 0.0)
            if min_cls > 0:
                axes[2].axvline(min_cls, color='cyan', linestyle='--',
                                linewidth=1.0, alpha=0.85,
                                label=f'min CLS={min_cls:.2f}')
            if min_iou > 0:
                axes[2].axhline(min_iou, color='lime', linestyle='--',
                                linewidth=1.0, alpha=0.85,
                                label=f'min IoU={min_iou:.2f}')
            if ht > 0 and w > 0:
                # hybrid = w*iou + (1-w)*cls = ht  →  iou = (ht - (1-w)*cls) / w
                cx = np.linspace(0.0, 1.0, 300)
                iy = (ht - (1.0 - w) * cx) / w
                vis = (iy >= 0) & (iy <= 1)
                axes[2].plot(cx[vis], iy[vis], color='orange', linewidth=1.2,
                             alpha=0.85, label=f'hybrid≥{ht:.2f} (w={w})')
        axes[2].legend(fontsize=7, facecolor='#222222',
                       edgecolor='white', labelcolor='white')

    fig.suptitle(title, color='white', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close(fig)
    print(f'Score distribution plot saved → {out_path}')

    # Text summary
    print(f'\n=== Score distributions ({len(cls)} boxes) ===')
    for name, data in [(score1_label, cls)] + ([('IoU', iou)] if has_iou else []):
        print(f'\n  {name}:  min={data.min():.3f}  max={data.max():.3f}  '
              f'mean={data.mean():.3f}  median={np.median(data):.3f}  std={data.std():.3f}')
        bins_e = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        counts, _ = np.histogram(data, bins=bins_e)
        for lo, hi, c in zip(bins_e, bins_e[1:], counts):
            bar = '█' * int(c / max(counts) * 30)
            print(f'    [{lo:.1f}-{hi:.1f}]: {c:6d}  {bar}')
    if has_iou:
        print(f'\n  Pearson r(CLS, IoU) = {corr:.4f}')
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
    model = init_model(args.baseline_config, args.baseline_ckpt, device=args.device)
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


def _sweep_thresholds(args) -> None:
    """Load raw-prediction cache and sweep threshold combinations."""
    print(f'Loading cache: {args.sweep_from_cache}')
    with open(args.sweep_from_cache, 'rb') as f:
        cache = pickle.load(f)

    scenes = cache['scenes']
    pred_car_label = cache['pred_car_label']
    n_total = len(scenes)
    print(f'  {n_total} scenes, pred_car_label={pred_car_label}')

    iou_weights   = _parse_floats(args.sweep_iou_weights)
    hybrid_thrs   = _parse_floats(args.sweep_hybrid_thrs)
    floors        = _parse_floats(args.sweep_floors)
    cov_target    = args.sweep_cov_target
    top_n         = args.sweep_top_n
    n_combos      = len(iou_weights) * len(hybrid_thrs) * len(floors)
    print(f'  Sweeping {n_combos} combinations '
          f'({len(iou_weights)} iou_w × {len(hybrid_thrs)} hybrid_thr × '
          f'{len(floors)} floor)...\n')

    # Reference point from current args (highlighted in table)
    ref = (round(args.iou_weight, 4),
           round(args.hybrid_thr, 4),
           round(min(args.min_cls_thr, args.min_iou_thr), 4))

    def _f05(p, r):
        return 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0

    results = []
    for w in iou_weights:
        for ht in hybrid_thrs:
            for fl in floors:
                tp25 = fp25 = fn25 = 0
                tp50 = fp50 = fn50 = 0
                n_with = 0
                total_boxes = 0
                for sc in scenes:
                    cls_s = sc['cls_scores']
                    iou_s = sc.get('iou_scores')
                    boxes = sc['boxes']
                    labels = sc['labels']
                    gt_car = sc['gt_car_boxes']

                    hybrid = (w * iou_s + (1.0 - w) * cls_s
                              if iou_s is not None and w > 0 else cls_s)
                    mask = hybrid >= ht
                    if fl > 0:
                        mask = mask & (cls_s >= fl)
                        if iou_s is not None:
                            mask = mask & (iou_s >= fl)

                    car_mask = mask & (labels == pred_car_label)
                    pred_car = boxes[car_mask]
                    total_boxes += int(car_mask.sum())
                    if len(pred_car) > 0:
                        n_with += 1

                    t, f_p, f_n = match_boxes(pred_car, gt_car, 0.25)
                    tp25 += t; fp25 += f_p; fn25 += f_n
                    t, f_p, f_n = match_boxes(pred_car, gt_car, 0.50)
                    tp50 += t; fp50 += f_p; fn50 += f_n

                p50 = tp50 / (tp50 + fp50) if (tp50 + fp50) > 0 else 0.0
                r50 = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
                cov = n_with / n_total if n_total > 0 else 0.0
                f = _f05(p50, r50)
                score = f * min(1.0, cov / cov_target)
                results.append((score, f, p50, r50, cov,
                                 total_boxes, w, ht, fl))

    results.sort(reverse=True)
    shown = results[:top_n]

    hdr = (f'{"Rank":>4}  {"iou_w":>5}  {"floor":>5}  {"hybrid":>6}  '
           f'{"P@0.5":>6}  {"R@0.5":>6}  {"F_0.5":>5}  '
           f'{"Cov%":>5}  {"Boxes":>6}  {"Score":>6}')
    sep = '─' * len(hdr)
    print(f'=== Threshold sweep — top {top_n} of {n_combos} by '
          f'F_0.5 × cov_factor (cov_target={cov_target:.0%}) ===')
    print(f'  (model test_cfg score_thr is an implicit CLS floor not shown here)')
    print(sep)
    print(hdr)
    print(sep)
    for rank, (score, f, p, r, cov, nb, w, ht, fl) in enumerate(shown, 1):
        marker = ' ←' if (round(w,4), round(ht,4), round(fl,4)) == ref else ''
        print(f'{rank:>4}  {w:>5.2f}  {fl:>5.2f}  {ht:>6.3f}  '
              f'{p:>6.3f}  {r:>6.3f}  {f:>5.3f}  '
              f'{cov*100:>5.1f}  {nb:>6}  {score:>6.3f}{marker}')
    print(sep)

    # CSV export alongside the cache
    csv_path = args.sweep_from_cache.replace('.pkl', '_sweep.csv')
    with open(csv_path, 'w') as fout:
        fout.write('iou_w,floor,hybrid_thr,precision,recall,f05,'
                   'coverage,n_boxes,score\n')
        for score, f, p, r, cov, nb, w, ht, fl in results:
            fout.write(f'{w},{fl},{ht},{p:.4f},{r:.4f},{f:.4f},'
                       f'{cov:.4f},{nb},{score:.4f}\n')
    print(f'\nFull sweep results saved → {csv_path}')


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
        baseline_model = init_model(
            args.baseline_config, args.baseline_ckpt, device=args.device)
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
    bl_scenes_with_boxes: int = 0

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
            bl_boxes, bl_scores, bl_labels, bl_iou_scores, bl_cls_scores = run_baseline(
                baseline_model, pts_nus, args.device,
                args.hybrid_thr, args.iou_weight,
                args.min_cls_thr, args.min_iou_thr)
            thr_info = (f'hybrid≥{args.hybrid_thr}, w={args.iou_weight}'
                        + (f', cls≥{args.min_cls_thr}' if args.min_cls_thr > 0 else '')
                        + (f', iou≥{args.min_iou_thr}' if args.min_iou_thr > 0 else ''))
            # print(f'  Baseline: {len(bl_boxes)} boxes ({thr_info})')
            if len(bl_scores) > 0:
                bl_all_scores.extend(bl_scores.tolist())
                bl_all_cls.extend(bl_cls_scores.tolist())
                if bl_iou_scores is not None:
                    bl_all_iou.extend(bl_iou_scores.tolist())
                bl_scenes_with_boxes += 1

        # Stats or render
        gt_car = (gt_boxes_nus[gt_labels == gt_car_label]
                  if len(gt_labels) > 0 else np.zeros((0, 7), dtype=np.float32))

        if args.stats_only:
            ps_labels_arr = ps_entry.get('gt_labels', np.zeros(len(ps_boxes), dtype=np.int64))
            ps_car_boxes = ps_boxes[ps_labels_arr == pred_car_label]
            # print(f'  GT Car={len(gt_car)}  Pseudo Car={len(ps_car_boxes)}/{len(ps_boxes)}')
            for thr_key, thr_val in (('0.25', 0.25), ('0.50', 0.50)):
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
        hdr1 = f'{"":12s}  {"── IoU@0.25 ──":^30s}  {"── IoU@0.50 ──":^30s}'
        hdr2 = (f'{"":12s}  {"TP":>5s} {"FP":>5s} {"FN":>5s}  {"R":>5s}  {"P":>5s}'
                f'  {"TP":>5s} {"FP":>5s} {"FN":>5s}  {"R":>5s}  {"P":>5s}')
        print(hdr1)
        print(hdr2)
        for name in ('pseudo', 'baseline'):
            if name == 'baseline' and args.no_baseline:
                continue
            row = (f'{name:12s}  {_row(stats[name], "0.25")}'
                   f'  {_row(stats[name], "0.50")}')
            print(row)
    else:
        print(f'\nDone. {n} PNG(s) written to {os.path.abspath(args.out_dir)}/')

    # ── Baseline score histogram (baseline-only mode, no pkl) ──
    if args.score_hist and not ps_labels:
        os.makedirs(args.out_dir, exist_ok=True)
        if bl_all_cls:
            plot_score_distributions(
                bl_all_cls, bl_all_iou,
                os.path.join(args.out_dir, 'score_distributions.png'),
                title='Baseline score distributions',
                score1_label='CLS score',
                thresholds=dict(
                    min_cls=args.min_cls_thr,
                    min_iou=args.min_iou_thr,
                    hybrid_thr=args.hybrid_thr,
                    iou_weight=args.iou_weight),
                scene_stats=(bl_scenes_with_boxes, n))
        else:
            print('--score-hist: no baseline scores collected '
                  '(add --baseline-config/--baseline-ckpt or lower --hybrid-thr).')


if __name__ == '__main__':
    main()
