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

def build_gt_lookup(info_path: str) -> dict:
    """Return {scene_id_str → dict} from kitti_infos_train.pkl.

    scene_id_str is the zero-padded 6-digit frame id, e.g. '000042'.
    Each value: {'cam_boxes': (N,7) float32, 'labels': (N,) int64,
                 'lidar2cam': (4,4) float64}
    """
    with open(info_path, 'rb') as f:
        d = pickle.load(f)
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
    return lookup


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

def run_baseline(model, pts_nus: np.ndarray, device: str,
                 score_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Run baseline VoxelNetBEVRoI on a single scene.

    Pattern mirrors PseudoLabelRefreshHook._run_teacher_inference.
    Returns (K, 7) boxes and (K,) scores in nuScenes frame, filtered by score_thr.
    """
    pts_tensor = torch.from_numpy(pts_nus).float()
    sample = Det3DDataSample()
    sample.set_metainfo({
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR,
    })
    with torch.no_grad():
        data = model.data_preprocessor(
            {'inputs': {'points': [pts_tensor]},
             'data_samples': [sample]},
            training=False)
        pred_list = model.predict(data['inputs'], data['data_samples'])
    pred = pred_list[0]
    inst = pred.pred_instances_3d
    scores = inst.scores_3d.cpu().numpy()
    boxes = inst.bboxes_3d.tensor.cpu().numpy()[:, :7]
    mask = scores >= score_thr
    return boxes[mask], scores[mask]


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
               gt_cars_only: bool = False) -> None:
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
    gt_car_boxes = (gt_boxes[gt_labels == 2]
                    if len(gt_labels) > 0 else np.zeros((0, 7), dtype=np.float32))
    ps_iou = bev_iou_matrix(ps_boxes, gt_car_boxes)
    bl_arr = bl_boxes if bl_boxes is not None else np.zeros((0, 7), dtype=np.float32)
    bl_iou = bev_iou_matrix(bl_arr, gt_car_boxes)

    legend_items = []

    # ── GT boxes ──
    n_gt_car = int((gt_labels == 2).sum()) if len(gt_labels) else 0
    n_gt_other = len(gt_labels) - n_gt_car
    for box, label in zip(gt_boxes, gt_labels):
        is_car = (label == 2)
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
        description='Visualise pseudo-labels vs GT vs baseline (BEV PNG)')
    p.add_argument('--ps-label-pkl',
                   help='Path to ps_label_e*.pkl from PseudoLabelRefreshHook')
    p.add_argument('--num-scenes', type=int, default=10,
                   help='Number of random scenes to visualise')
    p.add_argument('--out-dir', default='vis_ps_labels',
                   help='Directory for output PNGs')
    p.add_argument('--kitti-info',
                   default='data/kitti/kitti_infos_train.pkl',
                   help='KITTI train info pkl (GT source)')
    p.add_argument('--baseline-config', required=False, default=None,
                   help='Config .py for the baseline model '
                        '(required unless --no-baseline)')
    p.add_argument('--baseline-ckpt', required=False, default=None,
                   help='Checkpoint .pth for the baseline model '
                        '(required unless --no-baseline)')
    p.add_argument('--baseline-score-thr', type=float, default=0.3,
                   help='Score threshold for baseline predictions (default 0.3). '
                        'Independent of the pseudo-label conf_threshold — '
                        'ps_labels are already pre-filtered before saving.')
    p.add_argument('--no-baseline', action='store_true',
                   help='Skip baseline overlay (no GPU/model needed)')
    p.add_argument('--gt-cars-only', action='store_true',
                   help='Show only Car GT boxes (hide other classes)')
    p.add_argument('--stats-only', action='store_true',
                   help='Print per-scene TP/FP/FN/recall/precision at IoU 0.25 '
                        'and 0.5; skip PNG generation')
    p.add_argument('--device', default='cuda:0',
                   help='Torch device for baseline inference')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for scene sampling')
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


def main():
    args = parse_args()

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

    # ── Load pseudo-labels ──
    with open(args.ps_label_pkl, 'rb') as f:
        ps_labels = pickle.load(f)
    print(f'Loaded {len(ps_labels)} pseudo-label entries '
          f'from {args.ps_label_pkl}')

    # ── Augmentation sanity check ──
    mt_config_path = _find_mt_config(args.ps_label_pkl)
    if mt_config_path and os.path.isfile(mt_config_path):
        print(f'MT config (derived from pkl path): {mt_config_path}')
        _check_augmentation_warning(mt_config_path)
    else:
        print('  (MT config not found alongside pkl; skipping aug check)')

    # ── GT lookup ──
    print(f'Loading KITTI GT from {args.kitti_info}...')
    gt_lookup = build_gt_lookup(args.kitti_info)

    # ── Random scene selection ──
    available = list(ps_labels.keys())
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

    # ── Stats accumulators ──
    stats = {
        'pseudo':   _empty_stats(),
        'baseline': _empty_stats(),
    }

    # ── Per-scene loop ──
    for key in selected:
        scene_id = os.path.splitext(os.path.basename(key))[0]
        print(f'\nScene {scene_id}')

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
            print(f'  GT:     {len(gt_boxes_nus)} boxes '
                  f'({(gt_labels==2).sum()} Car)')

        # Pseudo-labels
        ps_entry = ps_labels[key]
        ps_boxes = ps_entry['gt_boxes']
        ps_scores = ps_entry['scores']
        print(f'  Pseudo: {len(ps_boxes)} boxes')

        # Baseline
        bl_boxes, bl_scores = None, None
        if baseline_model is not None:
            bl_boxes, bl_scores = run_baseline(
                baseline_model, pts_nus, args.device,
                args.baseline_score_thr)
            print(f'  Baseline: {len(bl_boxes)} boxes '
                  f'(thr={args.baseline_score_thr})')

        # Stats or render
        gt_car = (gt_boxes_nus[gt_labels == 2]
                  if len(gt_labels) > 0 else np.zeros((0, 7), dtype=np.float32))

        if args.stats_only:
            print(f'  GT Car={len(gt_car)}')
            for thr_key, thr_val in (('0.25', 0.25), ('0.50', 0.50)):
                ps_tp, ps_fp, ps_fn = match_boxes(ps_boxes, gt_car, thr_val)
                stats['pseudo'][thr_key]['tp'] += ps_tp
                stats['pseudo'][thr_key]['fp'] += ps_fp
                stats['pseudo'][thr_key]['fn'] += ps_fn
                if bl_boxes is not None:
                    bl_tp, bl_fp, bl_fn = match_boxes(bl_boxes, gt_car, thr_val)
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
                gt_cars_only=args.gt_cars_only)

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


if __name__ == '__main__':
    main()
