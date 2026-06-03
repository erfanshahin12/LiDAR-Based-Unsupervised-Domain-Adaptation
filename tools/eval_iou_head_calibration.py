#!/usr/bin/env python3
"""
Calibration analysis for the RoI IoU head in VoxelNetBEVRoI.

Compares the model's predicted ``iou_scores_3d`` against the actual
nearest-3D IoU between each prediction and the closest KITTI GT Car box.
The IoU head was trained against ``bbox_overlaps_nearest_3d`` (BEV/nearest-3D)
so that metric is used as the ground-truth target here as well.

All predictions/GT are in the nuScenes coordinate frame (X-right, Y-forward),
matching the coordinate frame the model was trained in.

Usage (from ~/mmdetection3d/):
    python tools/eval_iou_head_calibration.py \\
        --config work_dirs/iou_head_finetune_29may/mt_pretrain_iou_head_finetune.py \\
        --ckpt   work_dirs/iou_head_finetune_29may/epoch_6.pth \\
        --kitti-info data/kitti/kitti_infos_val.pkl \\
        --out-dir work_dirs/iou_head_calibration \\
        --num-scenes 0 --device cuda:0
"""

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# Reuse helpers from visualize_pseudo_labels (same tools/ directory).
sys.path.insert(0, os.path.dirname(__file__))
from visualize_pseudo_labels import build_gt_lookup, gt_boxes_to_nus, load_points_nus

from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes, Box3DMode
from mmdet3d.structures.ops.iou3d_calculator import bbox_overlaps_nearest_3d


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, pts_nus: np.ndarray, device: str):
    """Run model on a single scene and return ALL NMS survivors (no extra filter).

    The model's own test_cfg (score_thr=0.1, nms_thr=0.01) is the only gate.

    Returns:
        boxes    (K, 7) float32  nuScenes frame
        cls      (K,)   float32  scores_3d (raw cls)
        iou_pred (K,)   float32  iou_scores_3d from RoI head  (None if head absent)
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

    inst = pred_list[0].pred_instances_3d
    boxes   = inst.bboxes_3d.tensor.cpu().numpy()[:, :7].astype(np.float32)
    cls     = inst.scores_3d.cpu().numpy().astype(np.float32)
    iou_raw = getattr(inst, 'iou_scores_3d', None)
    iou_pred = iou_raw.cpu().numpy().astype(np.float32) if iou_raw is not None else None
    return boxes, cls, iou_pred


# ── IoU computation ────────────────────────────────────────────────────────────

def compute_actual_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Nearest-3D IoU between each prediction and the best-matching GT Car.

    Uses the same metric as the RoI IoU head's training target.

    Args:
        pred_boxes: (N, 7) in nuScenes/LIDAR frame.
        gt_boxes:   (M, 7) in nuScenes/LIDAR frame (Car GT only).

    Returns:
        (N,) float32 — per-prediction max IoU over all GT boxes (0 if M=0).
    """
    if len(pred_boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(gt_boxes) == 0:
        return np.zeros(len(pred_boxes), dtype=np.float32)

    pred_t = torch.from_numpy(pred_boxes).float()
    gt_t   = torch.from_numpy(gt_boxes).float()
    # (N_pred, N_gt) matrix; .max(dim=1) = per-prediction best-GT match.
    iou_mat = bbox_overlaps_nearest_3d(pred_t, gt_t, mode='iou',
                                       is_aligned=False, coordinate='lidar')
    return iou_mat.max(dim=1)[0].cpu().numpy().astype(np.float32)


# ── Plotting helpers ───────────────────────────────────────────────────────────

_DARK = '#0a0a0a'
_PANEL = '#111111'

def _style(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444444')
    ax.grid(True, alpha=0.15, color='white', linewidth=0.5)


def plot_scatter(pred_iou, actual_iou, out_path: str, r: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(_DARK)
    _style(ax)
    hb = ax.hexbin(actual_iou, pred_iou, gridsize=40, cmap='plasma',
                   extent=[0, 1, 0, 1], mincnt=1, bins='log')
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.ax.tick_params(colors='white', labelsize=7)
    cb.outline.set_edgecolor('#444444')
    cb.set_label('log count', color='white', fontsize=8)
    ax.plot([0, 1], [0, 1], 'w--', linewidth=1.0, alpha=0.6, label='perfect calibration')
    ax.set_xlabel('Actual nearest-3D IoU', color='white', fontsize=10)
    ax.set_ylabel('Predicted IoU (RoI head)', color='white', fontsize=10)
    ax.set_title(f'IoU Head Calibration  (Pearson r = {r:.3f})', color='white', fontsize=11)
    ax.legend(fontsize=8, facecolor='#222222', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_calibration_curve(pred_iou, actual_iou, out_path: str) -> None:
    bins = np.arange(0.0, 1.05, 0.05)
    bin_centres, mean_pred, mean_actual, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred_iou >= lo) & (pred_iou < hi)
        n = mask.sum()
        if n == 0:
            continue
        bin_centres.append((lo + hi) / 2)
        mean_pred.append(pred_iou[mask].mean())
        mean_actual.append(actual_iou[mask].mean())
        counts.append(n)

    bin_centres = np.array(bin_centres)
    mean_pred   = np.array(mean_pred)
    mean_actual = np.array(mean_actual)
    counts      = np.array(counts, dtype=float)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    fig.patch.set_facecolor(_DARK)
    ax1.set_facecolor(_PANEL); ax2.set_facecolor(_PANEL)

    width = 0.03
    ax2.bar(bin_centres, counts, width=width, color='#4488cc', alpha=0.35, label='count')
    ax1.plot([0, 1], [0, 1], 'w--', linewidth=1.0, alpha=0.6, label='perfect calibration')
    ax1.plot(mean_pred, mean_actual, 'o-', color='tomato', linewidth=2,
             markersize=5, label='mean actual IoU per predicted bin')

    ax1.set_xlabel('Mean predicted IoU (bin centre)', color='white', fontsize=10)
    ax1.set_ylabel('Mean actual IoU', color='white', fontsize=10)
    ax2.set_ylabel('Box count', color='#4488cc', fontsize=9)
    ax1.set_title('Calibration curve: predicted IoU bins → actual IoU', color='white', fontsize=11)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    for ax in (ax1, ax2):
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444444')
    ax1.grid(True, alpha=0.15, color='white', linewidth=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8, facecolor='#222222', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_error_histogram(pred_iou, actual_iou, out_path: str) -> None:
    err = pred_iou - actual_iou
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(_DARK); _style(ax)
    bins = np.linspace(-1, 1, 81)
    ax.hist(err, bins=bins, color='#ff7f50', alpha=0.85, edgecolor='none')
    ax.axvline(0,           color='white', linestyle='--', linewidth=1.2, alpha=0.7, label='zero error')
    ax.axvline(err.mean(),  color='yellow', linestyle='--', linewidth=1.2,
               label=f'mean={err.mean():.3f}')
    ax.axvline( 0.1, color='lime', linestyle=':', linewidth=1.0, alpha=0.7, label='±0.1')
    ax.axvline(-0.1, color='lime', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('Predicted IoU − Actual IoU', color='white', fontsize=10)
    ax.set_ylabel('Box count', color='white', fontsize=10)
    ax.set_title('IoU prediction error distribution', color='white', fontsize=11)
    ax.legend(fontsize=8, facecolor='#222222', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_cls_vs_actual(cls_scores, actual_iou, out_path: str) -> None:
    r_cls = float(np.corrcoef(cls_scores, actual_iou)[0, 1]) if len(cls_scores) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(_DARK); _style(ax)
    hb = ax.hexbin(cls_scores, actual_iou, gridsize=40, cmap='viridis',
                   extent=[0, 1, 0, 1], mincnt=1, bins='log')
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.ax.tick_params(colors='white', labelsize=7)
    cb.outline.set_edgecolor('#444444')
    cb.set_label('log count', color='white', fontsize=8)
    ax.set_xlabel('CLS score (scores_3d)', color='white', fontsize=10)
    ax.set_ylabel('Actual nearest-3D IoU', color='white', fontsize=10)
    ax.set_title(f'CLS score vs actual IoU  (r = {r_cls:.3f})', color='white', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f'Saved → {out_path}')


# ── Stats summary ──────────────────────────────────────────────────────────────

def print_and_save_stats(pred_iou, actual_iou, cls_scores, out_path: str) -> None:
    n = len(pred_iou)
    err = pred_iou - actual_iou
    mae  = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    r    = float(np.corrcoef(pred_iou, actual_iou)[0, 1]) if n > 1 else 0.0
    r_cls = float(np.corrcoef(cls_scores, actual_iou)[0, 1]) if n > 1 else 0.0
    bias = float(err.mean())
    f01  = float((np.abs(err) < 0.1).mean())
    f02  = float((np.abs(err) < 0.2).mean())

    lines = [
        f'=== IoU Head Calibration Summary ({n} predictions) ===',
        f'',
        f'  Pearson r (pred IoU vs actual IoU): {r:.4f}',
        f'  Pearson r (CLS score vs actual IoU): {r_cls:.4f}',
        f'  MAE:   {mae:.4f}',
        f'  RMSE:  {rmse:.4f}',
        f'  Bias (pred − actual):  {bias:+.4f}',
        f'  |error| < 0.10: {100*f01:.1f}%',
        f'  |error| < 0.20: {100*f02:.1f}%',
        f'  Mean predicted IoU: {pred_iou.mean():.4f}',
        f'  Mean actual    IoU: {actual_iou.mean():.4f}',
        f'',
        f'--- Per-bin breakdown (bins of predicted IoU) ---',
        f'  {"bin":>12s}  {"count":>7s}  {"mean_pred":>9s}  {"mean_actual":>11s}  {"MAE":>6s}',
    ]
    bins = np.arange(0.0, 1.05, 0.1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred_iou >= lo) & (pred_iou < hi)
        nc = mask.sum()
        if nc == 0:
            continue
        mp = pred_iou[mask].mean()
        ma = actual_iou[mask].mean()
        me = float(np.abs(pred_iou[mask] - actual_iou[mask]).mean())
        lines.append(f'  [{lo:.1f} – {hi:.1f}):  {nc:7d}  {mp:9.4f}  {ma:11.4f}  {me:6.4f}')

    text = '\n'.join(lines)
    print(text)
    with open(out_path, 'w') as f:
        f.write(text + '\n')
    print(f'\nStats saved → {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Calibration analysis for the RoI IoU head on KITTI val set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Model config (e.g. mt_pretrain_iou_head_finetune.py)')
    p.add_argument('--ckpt', required=True,
                   help='Model checkpoint (.pth)')
    p.add_argument('--kitti-info', default='data/kitti/kitti_infos_val.pkl',
                   help='KITTI info pkl (val or train)')
    p.add_argument('--kitti-root', default='data/kitti/',
                   help='KITTI data root (parent of training/)')
    p.add_argument('--out-dir', default='work_dirs/iou_head_calibration',
                   help='Directory for output plots and stats')
    p.add_argument('--num-scenes', type=int, default=0,
                   help='Number of scenes to evaluate (0 = all)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model ──
    print(f'Loading model: {args.ckpt}')
    model = init_model(args.config, args.ckpt, device=args.device)
    model.eval()
    has_iou_head = hasattr(model, 'bev_roi_iou_head') and model.bev_roi_iou_head is not None
    print(f'Model loaded.  RoI IoU head present: {has_iou_head}')

    # ── Load GT ──
    print(f'Loading KITTI GT from {args.kitti_info} ...')
    gt_lookup, gt_car_label = build_gt_lookup(args.kitti_info)
    velodyne_dir = os.path.join(args.kitti_root, 'training', 'velodyne_reduced')

    # ── Scene selection ──
    all_scenes = sorted(gt_lookup.keys())
    if args.num_scenes > 0:
        all_scenes = random.Random(args.seed).sample(all_scenes, min(args.num_scenes, len(all_scenes)))
    print(f'Evaluating {len(all_scenes)} scenes ...')

    # ── Per-scene loop ──
    all_cls, all_pred_iou, all_actual_iou = [], [], []
    n_no_iou_head = 0

    for scene_id in all_scenes:
        bin_path = os.path.join(velodyne_dir, scene_id + '.bin')
        if not os.path.isfile(bin_path):
            print(f'  WARN: {bin_path} not found, skipping.')
            continue

        pts_nus = load_points_nus(bin_path)
        boxes, cls, iou_pred = run_inference(model, pts_nus, args.device)

        if iou_pred is None:
            n_no_iou_head += 1
            continue

        # GT boxes in nuScenes frame, Car class only.
        info = gt_lookup[scene_id]
        gt_nus = gt_boxes_to_nus(info['cam_boxes'], info['lidar2cam'])
        gt_car_mask = info['labels'] == gt_car_label
        gt_car_boxes = gt_nus[gt_car_mask] if gt_car_mask.any() else np.zeros((0, 7), dtype=np.float32)

        actual_iou = compute_actual_iou(boxes, gt_car_boxes)

        all_cls.extend(cls.tolist())
        all_pred_iou.extend(iou_pred.tolist())
        all_actual_iou.extend(actual_iou.tolist())

    if n_no_iou_head > 0:
        print(f'WARNING: {n_no_iou_head} scenes had no iou_scores_3d (RoI head absent?).')

    if not all_pred_iou:
        print('ERROR: no predictions collected. Check model path and data.')
        return

    pred_iou   = np.array(all_pred_iou,   dtype=np.float32)
    actual_iou = np.array(all_actual_iou, dtype=np.float32)
    cls_scores = np.array(all_cls,        dtype=np.float32)

    r = float(np.corrcoef(pred_iou, actual_iou)[0, 1]) if len(pred_iou) > 1 else 0.0

    # ── Plots ──
    plot_scatter(
        pred_iou, actual_iou,
        os.path.join(args.out_dir, 'scatter_iou_calibration.png'), r)
    plot_calibration_curve(
        pred_iou, actual_iou,
        os.path.join(args.out_dir, 'calibration_curve.png'))
    plot_error_histogram(
        pred_iou, actual_iou,
        os.path.join(args.out_dir, 'error_histogram.png'))
    plot_cls_vs_actual(
        cls_scores, actual_iou,
        os.path.join(args.out_dir, 'score_vs_actual_iou.png'))

    # ── Stats ──
    print_and_save_stats(
        pred_iou, actual_iou, cls_scores,
        os.path.join(args.out_dir, 'calibration_stats.txt'))


if __name__ == '__main__':
    main()
