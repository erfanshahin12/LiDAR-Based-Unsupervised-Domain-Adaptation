#!/usr/bin/env python3
"""
Calibration analysis for the per-anchor IoU quality head (conv_iou) in Anchor3DHead.

Compares the model's predicted ``iou_scores_3d`` against the actual *rotated*
3D IoU between each prediction and the closest KITTI GT Car box.  The head is
trained against ``bbox_overlaps_3d`` (true rotated 3D IoU), so that same metric
is used as the reference — not the old nearest-BEV proxy.

Raw classification scores are extracted with ``score_type='cls'`` so the CLS
baseline is a pure sigmoid probability, not the hybrid cls+iou blend.  This
gives a clean head-to-head comparison: does the IoU head give a better
localization-quality signal than the classifier alone?

All predictions / GT are in the nuScenes coordinate frame (X-right, Y-forward),
matching the frame the model was trained in.

Usage (from ~/mmdetection3d/):
    python tools/eval_iou_head_calibration.py \\
        --config configs/mean_teacher/mt_pretrain_iouhead_pointpillars_config.py \\
        --ckpt   work_dirs/<run>/epoch_N.pth \\
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from visualize_pseudo_labels import build_gt_lookup, gt_boxes_to_nus, load_points_nus

from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes, Box3DMode
from mmdet3d.structures.ops.iou3d_calculator import (
    bbox_overlaps_3d,
    bbox_overlaps_nearest_3d,
)

try:
    from scipy.stats import spearmanr as _spearmanr
    def spearman_r(a, b):
        r, _ = _spearmanr(a, b)
        return float(r)
except ImportError:
    def spearman_r(a, b):
        # fallback: rank correlation via numpy argsort
        n = len(a)
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, pts_nus: np.ndarray, device: str,
                  nms_thr: float = 0.01,
                  score_thr: float = 0.05,
                  nms_pre: int = 1000,
                  max_num: int = 200):
    """Run model on a single scene with calibration-appropriate inference settings.

    Several test_cfg overrides are applied for the duration of the call and
    restored afterwards:

    * ``score_type='cls'`` — keeps ``scores_3d`` as a pure classification
      probability so the CLS baseline comparison is uncontaminated by the
      IoU blend.  ``iou_scores_3d`` is the raw sigmoid of ``conv_iou`` and
      is unaffected.
    * ``nms_thr`` — default 0.01 (KITTI-tight).  The training config uses
      0.2, which is so loose that many near-identical FP boxes around the
      same background region all survive NMS and dominate the analysis.
      With 0.2 each cluster of overlapping background anchors keeps ~20
      survivors; with 0.01 only the single best-scoring one does.
    * ``score_thr``, ``nms_pre``, ``max_num`` — standard KITTI-eval values.

    Returns:
        boxes    (K, 7) float32  nuScenes / LiDAR frame, post-NMS
        cls      (K,)   float32  pure CLS probability (scores_3d)
        iou_pred (K,)   float32  predicted 3D-IoU quality (iou_scores_3d);
                                 None if the head is absent / disabled.
    """
    pts_tensor = torch.from_numpy(pts_nus).float()
    sample = Det3DDataSample()
    sample.set_metainfo({
        'box_type_3d': LiDARInstance3DBoxes,
        'box_mode_3d': Box3DMode.LIDAR,
    })

    head = model.bbox_head
    overrides = dict(score_type='cls', nms_thr=nms_thr,
                     score_thr=score_thr, nms_pre=nms_pre, max_num=max_num)
    saved = {k: head.test_cfg.get(k) for k in overrides}
    for k, v in overrides.items():
        head.test_cfg[k] = v
    try:
        with torch.no_grad():
            data = model.data_preprocessor(
                {'inputs': {'points': [pts_tensor]},
                 'data_samples': [sample]},
                training=False)
            pred_list = model.predict(data['inputs'], data['data_samples'])
    finally:
        for k, v in saved.items():
            if v is None:
                head.test_cfg.pop(k, None)
            else:
                head.test_cfg[k] = v

    inst = pred_list[0].pred_instances_3d
    boxes    = inst.bboxes_3d.tensor.cpu().numpy()[:, :7].astype(np.float32)
    cls      = inst.scores_3d.cpu().numpy().astype(np.float32)
    iou_raw  = getattr(inst, 'iou_scores_3d', None)
    iou_pred = iou_raw.cpu().numpy().astype(np.float32) if iou_raw is not None else None
    return boxes, cls, iou_pred


# ── IoU computation ────────────────────────────────────────────────────────────

def compute_actual_iou_3d(pred_boxes: np.ndarray,
                          gt_boxes: np.ndarray,
                          device: str = 'cuda') -> np.ndarray:
    """Max *rotated* 3D IoU per prediction vs all GT Car boxes.

    Matches the training objective of the conv_iou head exactly.

    Args:
        pred_boxes: (N, 7) nuScenes / LiDAR frame.
        gt_boxes:   (M, 7) nuScenes / LiDAR frame (Car GT only).

    Returns:
        (N,) float32  per-prediction max IoU (0 if M=0).
    """
    if len(pred_boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(gt_boxes) == 0:
        return np.zeros(len(pred_boxes), dtype=np.float32)

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    pred_t = torch.from_numpy(pred_boxes).float().to(dev)
    gt_t   = torch.from_numpy(gt_boxes).float().to(dev)
    # bbox_overlaps_3d returns [N, M]; we want max over GT.
    iou_mat = bbox_overlaps_3d(pred_t, gt_t, mode='iou', coordinate='lidar')
    return iou_mat.max(dim=1)[0].cpu().numpy().astype(np.float32)


def compute_actual_iou_bev(pred_boxes: np.ndarray,
                           gt_boxes: np.ndarray) -> np.ndarray:
    """Max nearest-BEV IoU per prediction (KITTI eval metric for reference).

    Kept as a secondary metric because KITTI official eval uses BEV IoU@0.7
    for Cars.  Not used for training-target calibration.
    """
    if len(pred_boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(gt_boxes) == 0:
        return np.zeros(len(pred_boxes), dtype=np.float32)
    pred_t = torch.from_numpy(pred_boxes).float()
    gt_t   = torch.from_numpy(gt_boxes).float()
    iou_mat = bbox_overlaps_nearest_3d(pred_t, gt_t, mode='iou',
                                       is_aligned=False, coordinate='lidar')
    return iou_mat.max(dim=1)[0].cpu().numpy().astype(np.float32)


# ── Plotting ───────────────────────────────────────────────────────────────────

_DARK  = '#0a0a0a'
_PANEL = '#111111'

def _style(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444444')
    ax.grid(True, alpha=0.15, color='white', linewidth=0.5)


def plot_scatter(pred_iou, actual_iou, pearson_r, spearman_r_val, out_path):
    """Hexbin scatter: predicted IoU vs actual rotated-3D IoU."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(_DARK); _style(ax)
    hb = ax.hexbin(actual_iou, pred_iou, gridsize=40, cmap='plasma',
                   extent=[0, 1, 0, 1], mincnt=1, bins='log')
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.ax.tick_params(colors='white', labelsize=7)
    cb.outline.set_edgecolor('#444444')
    cb.set_label('log count', color='white', fontsize=8)
    ax.plot([0, 1], [0, 1], 'w--', linewidth=1.0, alpha=0.6, label='perfect calibration')
    ax.set_xlabel('Actual rotated-3D IoU', color='white', fontsize=10)
    ax.set_ylabel('Predicted IoU (conv_iou head)', color='white', fontsize=10)
    ax.set_title(
        f'IoU Head Calibration  (Pearson r={pearson_r:.3f}  Spearman ρ={spearman_r_val:.3f})',
        color='white', fontsize=10)
    ax.legend(fontsize=8, facecolor='#222222', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_calibration_curve(pred_iou, actual_iou, out_path):
    """Binned calibration: mean actual IoU in each predicted-IoU bin."""
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

    bc = np.array(bin_centres)
    mp = np.array(mean_pred)
    ma = np.array(mean_actual)
    cn = np.array(counts, dtype=float)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    fig.patch.set_facecolor(_DARK)
    ax1.set_facecolor(_PANEL); ax2.set_facecolor(_PANEL)

    ax2.bar(bc, cn, width=0.03, color='#4488cc', alpha=0.35, label='box count')
    ax1.plot([0, 1], [0, 1], 'w--', linewidth=1.0, alpha=0.6, label='perfect calibration')
    ax1.plot(mp, ma, 'o-', color='tomato', linewidth=2, markersize=5,
             label='mean actual IoU per predicted bin')

    ax1.set_xlabel('Mean predicted IoU (bin centre)', color='white', fontsize=10)
    ax1.set_ylabel('Mean actual rotated-3D IoU', color='white', fontsize=10)
    ax2.set_ylabel('Box count', color='#4488cc', fontsize=9)
    ax1.set_title('Calibration curve: predicted IoU bins → actual IoU',
                  color='white', fontsize=11)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    for ax in (ax1, ax2):
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444444')
    ax1.grid(True, alpha=0.15, color='white', linewidth=0.5)

    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2,
               fontsize=8, facecolor='#222222', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_error_histogram(pred_iou, actual_iou, out_path):
    """Distribution of (predicted IoU − actual IoU) errors."""
    err = pred_iou - actual_iou
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(_DARK); _style(ax)
    bins = np.linspace(-1, 1, 81)
    ax.hist(err, bins=bins, color='#ff7f50', alpha=0.85, edgecolor='none')
    ax.axvline(0, color='white', linestyle='--', linewidth=1.2, alpha=0.7, label='zero error')
    ax.axvline(err.mean(), color='yellow', linestyle='--', linewidth=1.2,
               label=f'mean = {err.mean():+.3f}')
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


def plot_cls_vs_actual(cls_scores, actual_iou, pearson_r_cls, spearman_r_cls,
                       out_path):
    """CLS score vs actual IoU — baseline comparison for the classifier."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(_DARK); _style(ax)
    hb = ax.hexbin(cls_scores, actual_iou, gridsize=40, cmap='viridis',
                   extent=[0, 1, 0, 1], mincnt=1, bins='log')
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.ax.tick_params(colors='white', labelsize=7)
    cb.outline.set_edgecolor('#444444')
    cb.set_label('log count', color='white', fontsize=8)
    ax.set_xlabel('CLS score (pure sigmoid, no IoU blend)', color='white', fontsize=10)
    ax.set_ylabel('Actual rotated-3D IoU', color='white', fontsize=10)
    ax.set_title(
        f'CLS score vs actual IoU  (Pearson r={pearson_r_cls:.3f}  Spearman ρ={spearman_r_cls:.3f})',
        color='white', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_quality_gating(pred_iou, cls_scores, actual_iou, iou_tp_thr, out_path):
    """Precision@TP vs score threshold for IoU head and CLS (quality-gate analysis).

    For pseudo-label filtering the key question is: if we keep only boxes
    with score > τ, what fraction are true positives (actual IoU > iou_tp_thr)?
    Plotted as a function of τ together with the fraction of boxes retained.
    A good quality signal has high precision even at moderate retention rates.

    Args:
        iou_tp_thr: actual-IoU threshold that defines a true positive
                    (e.g. 0.5 for KITTI/nuScenes Car matching).
    """
    thresholds = np.linspace(0.0, 1.0, 201)
    tp_mask = actual_iou >= iou_tp_thr
    N = len(pred_iou)

    iou_prec, iou_ret = [], []
    cls_prec, cls_ret = [], []

    for tau in thresholds:
        for scores, prec_list, ret_list in [
                (pred_iou,  iou_prec, iou_ret),
                (cls_scores, cls_prec, cls_ret)]:
            keep = scores >= tau
            n_keep = keep.sum()
            if n_keep == 0:
                prec_list.append(np.nan)
                ret_list.append(0.0)
            else:
                prec_list.append(tp_mask[keep].mean())
                ret_list.append(n_keep / N)

    iou_prec = np.array(iou_prec)
    cls_prec = np.array(cls_prec)
    iou_ret  = np.array(iou_ret)
    cls_ret  = np.array(cls_ret)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(_DARK)
    for ax in (ax1, ax2):
        _style(ax)

    # Left: precision vs threshold
    ax1.plot(thresholds, iou_prec, color='tomato',   linewidth=2,
             label='IoU head (conv_iou)')
    ax1.plot(thresholds, cls_prec, color='#4488cc',  linewidth=2,
             label='CLS score (baseline)')
    ax1.axhline(tp_mask.mean(), color='white', linestyle=':', alpha=0.5,
                label=f'overall TP rate = {tp_mask.mean():.2f}')
    ax1.set_xlabel('Score threshold τ', color='white', fontsize=10)
    ax1.set_ylabel(f'Precision  (actual IoU ≥ {iou_tp_thr})', color='white', fontsize=10)
    ax1.set_title('Quality gating: precision vs threshold', color='white', fontsize=11)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9, facecolor='#222222', edgecolor='white', labelcolor='white')

    # Right: precision vs fraction retained (PR-style)
    ax2.plot(iou_ret, iou_prec, color='tomato',  linewidth=2, label='IoU head')
    ax2.plot(cls_ret, cls_prec, color='#4488cc', linewidth=2, label='CLS score')
    ax2.set_xlabel('Fraction of boxes retained', color='white', fontsize=10)
    ax2.set_ylabel(f'Precision  (actual IoU ≥ {iou_tp_thr})', color='white', fontsize=10)
    ax2.set_title('Precision vs retention (quality-gate trade-off)',
                  color='white', fontsize=11)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9, facecolor='#222222', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_score_distributions(pred_iou, cls_scores, actual_iou, iou_tp_thr, out_path):
    """Histogram of predicted-IoU and CLS scores, split by TP / FP status."""
    tp_mask = actual_iou >= iou_tp_thr
    fp_mask = ~tp_mask

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(_DARK)
    bins = np.linspace(0, 1, 51)

    titles = ['Predicted IoU distribution (conv_iou head)',
              'CLS score distribution (pure classifier)']
    xlab   = ['Predicted IoU', 'CLS score']
    scores_list = [pred_iou, cls_scores]

    for ax, scores, title, xl in zip(axes, scores_list, titles, xlab):
        _style(ax)
        ax.hist(scores[tp_mask], bins=bins, color='#2ecc71', alpha=0.7,
                label=f'TP (actual IoU≥{iou_tp_thr}, n={tp_mask.sum()})')
        ax.hist(scores[fp_mask], bins=bins, color='tomato', alpha=0.7,
                label=f'FP (actual IoU<{iou_tp_thr}, n={fp_mask.sum()})')
        ax.set_xlabel(xl, color='white', fontsize=10)
        ax.set_ylabel('Box count', color='white', fontsize=10)
        ax.set_title(title, color='white', fontsize=10)
        ax.legend(fontsize=8, facecolor='#222222', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f'Saved → {out_path}')


def plot_bev_vs_3d_comparison(pred_iou, actual_3d, actual_bev, out_path):
    """Scatter: predicted IoU vs actual 3D IoU and actual BEV IoU side by side.

    Shows how much the training target (3D IoU) differs from the KITTI eval
    metric (nearest-BEV IoU) and which the head is better calibrated to.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(_DARK)

    data = [(actual_3d, 'Actual rotated-3D IoU\n(training target)',
             f'r={np.corrcoef(pred_iou, actual_3d)[0,1]:.3f}'),
            (actual_bev, 'Actual nearest-BEV IoU\n(KITTI eval metric)',
             f'r={np.corrcoef(pred_iou, actual_bev)[0,1]:.3f}')]

    for ax, (ref_iou, xlabel, corr_str) in zip(axes, data):
        _style(ax)
        hb = ax.hexbin(ref_iou, pred_iou, gridsize=35, cmap='plasma',
                       extent=[0, 1, 0, 1], mincnt=1, bins='log')
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.ax.tick_params(colors='white', labelsize=7)
        cb.outline.set_edgecolor('#444444')
        ax.plot([0, 1], [0, 1], 'w--', linewidth=1, alpha=0.6)
        ax.set_xlabel(xlabel, color='white', fontsize=10)
        ax.set_ylabel('Predicted IoU (conv_iou)', color='white', fontsize=10)
        ax.set_title(f'Pearson {corr_str}', color='white', fontsize=10)

    plt.suptitle('Predicted IoU vs 3D (training target) and BEV (KITTI eval)',
                 color='white', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK)
    plt.close(fig)
    print(f'Saved → {out_path}')


# ── 3D joint scatter ──────────────────────────────────────────────────────────

def plot_3d_scatter_mpl(pred_iou: np.ndarray, cls_scores: np.ndarray,
                         actual_3d: np.ndarray,
                         out_base: str, max_pts: int = 15000) -> None:
    """Save 4 matplotlib 3D scatter PNGs at different viewing azimuths.

    Works on headless servers (Agg backend).  Saves files at
    ``out_base_view{0..3}.png`` for azimuths [45, 135, 215, 315]° so
    the caller gets a 360° picture of the joint score space without
    needing an interactive display.

    Axes:
        x = cls_scores  (raw CLS sigmoid, [0,1])
        y = pred_iou    (conv_iou sigmoid, [0,1])
        z = actual_3d   (rotated 3D IoU with GT, [0,1])

    Color encodes actual_3d (plasma colormap, red=0 / yellow=1).
    A white diagonal from (0,0,0)→(1,1,1) marks perfect calibration
    (where predicted IoU == actual IoU == CLS).
    """
    n = len(pred_iou)
    if n > max_pts:
        rng = np.random.RandomState(0)
        idx = rng.choice(n, max_pts, replace=False)
        pred_iou   = pred_iou[idx]
        cls_scores = cls_scores[idx]
        actual_3d  = actual_3d[idx]

    azimuths = [45, 135, 215, 315]
    elev = 25
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.plasma
    colors = cmap(norm(actual_3d))

    for view_idx, az in enumerate(azimuths):
        fig = plt.figure(figsize=(9, 8))
        fig.patch.set_facecolor('#0a0a0a')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#111111')

        sc = ax.scatter(cls_scores, pred_iou, actual_3d,
                        c=actual_3d, cmap='plasma', vmin=0, vmax=1,
                        s=1.5, alpha=0.4, linewidths=0, depthshade=False)

        # Perfect-calibration diagonal: (t, t, t) for t in [0,1]
        t = np.linspace(0, 1, 50)
        ax.plot(t, t, t, color='white', linewidth=1.2, alpha=0.7,
                label='pred = actual (perfect calib)')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xlabel('CLS score', color='white', fontsize=9, labelpad=8)
        ax.set_ylabel('Pred IoU (conv_iou)', color='white', fontsize=9, labelpad=8)
        ax.set_zlabel('Actual 3D IoU', color='white', fontsize=9, labelpad=8)
        ax.tick_params(colors='white', labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333333')
        ax.yaxis.pane.set_edgecolor('#333333')
        ax.zaxis.pane.set_edgecolor('#333333')
        ax.set_title(
            f'CLS × Pred-IoU × Actual-IoU  '
            f'(az={az}°  n={len(pred_iou):,})',
            color='white', fontsize=10, pad=10)

        # Colorbar only on first view to save space.
        if view_idx == 0:
            cb = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
            cb.set_label('Actual rotated-3D IoU', color='white', fontsize=8)
            cb.ax.tick_params(colors='white', labelsize=7)
            cb.outline.set_edgecolor('#444444')

        ax.legend(fontsize=7, facecolor='#222222',
                  edgecolor='white', labelcolor='white',
                  loc='upper left')
        ax.view_init(elev=elev, azim=az)

        out_path = f'{out_base}_view{view_idx}.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close(fig)
        print(f'Saved → {out_path}')


def plot_3d_scatter_plotly(pred_iou: np.ndarray, cls_scores: np.ndarray,
                            actual_3d: np.ndarray,
                            out_path: str, max_pts: int = 15000) -> None:
    """Write an interactive Plotly 3D scatter to an HTML file.

    The file is self-contained (plotly.js loaded from CDN) and can be
    opened in any browser after ``scp``.  Rotation, zoom, and hover data
    (showing all three scores) work out of the box.

    Falls back with a printed warning if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print('plotly not installed — skipping scatter_3d.html '
              '(pip install plotly to enable).')
        return

    n = len(pred_iou)
    if n > max_pts:
        rng = np.random.RandomState(0)
        idx = rng.choice(n, max_pts, replace=False)
        pred_iou   = pred_iou[idx]
        cls_scores = cls_scores[idx]
        actual_3d  = actual_3d[idx]

    # ── Main scatter ──────────────────────────────────────────────────────────
    scatter = go.Scatter3d(
        x=cls_scores.tolist(),
        y=pred_iou.tolist(),
        z=actual_3d.tolist(),
        mode='markers',
        marker=dict(
            size=2,
            color=actual_3d.tolist(),
            colorscale='RdYlGn',
            cmin=0.0, cmax=1.0,
            colorbar=dict(title='Actual 3D IoU', thickness=15, len=0.7),
            opacity=0.5,
        ),
        hovertemplate=(
            'CLS: %{x:.3f}<br>'
            'Pred IoU: %{y:.3f}<br>'
            'Actual IoU: %{z:.3f}<extra></extra>'
        ),
        name=f'predictions (n={len(pred_iou):,})',
    )

    # ── Perfect-calibration diagonal (t, t, t) ────────────────────────────────
    t = np.linspace(0, 1, 60).tolist()
    diag = go.Scatter3d(
        x=t, y=t, z=t,
        mode='lines',
        line=dict(color='white', width=3, dash='dot'),
        name='pred = actual (perfect calib)',
        hoverinfo='skip',
    )

    fig = go.Figure(data=[scatter, diag])
    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text='3D Score Space: CLS × Predicted IoU × Actual Rotated-3D IoU',
            font=dict(size=13),
        ),
        scene=dict(
            xaxis=dict(title='CLS Score', range=[0, 1]),
            yaxis=dict(title='Predicted IoU (conv_iou)', range=[0, 1]),
            zaxis=dict(title='Actual 3D IoU', range=[0, 1]),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.0, y=1.0, font=dict(size=10)),
    )
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'Saved → {out_path}  (open in browser after scp)')


# ── Stats summary ──────────────────────────────────────────────────────────────

def print_and_save_stats(pred_iou, actual_iou, actual_bev, cls_scores,
                         iou_tp_thr, out_path,
                         nms_thr=0.01, score_thr=0.05,
                         nms_pre=1000, max_num=200):
    n = len(pred_iou)
    err = pred_iou - actual_iou
    mae  = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    bias = float(err.mean())
    f01  = float((np.abs(err) < 0.1).mean())
    f02  = float((np.abs(err) < 0.2).mean())

    pr_iou = float(np.corrcoef(pred_iou, actual_iou)[0, 1]) if n > 1 else 0.0
    pr_cls = float(np.corrcoef(cls_scores, actual_iou)[0, 1]) if n > 1 else 0.0
    sp_iou = spearman_r(pred_iou, actual_iou) if n > 1 else 0.0
    sp_cls = spearman_r(cls_scores, actual_iou) if n > 1 else 0.0

    # 3D vs BEV Pearson for the IoU head
    pr_iou_bev = float(np.corrcoef(pred_iou, actual_bev)[0, 1]) if n > 1 else 0.0

    # Precision @ IoU > iou_tp_thr for the two scores at their median value
    tp_mask = actual_iou >= iou_tp_thr
    iou_med_prec = tp_mask[pred_iou >= np.median(pred_iou)].mean()
    cls_med_prec = tp_mask[cls_scores >= np.median(cls_scores)].mean()

    lines = [
        f'=== IoU Head Calibration Summary  ({n} predictions) ===',
        f'',
        f'  Inference settings applied:',
        f'    nms_thr={nms_thr}  score_thr={score_thr}  '
        f'nms_pre={nms_pre}  max_num={max_num}  score_type=cls',
        f'  Reference metric: rotated 3D IoU (training target)',
        f'',
        f'  ── Correlation (vs actual rotated-3D IoU) ──────────────',
        f'  Pearson  r  — IoU head:   {pr_iou:+.4f}',
        f'  Pearson  r  — CLS score:  {pr_cls:+.4f}',
        f'  Spearman ρ  — IoU head:   {sp_iou:+.4f}',
        f'  Spearman ρ  — CLS score:  {sp_cls:+.4f}',
        f'',
        f'  Pearson r  — IoU head vs nearest-BEV IoU (KITTI eval): {pr_iou_bev:+.4f}',
        f'',
        f'  ── Calibration error (IoU head vs rotated-3D IoU) ─────',
        f'  MAE:   {mae:.4f}',
        f'  RMSE:  {rmse:.4f}',
        f'  Bias (pred − actual):  {bias:+.4f}',
        f'  |error| < 0.10: {100*f01:.1f}%',
        f'  |error| < 0.20: {100*f02:.1f}%',
        f'  Mean predicted IoU: {pred_iou.mean():.4f}',
        f'  Mean actual 3D IoU: {actual_iou.mean():.4f}',
        f'  Mean actual BEV IoU: {actual_bev.mean():.4f}',
        f'',
        f'  ── Quality-gating (actual IoU ≥ {iou_tp_thr}) ──────────────',
        f'  Overall TP rate: {tp_mask.mean():.4f}',
        f'  Precision @ pred-IoU ≥ median:  {iou_med_prec:.4f}',
        f'  Precision @ CLS     ≥ median:  {cls_med_prec:.4f}',
        f'',
        f'--- Per-bin breakdown (bins of predicted IoU, vs actual 3D IoU) ---',
        f'  {"bin":>12s}  {"count":>7s}  {"mean_pred":>9s}  {"mean_actual_3d":>14s}  '
        f'{"mean_actual_bev":>15s}  {"MAE":>6s}',
    ]
    bins = np.arange(0.0, 1.05, 0.1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred_iou >= lo) & (pred_iou < hi)
        nc = mask.sum()
        if nc == 0:
            continue
        mp  = pred_iou[mask].mean()
        ma  = actual_iou[mask].mean()
        mbev = actual_bev[mask].mean()
        me  = float(np.abs(pred_iou[mask] - actual_iou[mask]).mean())
        lines.append(
            f'  [{lo:.1f} – {hi:.1f}):  {nc:7d}  {mp:9.4f}  {ma:14.4f}  {mbev:15.4f}  {me:6.4f}')

    text = '\n'.join(lines)
    print(text)
    with open(out_path, 'w') as f:
        f.write(text + '\n')
    print(f'\nStats saved → {out_path}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Calibration analysis for the per-anchor IoU head (conv_iou) on KITTI val',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Model config (e.g. mt_pretrain_iouhead_pointpillars_config.py)')
    p.add_argument('--ckpt',   required=True,
                   help='Model checkpoint (.pth)')
    p.add_argument('--kitti-info', default='data/kitti/kitti_infos_val.pkl')
    p.add_argument('--kitti-root', default='data/kitti/')
    p.add_argument('--out-dir',    default='work_dirs/iou_head_calibration')
    p.add_argument('--num-scenes', type=int, default=0,
                   help='Scenes to evaluate (0 = all)')
    p.add_argument('--iou-tp-thr', type=float, default=0.5,
                   help='Actual-IoU threshold that counts as a true positive '
                        '(0.5 for nuScenes/KITTI Car matching)')
    # Inference overrides — applied on top of whatever the model config sets.
    # Defaults replicate the KITTI test config (test_kitti_nuspretrained_pointpillars.py).
    # The training config uses nms_thr=0.2, which keeps many near-duplicate FP
    # boxes per background cluster, dominating the analysis with an "ocean" of
    # sigmoid≈0.5 predictions. Use nms_thr=0.01 for a clean calibration view.
    p.add_argument('--nms-thr',   type=float, default=0.01,
                   help='NMS IoU threshold override (default 0.01 = KITTI-tight)')
    p.add_argument('--score-thr', type=float, default=0.05,
                   help='Score threshold override for pre-NMS filtering')
    p.add_argument('--nms-pre',   type=int,   default=1000,
                   help='Max candidates kept before NMS')
    p.add_argument('--max-num',   type=int,   default=200,
                   help='Max final boxes kept after NMS')
    p.add_argument('--plot3d-max-pts', type=int, default=15000,
                   help='Max points subsampled for 3D scatter plots '
                        '(matplotlib PNGs + Plotly HTML). '
                        'Increase for denser coverage, decrease for '
                        'faster browser/render performance.')
    p.add_argument('--seed',   type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────────
    print(f'Loading model: {args.ckpt}')
    model = init_model(args.config, args.ckpt, device=args.device)
    model.eval()

    head = model.bbox_head
    has_iou_head = (hasattr(head, 'predict_iou') and head.predict_iou
                    and hasattr(head, 'conv_iou'))
    print(f'Model loaded.  per-anchor IoU head (conv_iou) present: {has_iou_head}')
    print(f'Inference overrides: nms_thr={args.nms_thr}  score_thr={args.score_thr}  '
          f'nms_pre={args.nms_pre}  max_num={args.max_num}  score_type=cls')
    print(f'  (model test_cfg had: nms_thr={head.test_cfg.get("nms_thr")}  '
          f'nms_pre={head.test_cfg.get("nms_pre")}  '
          f'max_num={head.test_cfg.get("max_num")}  '
          f'score_type={head.test_cfg.get("score_type", "cls")})')
    if not has_iou_head:
        print('WARNING: predict_iou is False or conv_iou is missing — '
              'iou_scores_3d will be absent.  Run with a predict_iou=True config.')

    # ── Load GT ─────────────────────────────────────────────────────────────────
    print(f'Loading KITTI GT from {args.kitti_info} ...')
    gt_lookup, gt_car_label = build_gt_lookup(args.kitti_info)
    velodyne_dir = os.path.join(args.kitti_root, 'training', 'velodyne_reduced')

    all_scenes = sorted(gt_lookup.keys())
    if args.num_scenes > 0:
        all_scenes = random.Random(args.seed).sample(
            all_scenes, min(args.num_scenes, len(all_scenes)))
    print(f'Evaluating {len(all_scenes)} scenes ...')

    # ── Per-scene loop ───────────────────────────────────────────────────────────
    all_cls, all_pred_iou, all_actual_3d, all_actual_bev = [], [], [], []
    n_no_iou_head = 0

    for scene_id in all_scenes:
        bin_path = os.path.join(velodyne_dir, scene_id + '.bin')
        if not os.path.isfile(bin_path):
            print(f'  WARN: {bin_path} not found, skipping.')
            continue

        pts_nus = load_points_nus(bin_path)
        boxes, cls, iou_pred = run_inference(
            model, pts_nus, args.device,
            nms_thr=args.nms_thr, score_thr=args.score_thr,
            nms_pre=args.nms_pre, max_num=args.max_num)

        if iou_pred is None:
            n_no_iou_head += 1
            continue

        info = gt_lookup[scene_id]
        gt_nus     = gt_boxes_to_nus(info['cam_boxes'], info['lidar2cam'])
        gt_car_mask = info['labels'] == gt_car_label
        gt_car_boxes = (gt_nus[gt_car_mask]
                        if gt_car_mask.any()
                        else np.zeros((0, 7), dtype=np.float32))

        actual_3d  = compute_actual_iou_3d(boxes, gt_car_boxes, device=args.device)
        actual_bev = compute_actual_iou_bev(boxes, gt_car_boxes)

        all_cls.extend(cls.tolist())
        all_pred_iou.extend(iou_pred.tolist())
        all_actual_3d.extend(actual_3d.tolist())
        all_actual_bev.extend(actual_bev.tolist())

    if n_no_iou_head > 0:
        print(f'WARNING: {n_no_iou_head} scenes had no iou_scores_3d — '
              f'make sure predict_iou=True in the config.')

    if not all_pred_iou:
        print('ERROR: no predictions collected. Check model path and data.')
        return

    pred_iou   = np.array(all_pred_iou,   dtype=np.float32)
    actual_3d  = np.array(all_actual_3d,  dtype=np.float32)
    actual_bev = np.array(all_actual_bev, dtype=np.float32)
    cls_scores = np.array(all_cls,        dtype=np.float32)

    pr_iou = float(np.corrcoef(pred_iou, actual_3d)[0, 1]) if len(pred_iou) > 1 else 0.0
    sp_iou = spearman_r(pred_iou, actual_3d) if len(pred_iou) > 1 else 0.0
    pr_cls = float(np.corrcoef(cls_scores, actual_3d)[0, 1]) if len(cls_scores) > 1 else 0.0
    sp_cls = spearman_r(cls_scores, actual_3d) if len(cls_scores) > 1 else 0.0

    # ── Plots ────────────────────────────────────────────────────────────────────
    plot_scatter(
        pred_iou, actual_3d, pr_iou, sp_iou,
        os.path.join(args.out_dir, 'scatter_iou_calibration.png'))
    plot_calibration_curve(
        pred_iou, actual_3d,
        os.path.join(args.out_dir, 'calibration_curve.png'))
    plot_error_histogram(
        pred_iou, actual_3d,
        os.path.join(args.out_dir, 'error_histogram.png'))
    plot_cls_vs_actual(
        cls_scores, actual_3d, pr_cls, sp_cls,
        os.path.join(args.out_dir, 'cls_vs_actual_iou.png'))
    plot_quality_gating(
        pred_iou, cls_scores, actual_3d, args.iou_tp_thr,
        os.path.join(args.out_dir, 'quality_gating.png'))
    plot_score_distributions(
        pred_iou, cls_scores, actual_3d, args.iou_tp_thr,
        os.path.join(args.out_dir, 'score_distributions.png'))
    plot_bev_vs_3d_comparison(
        pred_iou, actual_3d, actual_bev,
        os.path.join(args.out_dir, 'bev_vs_3d_comparison.png'))

    # ── 3D joint scatter: CLS × Pred-IoU × Actual-IoU ────────────────────────
    # matplotlib multi-view PNGs work on headless servers (Agg).
    # Plotly HTML is interactive — open in a browser after scp.
    plot_3d_scatter_mpl(
        pred_iou, cls_scores, actual_3d,
        os.path.join(args.out_dir, 'scatter_3d'),
        max_pts=args.plot3d_max_pts)
    plot_3d_scatter_plotly(
        pred_iou, cls_scores, actual_3d,
        os.path.join(args.out_dir, 'scatter_3d.html'),
        max_pts=args.plot3d_max_pts)

    # ── Stats ────────────────────────────────────────────────────────────────────
    print_and_save_stats(
        pred_iou, actual_3d, actual_bev, cls_scores, args.iou_tp_thr,
        os.path.join(args.out_dir, 'calibration_stats.txt'),
        nms_thr=args.nms_thr, score_thr=args.score_thr,
        nms_pre=args.nms_pre, max_num=args.max_num)


if __name__ == '__main__':
    main()
