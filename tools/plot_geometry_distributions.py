#!/usr/bin/env python3
"""Thesis §6.5 geometry-distribution figures (CenterPoint, KITTI val).

Runs three checkpoints over KITTI val, matches predicted Car boxes to GT
(BEV-IoU >= 0.5), and for each matched pair collects the true 3D-IoU and the
signed bottom-z error (pred_z - gt_z). Renders two overlaid-KDE figures:

  Fig 1  fig_iou3d_dist.png : 3D-IoU distribution, vlines at IoU 0.5 / 0.7.
  Fig 2  fig_dz_dist.png    : signed bottom-z error, vline at 0 (= GT).

Per-condition arrays are cached to <out>/<label>.npz so plots can be re-tuned
without re-running inference (delete the npz or pass --force to recompute).

Usage:
  python tools/plot_geometry_distributions.py [--max-frames N] [--force]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
# LaTeX-matching serif. Times New Roman / Computer Modern are not installed here;
# STIXGeneral is the Times-metric-compatible stand-in (mathtext 'stix' to match).
# To get true Computer Modern in the thesis, install CMU/Times fonts (or set
# text.usetex=True) and re-run the plotting phase off the cached .npz.
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mmengine.config import Config

sys.path.insert(0, os.path.dirname(__file__))
from mmdet3d.apis import init_model
from visualize_pseudo_labels import (build_gt_lookup, gt_boxes_to_nus,
                                      load_points_nus, run_baseline_raw)
from analyze_ps_geometry_bias import matched_pairs
from ceiling_oracle_decomp import three_d_iou
from mmdet3d.structures.ops import ground_snap_boxes

WD = 'work_dirs'
# (label, config, checkpoint, apply_ground_snap)  — order = legend / z-order
CONDS = [
    ('Source-Only',
     f'{WD}/baseline_centerpoint_18may/mt_pretrain_centerpoint_config.py',
     f'{WD}/baseline_centerpoint_18may/epoch_20.pth', False),
    ('Adapted w/o GS',
     f'{WD}/mt_cp_no_contrastive_ema0.99995/mean_teacher_centerpoint_config.py',
     f'{WD}/mt_cp_no_contrastive_ema0.99995/epoch_4.pth', False),
    ('Adapted',
     f'{WD}/mt_cp_groundSnap_19jun/mean_teacher_centerpoint_config.py',
     f'{WD}/mt_cp_groundSnap_19jun/epoch_4.pth', True),
]
GS_KW = dict(pctl=2.0, min_pts=25, margin=0.3, max_disp=0.0)
PRED_CAR = 0          # single-class CenterPoint -> Car is label 0
KITTI_VAL = 'data/kitti/kitti_infos_val.pkl'
VELO = 'data/kitti/training/velodyne_reduced'
COLORS = {'Source-Only': '#d62728', 'Adapted w/o GS': '#ff7f0e', 'Adapted': '#1f77b4'}


def collect(label, config, ckpt, apply_gs, gt_lookup, gt_car, device, max_frames):
    """Run one checkpoint over val -> (iou3d, dz) arrays over matched pairs."""
    # Some older configs lack a top-level `class_names` (init_model reads it for
    # dataset_meta). These are all single-class Car models, so inject it.
    cfg = Config.fromfile(config)
    cfg['class_names'] = ['Car']
    model = init_model(cfg, ckpt, device=device)
    P_all, G_all = [], []
    keys = list(gt_lookup.keys())
    if max_frames:
        keys = keys[:max_frames]
    for n, scene in enumerate(keys):
        gi = gt_lookup[scene]
        if len(gi['cam_boxes']) == 0:
            continue
        G = gt_boxes_to_nus(gi['cam_boxes'], gi['lidar2cam'])
        G = G[gi['labels'] == gt_car].astype(np.float32)
        if len(G) == 0:
            continue
        bin_path = os.path.join(VELO, scene + '.bin')
        if not os.path.exists(bin_path):
            continue
        pts = load_points_nus(bin_path)
        boxes, _cls, _iou, labels = run_baseline_raw(model, pts, device)
        P = boxes[labels == PRED_CAR].astype(np.float32)
        if len(P) == 0:
            continue
        if apply_gs:
            P, _ = ground_snap_boxes(P, pts, **GS_KW)
        for i, j in matched_pairs(P, G, 0.5):
            P_all.append(P[i]); G_all.append(G[j])
        if (n + 1) % 500 == 0:
            print(f'  [{label}] {n+1}/{len(keys)} frames, {len(P_all)} pairs',
                  flush=True)
    P_all, G_all = np.array(P_all, np.float32), np.array(G_all, np.float32)
    iou3d = three_d_iou(P_all, G_all) if len(P_all) else np.zeros(0, np.float32)
    dz = (P_all[:, 2] - G_all[:, 2]) if len(P_all) else np.zeros(0, np.float32)
    return iou3d.astype(np.float32), dz.astype(np.float32)


def kde_line(ax, data, lo, hi, color, label, lw=2.2):
    xs = np.linspace(lo, hi, 400)
    k = gaussian_kde(data)
    ax.plot(xs, k(xs), color=color, lw=lw, label=label)


def fig_iou(data, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, _, _, _ in CONDS:
        iou = data[label]['iou3d']
        f7 = 100 * (iou >= 0.7).mean()
        # stats on a second line so the legend stays narrow (no overlap with curves)
        kde_line(ax, np.clip(iou, 0, 1), 0, 1, COLORS[label],
                 f'{label}\n(frac$\\geq$0.7 = {f7:.0f}%)')
    ymax = ax.get_ylim()[1]
    for thr, ls in ((0.5, ':'), (0.7, '--')):
        ax.axvline(thr, color='0.4', ls=ls, lw=1.3)
        ax.text(thr + 0.006, ymax * 0.30, f'IoU={thr}', color='0.35',
                fontsize=8, va='center', rotation=90)
    ax.set_xlim(0, 1); ax.set_ylim(0, ymax * 1.02)
    ax.set_yticks([])   # KDE height is not directly interpretable; only shape/area matters
    ax.set_xlabel('3D IoU of matched detections')
    ax.set_ylabel('probability density'); ax.legend(loc='upper left', frameon=False)
    ax.set_title('3D-IoU distribution (KITTI val)')
    fig.tight_layout(); fig.savefig(out, dpi=200, transparent=True)
    print('wrote', out)


def fig_dz(data, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, _, _, _ in CONDS:
        dz = data[label]['dz']
        med, q1, q3 = np.median(dz), np.percentile(dz, 25), np.percentile(dz, 75)
        # KDE on raw dz (not clipped) so there is no edge pile-up; window to [-0.5,0.5].
        # stats on a second line so the legend stays narrow (clears the central peak).
        kde_line(ax, dz, -0.5, 0.5, COLORS[label],
                 f'{label}\n(med={med:+.3f}, IQR={q3-q1:.3f})')
    ymax = ax.get_ylim()[1]
    ax.axvline(0.0, color='0.2', ls='-', lw=1.5)
    ax.text(0.014, ymax * 0.30, 'GT', color='0.2', fontsize=8, va='center', rotation=90)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, ymax * 1.02)
    ax.set_yticks([])   # KDE height is not directly interpretable; only shape/area matters
    ax.set_xlabel('signed bottom-z error  (pred $-$ GT)  [m]')
    ax.set_ylabel('probability density'); ax.legend(loc='upper left', frameon=False)
    ax.set_title('Box bottom-z error vs GT (KITTI val)')
    fig.tight_layout(); fig.savefig(out, dpi=200, transparent=True)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='work_dirs/thesis_geom_figs')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all val frames')
    ap.add_argument('--force', action='store_true', help='recompute cached npz')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt_lookup, gt_car = build_gt_lookup(KITTI_VAL)
    data = {}
    for label, config, ckpt, gs in CONDS:
        npz = os.path.join(args.out, label.replace(' ', '_').replace('/', '') + '.npz')
        if os.path.exists(npz) and not args.force:
            d = np.load(npz); data[label] = dict(iou3d=d['iou3d'], dz=d['dz'])
            print(f'[{label}] loaded cache ({len(d["iou3d"])} pairs)')
            continue
        print(f'[{label}] inferring ...', flush=True)
        iou3d, dz = collect(label, config, ckpt, gs, gt_lookup, gt_car,
                            args.device, args.max_frames)
        np.savez(npz, iou3d=iou3d, dz=dz)
        data[label] = dict(iou3d=iou3d, dz=dz)
        print(f'[{label}] {len(iou3d)} pairs  frac>=0.7={100*(iou3d>=0.7).mean():.1f}%'
              f'  dz med={np.median(dz):+.3f}')

    # stats file
    with open(os.path.join(args.out, 'stats.txt'), 'w') as f:
        f.write(f'{"cond":16s} {"n":>6s} {"frac>=0.5":>9s} {"frac>=0.7":>9s} '
                f'{"dz_med":>7s} {"dz_IQR":>7s}\n')
        for label, _, _, _ in CONDS:
            iou, dz = data[label]['iou3d'], data[label]['dz']
            f.write(f'{label:16s} {len(iou):6d} {100*(iou>=0.5).mean():8.1f}% '
                    f'{100*(iou>=0.7).mean():8.1f}% {np.median(dz):+7.3f} '
                    f'{np.percentile(dz,75)-np.percentile(dz,25):7.3f}\n')
    print('\n' + open(os.path.join(args.out, 'stats.txt')).read())

    fig_iou(data, os.path.join(args.out, 'fig_iou3d_dist.png'))
    fig_dz(data, os.path.join(args.out, 'fig_dz_dist.png'))


if __name__ == '__main__':
    main()
