#!/usr/bin/env python3
"""Effect of source ROS on predicted object SIZE (CenterPoint, KITTI val).

Controlled comparison (same architecture, differs only in ROS during pretraining):
  Source-Only        : baseline_centerpoint_18may
  Source-Only + ROS  : baseline_centerpoint_ros_10jun
vs KITTI GT car sizes. For each model, predicted Car boxes are matched to GT
(BEV-IoU >= 0.5) and their l/w/h collected; GT is all KITTI val Car boxes.

Models are built from the provided KITTI *test* configs (proper test_cfg), so
inference matches the reported baseline evaluation. Arrays cached to npz.

Usage: python tools/plot_size_distributions.py [--max-frames N] [--force]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
})
import matplotlib.pyplot as plt
import torch
from scipy.stats import gaussian_kde
from mmengine.config import Config

sys.path.insert(0, os.path.dirname(__file__))
from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes, Box3DMode
from visualize_pseudo_labels import build_gt_lookup, gt_boxes_to_nus, load_points_nus
from analyze_ps_geometry_bias import matched_pairs

WD = 'work_dirs'
CONDS = [
    ('Source-Only', '#d62728',
     f'{WD}/baseline_centerpoint_18may/test_kitti_offset0.29/test_kitti_nuspretrained_centerpoint.py',
     f'{WD}/baseline_centerpoint_18may/epoch_20.pth'),
    ('Source-Only + ROS', '#1f77b4',
     f'{WD}/baseline_centerpoint_ros_10jun/test_kitti_shift0.29/test_kitti_nuspretrained_centerpoint.py',
     f'{WD}/baseline_centerpoint_ros_10jun/epoch_20.pth'),
    ('Adapted', '#9467bd',
     f'{WD}/mt_cp_groundSnap_19jun/mean_teacher_centerpoint_config.py',
     f'{WD}/mt_cp_groundSnap_19jun/epoch_4.pth'),
]
GT_COLOR = '#2ca02c'
PRED_CAR = 0
KITTI_VAL = 'data/kitti/kitti_infos_val.pkl'
VELO = 'data/kitti/training/velodyne_reduced'
# box index -> (name, x-range) ; nus format [x,y,z,l,w,h,yaw]
DIMS = [(3, 'length $\\ell$ [m]', (2.5, 6.0)),
        (4, 'width $w$ [m]', (1.0, 2.6)),
        (5, 'height $h$ [m]', (1.0, 2.4))]


def predict_boxes(model, pts_nus):
    """Run the model (its own test_cfg) -> (boxes (K,7) nus, labels (K,))."""
    sample = Det3DDataSample()
    sample.set_metainfo({'box_type_3d': LiDARInstance3DBoxes,
                         'box_mode_3d': Box3DMode.LIDAR})
    with torch.no_grad():
        data = model.data_preprocessor(
            {'inputs': {'points': [torch.from_numpy(pts_nus).float()]},
             'data_samples': [sample]}, training=False)
        pred = model.predict(data['inputs'], data['data_samples'])[0]
    inst = pred.pred_instances_3d
    return inst.bboxes_3d.tensor.cpu().numpy()[:, :7], inst.labels_3d.cpu().numpy()


def collect(label, test_cfg_path, ckpt, gt_lookup, gt_car, device, max_frames):
    cfg = Config.fromfile(test_cfg_path)
    cfg['class_names'] = ['Car']            # init_model reads it for dataset_meta
    model = init_model(cfg, ckpt, device=device)
    lwh = []
    keys = list(gt_lookup.keys())
    if max_frames:
        keys = keys[:max_frames]
    for n, scene in enumerate(keys):
        gi = gt_lookup[scene]
        if len(gi['cam_boxes']) == 0:
            continue
        G = gt_boxes_to_nus(gi['cam_boxes'], gi['lidar2cam'])
        G = G[gi['labels'] == gt_car].astype(np.float32)
        bin_path = os.path.join(VELO, scene + '.bin')
        if len(G) == 0 or not os.path.exists(bin_path):
            continue
        pts = load_points_nus(bin_path)
        boxes, labels = predict_boxes(model, pts)
        P = boxes[labels == PRED_CAR].astype(np.float32)
        if len(P) == 0:
            continue
        for i, j in matched_pairs(P, G, 0.5):
            lwh.append(P[i, 3:6])
        if (n + 1) % 500 == 0:
            print(f'  [{label}] {n+1}/{len(keys)} frames, {len(lwh)} matched',
                  flush=True)
    return np.array(lwh, np.float32)


def gt_sizes(gt_lookup, gt_car):
    out = []
    for gi in gt_lookup.values():
        if len(gi['cam_boxes']) == 0:
            continue
        G = gt_boxes_to_nus(gi['cam_boxes'], gi['lidar2cam'])
        G = G[gi['labels'] == gt_car]
        if len(G):
            out.append(G[:, 3:6])
    return np.concatenate(out).astype(np.float32)


def kde_line(ax, data, lo, hi, color, label, lw=2.2, ls='-'):
    xs = np.linspace(lo, hi, 400)
    ax.plot(xs, gaussian_kde(data)(xs), color=color, lw=lw, ls=ls, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='work_dirs/thesis_size_figs')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt_lookup, gt_car = build_gt_lookup(KITTI_VAL)
    data = {}
    for label, _c, tcfg, ckpt in CONDS:
        npz = os.path.join(args.out, label.replace(' ', '_').replace('+', '') + '.npz')
        if os.path.exists(npz) and not args.force:
            data[label] = np.load(npz)['lwh']; print(f'[{label}] cache {len(data[label])}')
            continue
        print(f'[{label}] inferring ...', flush=True)
        lwh = collect(label, tcfg, ckpt, gt_lookup, gt_car, args.device, args.max_frames)
        np.savez(npz, lwh=lwh); data[label] = lwh
        print(f'[{label}] {len(lwh)} matched  median l/w/h='
              f'{np.median(lwh,axis=0).round(3)}')
    gpath = os.path.join(args.out, 'GT.npz')
    if os.path.exists(gpath) and not args.force:
        GT = np.load(gpath)['lwh']
    else:
        GT = gt_sizes(gt_lookup, gt_car); np.savez(gpath, lwh=GT)
    print(f'[GT] {len(GT)} cars  median l/w/h={np.median(GT,axis=0).round(3)}')

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for ax, (idx, xlabel, (lo, hi)) in zip(axes, DIMS):
        meds = []
        for label, color, _, _ in CONDS:
            d = data[label][:, idx - 3]
            kde_line(ax, d, lo, hi, color, label)
            m = float(np.median(d)); meds.append((color, m))
            ax.axvline(m, color=color, ls=':', lw=1.2, alpha=0.85)
        gd = GT[:, idx - 3]
        kde_line(ax, gd, lo, hi, GT_COLOR, 'KITTI GT', ls='--')
        gm = float(np.median(gd)); meds.append((GT_COLOR, gm))
        ax.axvline(gm, color=GT_COLOR, ls=':', lw=1.2, alpha=0.85)
        # color-coded median values, upper-right corner (curves' right tail is empty)
        ax.text(0.97, 0.97, 'median [m]', color='0.4', transform=ax.transAxes,
                ha='right', va='top', fontsize=8)
        for k, (color, m) in enumerate(meds):
            ax.text(0.97, 0.88 - 0.085 * k, f'{m:.2f}', color=color,
                    transform=ax.transAxes, ha='right', va='top', fontsize=9)
        ax.set_xlim(lo, hi); ax.set_yticks([])
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel('probability density')
    axes[0].legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.suptitle('Predicted vs. GT object size (KITTI val)', y=0.98)
    out = os.path.join(args.out, 'fig_size_dist.png')
    fig.savefig(out, dpi=200, transparent=True, bbox_inches='tight')
    print('wrote', out)

    # medians table for caption
    with open(os.path.join(args.out, 'size_medians.txt'), 'w') as f:
        f.write(f'{"cond":20s} {"l":>6s} {"w":>6s} {"h":>6s}\n')
        for label, _, _, _ in CONDS:
            m = np.median(data[label], axis=0)
            f.write(f'{label:20s} {m[0]:6.3f} {m[1]:6.3f} {m[2]:6.3f}\n')
        m = np.median(GT, axis=0)
        f.write(f'{"KITTI GT":20s} {m[0]:6.3f} {m[1]:6.3f} {m[2]:6.3f}\n')
    print(open(os.path.join(args.out, 'size_medians.txt')).read())


if __name__ == '__main__':
    main()
