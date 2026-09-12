#!/usr/bin/env python3
"""Render 30 candidate qualitative samples (BEV + side) for the thesis figure,
from the final adapted detector. GT green, predictions (score>=THR, GS-applied)
orange. One PNG per scene in work_dirs/thesis_qual/samples/, plus a contact
sheet to browse. Frame selection is just a spread (some scenes with misses,
mostly clean) — the user decides which to keep.
"""
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                            'mathtext.fontset': 'stix'})
import matplotlib.pyplot as plt
from mmengine.config import Config

sys.path.insert(0, os.path.dirname(__file__))
from mmdet3d.apis import init_model
from mmdet3d.structures.ops import ground_snap_boxes
from mmdet3d.structures.ops.box_np_ops import points_in_rbbox
from visualize_pseudo_labels import (build_gt_lookup, gt_boxes_to_nus,
                                      load_points_nus, run_baseline_raw)

CFG = 'work_dirs/mt_cp_groundSnap_19jun/mean_teacher_centerpoint_config.py'
CKPT = 'work_dirs/mt_cp_groundSnap_19jun/epoch_4.pth'
GS = dict(pctl=2.0, min_pts=25, margin=0.3, max_disp=0.0)
VAL = 'data/kitti/kitti_infos_val.pkl'
VELO = 'data/kitti/training/velodyne_reduced'
CACHE = 'work_dirs/thesis_qual/scan_cache.pkl'
OUT = 'work_dirs/thesis_qual/samples'
SCORE_THR = 0.40
KEEP_FRAC = 0.70   # top fraction of preds per scene (by score)
MIN_PTS   = 10     # drop boxes with fewer interior LiDAR points
YMIN, YMAX = 2.0, 55.0
GT_C, PR_C = '#2ca02c', '#ff7f0e'   # GT green, pred orange


def in_region(b):
    y, x = b[:, 1], b[:, 0]
    return (y > YMIN) & (y < YMAX) & (np.abs(x) < 0.9 * y + 3.0)


def bev_corners(box):
    cx, cy, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    return loc @ np.array([[c, -s], [s, c]]).T + [cx, cy]


def draw_bev(ax, pts, G, P):
    p = pts[in_region(pts)] if len(pts) else pts
    ax.scatter(p[:, 0], p[:, 1], s=0.2, c='0.75', linewidths=0)
    for b, col in [(G, GT_C), (P, PR_C)]:
        for box in b:
            c = np.vstack([bev_corners(box), bev_corners(box)[0]])
            ax.plot(c[:, 0], c[:, 1], color=col, lw=1.5)
    ax.set_xlim(-30, 30); ax.set_ylim(YMIN, YMAX); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title('BEV', fontsize=9)


def draw_side(ax, pts, G, P):
    m = in_region(pts) if len(pts) else np.zeros(0, bool)
    p = pts[m]
    ax.scatter(p[:, 1], p[:, 2], s=0.2, c='0.75', linewidths=0)
    from matplotlib.patches import Rectangle
    for b, col in [(G, GT_C), (P, PR_C)]:
        for box in b:   # side: forward (y) x height (z); depth extent ~ l
            ax.add_patch(Rectangle((box[1] - box[3] / 2, box[2]), box[3], box[5],
                                   fill=False, edgecolor=col, lw=1.3))
    ax.set_xlim(YMIN, YMAX); ax.set_ylim(-2.4, 0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title('side (forward × height)', fontsize=9)


def select_ids(cache, n=30):
    cands = [(k, v) for k, v in cache.items() if 3 <= v['n_gt'] <= 8]
    miss = [k for k, v in cands if v['fn'] >= 1]
    clean = [k for k, v in cands if v['fn'] == 0]
    rng = np.random.default_rng(0)
    miss_s = list(rng.choice(miss, min(10, len(miss)), replace=False))
    clean_s = list(rng.choice(clean, n - len(miss_s), replace=False))
    return sorted(set(miss_s + clean_s))


def main():
    os.makedirs(OUT, exist_ok=True)
    cache = pickle.load(open(CACHE, 'rb'))
    ids = select_ids(cache, 30)
    print(f'rendering {len(ids)} samples ...')
    cfg = Config.fromfile(CFG); cfg['class_names'] = ['Car']
    model = init_model(cfg, CKPT, device='cuda:0')
    gt_lookup, gt_car = build_gt_lookup(VAL)

    thumbs = []
    for k in ids:
        gi = gt_lookup[k]
        pts = load_points_nus(os.path.join(VELO, k + '.bin'))
        G = (gt_boxes_to_nus(gi['cam_boxes'], gi['lidar2cam'])[gi['labels'] == gt_car]
             if len(gi['cam_boxes']) else np.zeros((0, 7), np.float32)).astype(np.float32)
        G = G[in_region(G)] if len(G) else G
        boxes, cls, _iou, labels = run_baseline_raw(model, pts, 'cuda:0')
        car_mask = (labels == 0) & (cls >= SCORE_THR)
        P = boxes[car_mask].astype(np.float32)
        sc = cls[car_mask]
        if len(P):
            # top keep_frac by score
            n_keep = int(np.ceil(KEEP_FRAC * len(P)))
            order = np.argsort(sc)[::-1]
            P = P[order[:n_keep]]; sc = sc[order[:n_keep]]
            # min-pts guard
            pt_cnts = points_in_rbbox(pts[:, :3], P, z_axis=2, origin=(0.5, 0.5, 0)).sum(0)
            P = P[pt_cnts >= MIN_PTS]
            P, _ = ground_snap_boxes(P, pts, **GS)
            P = P[in_region(P)]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.2, 5.4),
                                     gridspec_kw=dict(height_ratios=[3, 1.3]))
        draw_bev(a1, pts, G, P); draw_side(a2, pts, G, P)
        fig.suptitle(f'{k}   (GT={len(G)}, pred={len(P)})  GT green / pred orange',
                     fontsize=9, y=0.99)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'sample_{k}.png'), dpi=140,
                    transparent=False, bbox_inches='tight')
        plt.close(fig)
        thumbs.append((k, pts, G, P))
        print(f'  {k}: GT={len(G)} pred={len(P)}', flush=True)

    # contact sheet (BEV thumbnails) to browse all at once
    nc = 6; nr = int(np.ceil(len(thumbs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 2.3, nr * 2.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, (k, pts, G, P) in zip(axes, thumbs):
        ax.axis('on'); draw_bev(ax, pts, G, P); ax.set_title(k, fontsize=8)
    fig.tight_layout()
    fig.savefig('work_dirs/thesis_qual/contact_sheet.png', dpi=140,
                transparent=False, bbox_inches='tight')
    print('wrote per-sample PNGs ->', OUT)
    print('wrote work_dirs/thesis_qual/contact_sheet.png')


if __name__ == '__main__':
    main()
