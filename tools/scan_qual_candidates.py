#!/usr/bin/env python3
"""Scan KITTI val with the final adapted detector and surface candidate scenes
for the qualitative figure: clean ('good') scenes and 'failure' scenes (a few
missed / mislocalized cars). Prints ranked frame IDs + stats and renders a BEV
contact sheet so frames can be chosen by eye.

Region: front ~0-55 m (croppable). GS applied to predictions (final method).
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
from mmdet3d.structures.ops.box_np_ops import points_in_rbbox
from mmdet3d.structures.ops import ground_snap_boxes
from visualize_pseudo_labels import (build_gt_lookup, gt_boxes_to_nus,
                                      load_points_nus, bev_iou_matrix)
from analyze_ps_geometry_bias import matched_pairs
from ceiling_oracle_decomp import three_d_iou
from plot_size_distributions import predict_boxes

CFG = 'work_dirs/mt_cp_groundSnap_19jun/mean_teacher_centerpoint_config.py'
CKPT = 'work_dirs/mt_cp_groundSnap_19jun/epoch_4.pth'
GS = dict(pctl=2.0, min_pts=25, margin=0.3, max_disp=0.0)
VAL = 'data/kitti/kitti_infos_val.pkl'
VELO = 'data/kitti/training/velodyne_reduced'
OUT = 'work_dirs/thesis_qual'
# front camera-FOV cone (nuScenes frame): y_nus = forward, x_nus = lateral.
# KITTI annotates only the front-camera FOV (~+-42 deg), so restrict here -> FP/FN
# become meaningful (out-of-FOV predictions are not counted as false positives).
YMIN, YMAX, XABS = 2.0, 55.0, 35.0
MIN_PTS = 25          # a GT car with >= this many points "should" be detected
CACHE = 'work_dirs/thesis_qual/scan_cache.pkl'


def in_region(b):
    y, x = b[:, 1], b[:, 0]
    return (y > YMIN) & (y < YMAX) & (np.abs(x) < 0.9 * y + 3.0)


def bev_corners(box):
    cx, cy, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    return loc @ np.array([[c, -s], [s, c]]).T + [cx, cy]


def scan():
    os.makedirs(OUT, exist_ok=True)
    cfg = Config.fromfile(CFG); cfg['class_names'] = ['Car']
    model = init_model(cfg, CKPT, device='cuda:0')
    gt_lookup, gt_car = build_gt_lookup(VAL)
    scenes = {}
    for n, scene in enumerate(gt_lookup):
        gi = gt_lookup[scene]
        bin_path = os.path.join(VELO, scene + '.bin')
        if not os.path.exists(bin_path):
            continue
        pts = load_points_nus(bin_path)
        G = (gt_boxes_to_nus(gi['cam_boxes'], gi['lidar2cam'])[gi['labels'] == gt_car]
             if len(gi['cam_boxes']) else np.zeros((0, 7), np.float32)).astype(np.float32)
        boxes, labels = predict_boxes(model, pts)
        P = boxes[labels == 0].astype(np.float32)
        if len(P):
            P, _ = ground_snap_boxes(P, pts, **GS)
        # restrict to front region
        G = G[in_region(G)] if len(G) else G
        P = P[in_region(P)] if len(P) else P
        # salient GT = enough points
        if len(G):
            ptc = points_in_rbbox(pts[:, :3], G, z_axis=2, origin=(0.5, 0.5, 0)).sum(0)
            Gs = G[ptc >= MIN_PTS]
        else:
            Gs = G
        n_gt = len(Gs)
        # match salient GT to predictions (BEV IoU >= 0.5)
        tp, fn, miou, n_misloc = 0, n_gt, 1.0, 0
        matched_gt = set()
        if n_gt and len(P):
            pairs = matched_pairs(P, Gs, 0.5)        # (pred_i, gt_j)
            tp = len(pairs); fn = n_gt - tp
            if pairs:
                iou3d = [float(three_d_iou(P[i:i+1], Gs[j:j+1])[0]) for i, j in pairs]
                miou = float(np.mean(iou3d))
                n_misloc = int(np.sum(np.array(iou3d) < 0.55))
                matched_gt = {j for _, j in pairs}
        # FP = preds not matching any GT (in region)
        fp = 0
        if len(P) and len(G):
            bp = bev_iou_matrix(P, G)
            fp = int((bp.max(1) < 0.3).sum())
        elif len(P):
            fp = len(P)
        scenes[scene] = dict(n_gt=n_gt, tp=tp, fn=fn, fp=fp, miou=miou,
                             misloc=n_misloc, P=P, Gs=Gs)   # points reloaded for render
        if (n + 1) % 500 == 0:
            print(f'  scanned {n+1}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(scenes, f)
    print('cached', CACHE)
    return scenes


def categorize(scenes):
    # 'good' = detects all cars (fn=0), few/no spurious boxes, decent localization.
    # We do NOT require every box to clear 3D-IoU 0.55: a low-3D-IoU box is usually
    # just the height/z residual and still looks correct in BEV (the figure's view).
    good, fail = [], []
    for k, s in scenes.items():
        ng = s['n_gt']
        if not (3 <= ng <= 7):
            continue
        if s['fn'] == 0 and s['fp'] <= 1 and s['miou'] >= 0.6:
            good.append((s['miou'], k, s))                       # clean
        elif (1 <= s['fn'] <= 2 or 1 <= s['misloc'] <= 2) and s['tp'] >= 2 and s['fp'] <= 2:
            # prefer clear misses (fn>0) over height-only mislocalizations
            key = (0 if s['fn'] > 0 else 1, s['fn'] + s['misloc'], -s['miou'])
            fail.append((key, k, s))
    good.sort(reverse=True)        # highest mean IoU first
    fail.sort(key=lambda x: x[0])  # misses first, then fewest/cleanest errors
    return good[:10], [(k_, ki, si) for (k_, ki, si) in fail[:6]]


def draw_bev(ax, s, title, pts):
    m = in_region(pts) if len(pts) else np.zeros(0, bool)
    p = pts[m]
    ax.scatter(p[:, 0], p[:, 1], s=0.2, c='0.7', linewidths=0)
    for g in s['Gs']:
        c = np.vstack([bev_corners(g), bev_corners(g)[0]])
        ax.plot(c[:, 0], c[:, 1], color='#2ca02c', lw=1.4)        # GT green
    for pb in s['P']:
        c = np.vstack([bev_corners(pb), bev_corners(pb)[0]])
        ax.plot(c[:, 0], c[:, 1], color='#ff7f0e', lw=1.2)        # pred orange
    ax.set_xlim(-XABS, XABS); ax.set_ylim(YMIN, YMAX)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def contact_sheet(good, fail):
    cand = [('GOOD', k, s) for _, k, s in good] + [('FAIL', k, s) for _, k, s in fail]
    ncol = 5
    nrow = int(np.ceil(len(cand) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.0))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, (tag, k, s) in zip(axes, cand):
        ax.axis('on')
        t = (f'{k} [{tag}]\n gt{s["n_gt"]} tp{s["tp"]} fn{s["fn"]} '
             f'fp{s["fp"]} ml{s["misloc"]} IoU{s["miou"]:.2f}')
        pts = load_points_nus(os.path.join(VELO, k + '.bin'))
        draw_bev(ax, s, t, pts)
    fig.tight_layout()
    out = os.path.join(OUT, 'candidates_contact.png')
    fig.savefig(out, dpi=160, transparent=False, bbox_inches='tight')
    print('wrote', out)


def main():
    if os.path.exists(CACHE) and '--rescan' not in sys.argv:
        with open(CACHE, 'rb') as f:
            scenes = pickle.load(f)
        print(f'loaded cache {CACHE} ({len(scenes)} scenes)')
    else:
        scenes = scan()
    good, fail = categorize(scenes)
    print('\n===== GOOD candidates (clean) =====')
    print(f'{"frame":8s} {"gt":>3s} {"tp":>3s} {"fn":>3s} {"fp":>3s} {"meanIoU":>7s}')
    for _, k, s in good:
        print(f'{k:8s} {s["n_gt"]:3d} {s["tp"]:3d} {s["fn"]:3d} {s["fp"]:3d} {s["miou"]:7.2f}')
    print('\n===== FAILURE candidates (1-2 misses / mislocalizations) =====')
    print(f'{"frame":8s} {"gt":>3s} {"tp":>3s} {"fn":>3s} {"fp":>3s} {"misloc":>6s} {"meanIoU":>7s}')
    for _, k, s in fail:
        print(f'{k:8s} {s["n_gt"]:3d} {s["tp"]:3d} {s["fn"]:3d} {s["fp"]:3d} '
              f'{s["misloc"]:6d} {s["miou"]:7.2f}')
    contact_sheet(good, fail)


if __name__ == '__main__':
    main()
