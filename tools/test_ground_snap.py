#!/usr/bin/env python3
"""Test 'ground-snap' on pseudo-labels: snap each box bottom to local ground
(robust low-percentile of the LiDAR points in its BEV footprint), keeping height.

Association-free: each box uses only its own column of points. Reports, before vs
after snap: z-bottom error vs GT (median/std), and 3D-IoU precision/recall @0.5,0.7.
Optionally writes a snapped pkl so the existing analyze_/ceiling_ tools can run on it.

Usage:
  python tools/test_ground_snap.py PS_LABEL.pkl [--pctl 10] [--min-pts 10]
                                   [--margin 0.0] [--out SNAPPED.pkl]
"""
import argparse
import copy
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from visualize_pseudo_labels import (build_gt_lookup, gt_boxes_to_nus,
                                      bev_iou_matrix, load_points_nus)
from mmdet3d.structures.ops.box_np_ops import points_in_rbbox


def _col_mask(box, pts, lpad, wpad):
    col = box.copy()
    col[3] += lpad; col[4] += wpad
    col[2] = -5.0; col[5] = 10.0
    return points_in_rbbox(pts[:, :3], col[None], z_axis=2, origin=(0.5, 0.5, 0))[:, 0]


def estimate_ground(box, pts, pctl, min_pts, margin, mode):
    """Robust local ground z, or None if too few points.

    mode='inbox' : low percentile of points inside footprint(+margin).
    mode='annulus': median of road points in a ring (box+margin) minus (box),
                    i.e. visible ground around the car (not occluded)."""
    if mode == 'annulus':
        outer = _col_mask(box, pts, margin, margin)
        inner = _col_mask(box, pts, 0.0, 0.0)
        ring = outer & ~inner
        z = pts[ring, 2]
        if z.size < min_pts:
            return None
        return float(np.percentile(z, 50))      # median road height
    z = pts[_col_mask(box, pts, margin, margin), 2]
    if z.size < min_pts:
        return None
    return float(np.percentile(z, pctl))


def ground_snap(boxes, pts, pctl, min_pts, margin, mode):
    """Return (snapped boxes, n_snapped). Rigid z shift: bottom->ground, h kept."""
    out = boxes.copy()
    n = 0
    for i in range(len(boxes)):
        g = estimate_ground(boxes[i], pts, pctl, min_pts, margin, mode)
        if g is not None:
            out[i, 2] = g
            n += 1
    return out, n


def iou3d_matrix(P, G):
    """3D IoU between P(M,7) and G(N,7); boxes [x,y,z(bottom),l,w,h,yaw]."""
    if len(P) == 0 or len(G) == 0:
        return np.zeros((len(P), len(G)), np.float32)
    bev = bev_iou_matrix(P, G)
    ap = (P[:, 3] * P[:, 4])[:, None]; ag = (G[:, 3] * G[:, 4])[None, :]
    bev_inter = np.where(bev > 0, bev * (ap + ag) / (1.0 + bev), 0.0)
    topp = (P[:, 2] + P[:, 5])[:, None]; topg = (G[:, 2] + G[:, 5])[None, :]
    zov = np.clip(np.minimum(topp, topg) - np.maximum(P[:, 2][:, None], G[:, 2][None, :]), 0, None)
    vi = bev_inter * zov
    vp = (ap * P[:, 5][:, None]); vg = (ag * G[:, 5][None, :])
    denom = vp + vg - vi
    return np.where(denom > 0, vi / denom, 0.0).astype(np.float32)


def greedy_tp(iou, thr):
    """# matched pairs with IoU>=thr (greedy, 1-1)."""
    if iou.size == 0:
        return 0
    mp, mg, tp = set(), set(), 0
    for idx in np.argsort(iou.ravel())[::-1]:
        i, j = divmod(int(idx), iou.shape[1])
        if iou[i, j] < thr:
            break
        if i not in mp and j not in mg:
            mp.add(i); mg.add(j); tp += 1
    return tp


def evaluate(ps_data, gt_lookup, gt_car_label, car_label,
             mode='orig', pctl=10, min_pts=10, margin=0.0, out_pkl=None):
    snap = mode in ('inbox', 'annulus')
    n_pred = n_gt = tp5 = tp7 = n_snap = 0
    dz, hr = [], []
    snapped_store = {} if out_pkl else None
    for key, entry in ps_data.items():
        if not isinstance(entry, dict) or 'gt_boxes' not in entry:
            continue
        scene = os.path.splitext(os.path.basename(key))[0]
        m = entry['gt_labels'] == car_label
        P = entry['gt_boxes'][m].astype(np.float32)
        gi = gt_lookup.get(scene)
        G = (gt_boxes_to_nus(gi['cam_boxes'], gi['lidar2cam'])[gi['labels'] == gt_car_label]
             if gi is not None else np.zeros((0, 7), np.float32)).astype(np.float32)
        if snap and len(P):
            pts = load_points_nus(key)
            P, ns = ground_snap(P, pts, pctl, min_pts, margin, mode)
            n_snap += ns
        elif mode == 'oracle_z' and len(P) and len(G):
            bev = bev_iou_matrix(P, G)
            best = bev.argmax(axis=1)
            ok = bev.max(axis=1) >= 0.3
            P = P.copy()
            P[ok, 2] = G[best[ok], 2]            # snap bottom z to matched GT
        if out_pkl is not None:
            e = copy.deepcopy(entry)
            e['gt_boxes'] = e['gt_boxes'].astype(np.float32)
            e['gt_boxes'][m] = P
            snapped_store[key] = e
        n_pred += len(P); n_gt += len(G)
        iou = iou3d_matrix(P, G)
        tp5 += greedy_tp(iou, 0.5); tp7 += greedy_tp(iou, 0.7)
        # geometry vs GT (BEV-matched pairs for fair dz/hr, reuse bev>=0.5)
        if len(P) and len(G):
            bev = bev_iou_matrix(P, G)
            mp, mg = set(), set()
            for idx in np.argsort(bev.ravel())[::-1]:
                i, j = divmod(int(idx), bev.shape[1])
                if bev[i, j] < 0.5:
                    break
                if i not in mp and j not in mg:
                    mp.add(i); mg.add(j)
                    dz.append(P[i, 2] - G[j, 2]); hr.append(P[i, 5] / G[j, 5])
    if out_pkl is not None:
        with open(out_pkl, 'wb') as f:
            pickle.dump(snapped_store, f)
    dz, hr = np.array(dz), np.array(hr)
    tags = {'orig': 'ORIGINAL', 'oracle_z': 'ORACLE-z (snap bottom->GT z)',
            'inbox': f'SNAP inbox(p{pctl},mg{margin})',
            'annulus': f'SNAP annulus(mg{margin},median)'}
    tag = tags.get(mode, mode)
    print(f'\n== {tag} ==   pred={n_pred} gt={n_gt}'
          + (f'  snapped={n_snap}/{n_pred}' if snap else ''))
    sys.stdout.flush()
    print(f'  dz_bottom  median={np.median(dz):+.3f}  mean={dz.mean():+.3f}  std={dz.std():.3f} m')
    print(f'  h_ratio    median={np.median(hr):+.3f}  (unchanged by snap)')
    for thr, tp in ((0.5, tp5), (0.7, tp7)):
        p = 100 * tp / max(n_pred, 1); r = 100 * tp / max(n_gt, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f'  3D@{thr}: precision={p:5.2f}  recall={r:5.2f}  F1={f1:5.2f}  (TP={tp})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl')
    ap.add_argument('--kitti-info', default='data/kitti/kitti_infos_train.pkl')
    ap.add_argument('--car-label', type=int, default=0, help='ps Car label')
    ap.add_argument('--pctl', type=float, default=10.0)
    ap.add_argument('--min-pts', type=int, default=10)
    ap.add_argument('--margin', type=float, default=0.0)
    ap.add_argument('--modes', default='orig,oracle_z,inbox,annulus')
    ap.add_argument('--out', default=None, help='write snapped pkl here (uses last mode)')
    args = ap.parse_args()

    gt_lookup, gt_car_label = build_gt_lookup(args.kitti_info)
    with open(args.pkl, 'rb') as f:
        ps_data = pickle.load(f)

    modes = args.modes.split(',')
    for mode in modes:
        evaluate(ps_data, gt_lookup, gt_car_label, args.car_label, mode=mode,
                 pctl=args.pctl, min_pts=args.min_pts, margin=args.margin,
                 out_pkl=args.out if mode == modes[-1] else None)


if __name__ == '__main__':
    main()
