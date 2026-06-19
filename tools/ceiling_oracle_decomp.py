#!/usr/bin/env python3
"""Decompose the 3D-strict (IoU>=0.7) ceiling into size vs vertical-localization.

For each teacher pseudo-box matched to KITTI GT (greedy BEV-IoU >= thr), compute the
true 3D IoU, then recompute it under oracle corrections:
  size  : set pred l,w,h := gt l,w,h   (keep center x,y, bottom z, yaw)
  z     : set pred bottom z := gt z     (keep size)
  both  : size and z
Report median 3D IoU and the fraction crossing 0.7 (the AP40-strict bar) for each.

This isolates whether the ceiling is median-size-bound (size oracle recovers it) or
localization/variance-bound (it does not).

Usage: python tools/ceiling_oracle_decomp.py PS_LABEL.pkl [--iou 0.5]
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from visualize_pseudo_labels import build_gt_lookup, gt_boxes_to_nus, bev_iou_matrix
from analyze_ps_geometry_bias import matched_pairs


def diag_bev_iou(P, G):
    """BEV IoU for paired boxes P[k], G[k] -> (K,) via blocked diagonal."""
    K = len(P)
    out = np.zeros(K, dtype=np.float32)
    for s in range(0, K, 1000):
        e = min(s + 1000, K)
        m = bev_iou_matrix(P[s:e], G[s:e])
        out[s:e] = np.diag(m)
    return out


def three_d_iou(P, G):
    """3D IoU for paired boxes. Boxes: [x,y,z(bottom),l,w,h,yaw]."""
    bev = diag_bev_iou(P, G)
    area_p = P[:, 3] * P[:, 4]
    area_g = G[:, 3] * G[:, 4]
    bev_inter = bev * (area_p + area_g) / (1.0 + bev)
    top_p, top_g = P[:, 2] + P[:, 5], G[:, 2] + G[:, 5]
    zov = np.clip(np.minimum(top_p, top_g) - np.maximum(P[:, 2], G[:, 2]), 0, None)
    vol_inter = bev_inter * zov
    vol_p, vol_g = area_p * P[:, 5], area_g * G[:, 5]
    denom = vol_p + vol_g - vol_inter
    return np.where(denom > 0, vol_inter / denom, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl')
    ap.add_argument('--kitti-info', default='data/kitti/kitti_infos_train.pkl')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--ps-car-label', type=int, default=0)
    args = ap.parse_args()

    gt_lookup, gt_car_label = build_gt_lookup(args.kitti_info)
    with open(args.pkl, 'rb') as f:
        ps_data = pickle.load(f)

    P, G = [], []
    for key, entry in ps_data.items():
        if not isinstance(entry, dict) or 'gt_boxes' not in entry:
            continue
        scene = os.path.splitext(os.path.basename(key))[0]
        m = entry['gt_labels'] == args.ps_car_label
        ps = entry['gt_boxes'][m].astype(np.float32)
        gi = gt_lookup.get(scene)
        if gi is None or len(gi['cam_boxes']) == 0 or len(ps) == 0:
            continue
        gt = gt_boxes_to_nus(gi['cam_boxes'], gi['lidar2cam'])
        gt = gt[gi['labels'] == gt_car_label].astype(np.float32)
        if len(gt) == 0:
            continue
        for i, j in matched_pairs(ps, gt, args.iou):
            P.append(ps[i, :7]); G.append(gt[j, :7])
    P, G = np.array(P, np.float32), np.array(G, np.float32)
    print(f'Matched pairs (BEV IoU>={args.iou}): {len(P)}\n')

    def variant(name, Pmod):
        iou = three_d_iou(Pmod, G)
        for thr in (0.7, 0.5):
            frac = (iou >= thr).mean() * 100
            print(f'  {name:16s} median3DIoU={np.median(iou):.3f}  '
                  f'frac>={thr}: {frac:5.1f}%', end='')
            if thr == 0.7:
                print('   |', end='')
        print()

    # actual
    variant('actual', P.copy())
    # size -> gt (keep x,y,z,yaw)
    Ps = P.copy(); Ps[:, 3:6] = G[:, 3:6]
    variant('oracle size', Ps)
    # z -> gt (keep size)
    Pz = P.copy(); Pz[:, 2] = G[:, 2]
    variant('oracle z', Pz)
    # both
    Pb = P.copy(); Pb[:, 2] = G[:, 2]; Pb[:, 3:6] = G[:, 3:6]
    variant('oracle size+z', Pb)
    # full footprint+yaw too (everything but center x,y) -> isolates pure x,y center error
    Pc = G.copy(); Pc[:, 0:2] = P[:, 0:2]
    variant('only xy err', Pc)


if __name__ == '__main__':
    main()
