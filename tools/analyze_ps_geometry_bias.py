#!/usr/bin/env python3
"""Measure systematic per-parameter geometry bias of pseudo-boxes vs KITTI GT.

Greedy BEV-IoU match (thr 0.5) between a run's ps_label_eN.pkl and KITTI GT,
then report median/mean signed error per box parameter for matched pairs:
  dz_bottom = ps_z - gt_z        (vertical position; z is bottom-center)
  h_ratio   = ps_h / gt_h        (height scale)
  l_ratio, w_ratio               (footprint scale — already good per BEV-strict)
  yaw_err   (deg, folded to [0,90])

Usage:
  python tools/analyze_ps_geometry_bias.py PS_LABEL.pkl [--iou 0.5]
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from visualize_pseudo_labels import build_gt_lookup, gt_boxes_to_nus, bev_iou_matrix


def matched_pairs(pred, gt, iou_thr):
    """Greedy BEV-IoU match → list of (pred_idx, gt_idx)."""
    if len(pred) == 0 or len(gt) == 0:
        return []
    iou = bev_iou_matrix(pred, gt)
    mp, mg, pairs = set(), set(), []
    for idx in np.argsort(iou.ravel())[::-1]:
        i, j = divmod(int(idx), iou.shape[1])
        if iou[i, j] < iou_thr:
            break
        if i not in mp and j not in mg:
            mp.add(i); mg.add(j); pairs.append((i, j))
    return pairs


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

    dz, hr, lr, wr, yaw, gtz, psz, gth, psh = ([] for _ in range(9))
    n_match = 0
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
            n_match += 1
            dz.append(ps[i, 2] - gt[j, 2])
            psz.append(ps[i, 2]); gtz.append(gt[j, 2])
            hr.append(ps[i, 5] / gt[j, 5]); psh.append(ps[i, 5]); gth.append(gt[j, 5])
            lr.append(ps[i, 3] / gt[j, 3])
            wr.append(ps[i, 4] / gt[j, 4])
            d = abs(ps[i, 6] - gt[j, 6]) % np.pi
            yaw.append(np.degrees(min(d, np.pi - d)))

    def rep(name, a, unit=''):
        a = np.array(a)
        print(f'  {name:11s} median={np.median(a):+.3f}  mean={a.mean():+.3f}  '
              f'std={a.std():.3f}  p25={np.percentile(a,25):+.3f}  p75={np.percentile(a,75):+.3f} {unit}')

    print(f'\nMatched pairs (BEV IoU≥{args.iou}): {n_match}\n')
    rep('dz_bottom', dz, 'm   (ps_z - gt_z; +=pred too high)')
    rep('h_ratio',   hr, '    (ps_h / gt_h; >1=pred too tall)')
    rep('l_ratio',   lr, '    (length)')
    rep('w_ratio',   wr, '    (width)')
    rep('yaw_err',   yaw, 'deg')
    print()
    rep('gt_z',  gtz, 'm'); rep('ps_z',  psz, 'm')
    rep('gt_h',  gth, 'm'); rep('ps_h',  psh, 'm')


if __name__ == '__main__':
    main()
