#!/usr/bin/env python3
"""Offline sweep of ground-snap params on a stored ps-label pkl.

Re-snaps each config from the stored boxes (valid: the ground estimate is
footprint-based, independent of the box's current z, so re-snapping with the
SAME params is idempotent and with DIFFERENT params equals snapping the raw
prediction). Reports the same matched-pair frac@3D-IoU>=0.7 metric as
ceiling_oracle_decomp ('actual'), plus dz_bottom std, so each config is directly
comparable to the 'actual' (58%) and 'oracle z' (70.4%) numbers.

NOTE: max_disp is NOT swept here — it caps displacement relative to the RAW
prediction z, which is unrecoverable from an already-snapped store. Validate
max_disp in-run.

Usage: python tools/sweep_ground_snap.py PKL
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from visualize_pseudo_labels import build_gt_lookup, gt_boxes_to_nus, load_points_nus
from analyze_ps_geometry_bias import matched_pairs
from ceiling_oracle_decomp import three_d_iou
from mmdet3d.structures.ops import ground_snap_boxes

PKL = sys.argv[1]
CAR = 0

CONFIGS = [
    ('current  p2/m25/mg.3', dict(pctl=2, min_pts=25, margin=0.3)),
    ('p1/m25/mg.3',          dict(pctl=1, min_pts=25, margin=0.3)),
    ('p3/m25/mg.3',          dict(pctl=3, min_pts=25, margin=0.3)),
    ('p2/m15/mg.3',          dict(pctl=2, min_pts=15, margin=0.3)),
    ('p2/m40/mg.3',          dict(pctl=2, min_pts=40, margin=0.3)),
    ('p2/m25/mg.2',          dict(pctl=2, min_pts=25, margin=0.2)),
    ('p2/m25/mg.5',          dict(pctl=2, min_pts=25, margin=0.5)),
    ('p1/m40/mg.3',          dict(pctl=1, min_pts=40, margin=0.3)),
    ('p1/m15/mg.2',          dict(pctl=1, min_pts=15, margin=0.2)),
]

gt_lookup, gt_car = build_gt_lookup('data/kitti/kitti_infos_train.pkl')
data = pickle.load(open(PKL, 'rb'))

# Measured e0 median size ratios (pred/GT) → multiplicative de-bias constants.
RL, RW, RH = 1.046, 1.056, 1.025

# accumulators per config: matched count, >=0.7 count, dz list
acc = {name: [0, 0, []] for name, _ in CONFIGS}
for extra in ('SIZE-debias (on GS)', 'ORACLE size', 'ORACLE size+z'):
    acc[extra] = [0, 0, []]
acc['ORACLE z'] = [0, 0, []]

for key, e in data.items():
    if not isinstance(e, dict) or 'gt_boxes' not in e:
        continue
    scene = os.path.splitext(os.path.basename(key))[0]
    m = e['gt_labels'] == CAR
    P0 = e['gt_boxes'][m].astype(np.float32)
    gi = gt_lookup.get(scene)
    if gi is None or len(P0) == 0:
        continue
    G = gt_boxes_to_nus(gi['cam_boxes'], gi['lidar2cam'])
    G = G[gi['labels'] == gt_car].astype(np.float32)
    if len(G) == 0:
        continue
    pts = load_points_nus(key)

    for name, cfg in CONFIGS:
        Pc, _ = ground_snap_boxes(P0, pts, labels=e['gt_labels'][m], **cfg)
        for i, j in matched_pairs(Pc, G, 0.5):
            iou = float(three_d_iou(Pc[i:i+1], G[j:j+1])[0])
            acc[name][0] += 1
            acc[name][1] += int(iou >= 0.7)
            acc[name][2].append(Pc[i, 2] - G[j, 2])
    # size de-bias on the (already ground-snapped) store: shrink l/w/h toward
    # the KITTI prior, keep x/y/yaw and the ground-snapped bottom z.
    Pd = P0.copy()
    Pd[:, 3] /= RL; Pd[:, 4] /= RW; Pd[:, 5] /= RH
    for i, j in matched_pairs(Pd, G, 0.5):
        iou = float(three_d_iou(Pd[i:i+1], G[j:j+1])[0])
        acc['SIZE-debias (on GS)'][0] += 1
        acc['SIZE-debias (on GS)'][1] += int(iou >= 0.7)

    # oracles: snap bottom z and/or size to matched GT
    for i, j in matched_pairs(P0, G, 0.5):
        Pz = P0[i:i+1].copy(); Pz[0, 2] = G[j, 2]
        acc['ORACLE z'][0] += 1
        acc['ORACLE z'][1] += int(float(three_d_iou(Pz, G[j:j+1])[0]) >= 0.7)
        Ps = P0[i:i+1].copy(); Ps[0, 3:6] = G[j, 3:6]
        acc['ORACLE size'][0] += 1
        acc['ORACLE size'][1] += int(float(three_d_iou(Ps, G[j:j+1])[0]) >= 0.7)
        Pb = P0[i:i+1].copy(); Pb[0, 2] = G[j, 2]; Pb[0, 3:6] = G[j, 3:6]
        acc['ORACLE size+z'][0] += 1
        acc['ORACLE size+z'][1] += int(float(three_d_iou(Pb, G[j:j+1])[0]) >= 0.7)

print(f'\n{PKL}')
print(f'{"config":22s}  {"matched":>8s}  {"frac@0.7":>8s}  {"dz_std":>7s}')
for name in ([c[0] for c in CONFIGS]
             + ['SIZE-debias (on GS)', 'ORACLE z', 'ORACLE size', 'ORACLE size+z']):
    mt, ge, dz = acc[name]
    frac = 100 * ge / max(mt, 1)
    s = f'{np.std(dz):.3f}' if dz else '  -  '
    print(f'{name:22s}  {mt:8d}  {frac:7.1f}%  {s:>7s}')
