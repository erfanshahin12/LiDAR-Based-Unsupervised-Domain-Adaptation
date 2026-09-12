#!/usr/bin/env python3
"""Final thesis qualitative figures — 3 selected KITTI-val scenes.

One dark-background PNG per scene: BEV (top) + side-elevation (bottom).
Points colored by height (z).  GT green, adapted predictions orange.
Prediction filtering: score ≥ 0.40, top-70%, min-10-pts, ground-snap,
then center within PC range ±51.2 m in x and y.

Usage:
    python tools/render_thesis_samples.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mmengine.config import Config

sys.path.insert(0, os.path.dirname(__file__))
from mmdet3d.apis import init_model
from mmdet3d.structures.ops import ground_snap_boxes
from mmdet3d.structures.ops.box_np_ops import points_in_rbbox
from visualize_pseudo_labels import (build_gt_lookup, gt_boxes_to_nus,
                                      load_points_nus, run_baseline_raw)

# ── scene selection & paths ────────────────────────────────────────────────────
SCENES      = ['003708', '004188', '002454']
SCENE_3WAY  = '000684'   # rendered with GT + adapted + source-only
CFG         = 'work_dirs/mt_cp_groundSnap_19jun/mean_teacher_centerpoint_config.py'
CKPT        = 'work_dirs/mt_cp_groundSnap_19jun/epoch_4.pth'
SO_CFG      = 'work_dirs/baseline_centerpoint_18may/mt_pretrain_centerpoint_config.py'
SO_CKPT     = 'work_dirs/baseline_centerpoint_18may/epoch_20.pth'
GS_KW     = dict(pctl=2.0, min_pts=25, margin=0.3, max_disp=0.0)
VAL       = 'data/kitti/kitti_infos_val.pkl'
VELO      = 'data/kitti/training/velodyne_reduced'
OUT       = 'work_dirs/thesis_qual/final'

# ── prediction filtering ───────────────────────────────────────────────────────
SCORE_THR = 0.40
KEEP_FRAC = 0.70
MIN_PTS   = 10
PC_RANGE  = 51.2   # discard box if |x| or |y| of center exceeds this

# ── rendering style ────────────────────────────────────────────────────────────
BG     = 'black'
PT_C   = 'white'     # point color (flat white, no colormap)
GT_C   = '#2ca02c'   # green
PR_C   = '#ff7f0e'   # orange  (adapted)
SO_C   = '#d62728'   # red     (source-only)
PT_S   =  0.3        # scatter dot size

# display extents (nuScenes frame: x=right, y=forward, z=up)
# x: ±51.2 m  (full PC range, 102.4 m total)
# y: 0–51.2 m (frontal half of PC range, 51.2 m → 1:2 aspect with equal axes)
BEV_XLIM  = (-51.2,  51.2)
BEV_YLIM  = (  0.0,  51.2)
SIDE_XLIM = (  0.0,  51.2)   # y (forward), same range as BEV y
SIDE_YLIM = ( -3.0,   1.0)   # z (height)

LABEL_KW  = dict(color='0.55', fontsize=8)
BOX_LW    = 0.8   # box edge line width (thin for thesis)


# ── helpers ────────────────────────────────────────────────────────────────────

def in_pc_range(b):
    """True where box center |x| ≤ PC_RANGE and |y| ≤ PC_RANGE."""
    return (np.abs(b[:, 0]) <= PC_RANGE) & (np.abs(b[:, 1]) <= PC_RANGE)


def bev_corners(box):
    cx, cy, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    loc  = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    return loc @ np.array([[c, -s], [s, c]]).T + [cx, cy]


def filter_preds(boxes, cls_sc, labels, pts):
    """Score-thr → top-KEEP_FRAC → min-pts → GS → PC-range."""
    car_mask = (labels == 0) & (cls_sc >= SCORE_THR)
    P  = boxes[car_mask].astype(np.float32)
    sc = cls_sc[car_mask]
    if len(P) == 0:
        return P
    # top-KEEP_FRAC by score
    n_keep = int(np.ceil(KEEP_FRAC * len(P)))
    order  = np.argsort(sc)[::-1]
    P  = P[order[:n_keep]]
    # min interior points
    pt_cnts = points_in_rbbox(pts[:, :3], P, z_axis=2, origin=(0.5, 0.5, 0)).sum(0)
    P = P[pt_cnts >= MIN_PTS]
    if len(P) == 0:
        return P
    P, _ = ground_snap_boxes(P, pts, **GS_KW)
    P = P[in_pc_range(P)]
    return P


def scatter_pts(ax, pts, xidx, yidx, xlim, ylim):
    """Scatter pts[:,xidx] vs pts[:,yidx] in flat white within display bounds."""
    m = ((pts[:, xidx] >= xlim[0]) & (pts[:, xidx] <= xlim[1]) &
         (pts[:, yidx] >= ylim[0]) & (pts[:, yidx] <= ylim[1]))
    p = pts[m]
    ax.scatter(p[:, xidx], p[:, yidx], s=PT_S, c=PT_C,
               linewidths=0, rasterized=True)


ARROW_KW = dict(arrowstyle='->', color='0.55', lw=0.8, mutation_scale=8)
ARROW_C  = '0.55'
ARROW_FS = 7
_AL = 0.11   # arrow length in axes-fraction units (0.7 × ~0.16 ≈ 0.11)
_AX = 0.04   # arrow origin x (axes fraction)
_AY = 0.06   # arrow origin y (axes fraction)
_AF = 'axes fraction'


def _axis_arrows_bev(ax):
    """y (upward) and x (rightward) arrows in axes-fraction coords, bottom-left."""
    # y arrow — forward direction
    ax.annotate('', xy=(_AX, _AY + _AL), xytext=(_AX, _AY),
                xycoords=_AF, textcoords=_AF,
                arrowprops=ARROW_KW, annotation_clip=False)
    ax.text(_AX, _AY + _AL + 0.03, 'y', color=ARROW_C, fontsize=ARROW_FS,
            ha='center', va='bottom', transform=ax.transAxes)
    # x arrow — lateral direction
    ax.annotate('', xy=(_AX + _AL, _AY), xytext=(_AX, _AY),
                xycoords=_AF, textcoords=_AF,
                arrowprops=ARROW_KW, annotation_clip=False)
    ax.text(_AX + _AL + 0.02, _AY, 'x', color=ARROW_C, fontsize=ARROW_FS,
            ha='left', va='center', transform=ax.transAxes)


def _axis_arrows_side(ax):
    """y (rightward) and z (upward) arrows in axes-fraction coords, bottom-left."""
    # y arrow — forward direction
    ax.annotate('', xy=(_AX + _AL, _AY), xytext=(_AX, _AY),
                xycoords=_AF, textcoords=_AF,
                arrowprops=ARROW_KW, annotation_clip=False)
    ax.text(_AX + _AL + 0.02, _AY, 'y', color=ARROW_C, fontsize=ARROW_FS,
            ha='left', va='center', transform=ax.transAxes)
    # z arrow — height direction
    ax.annotate('', xy=(_AX, _AY + _AL), xytext=(_AX, _AY),
                xycoords=_AF, textcoords=_AF,
                arrowprops=ARROW_KW, annotation_clip=False)
    ax.text(_AX, _AY + _AL + 0.05, 'z', color=ARROW_C, fontsize=ARROW_FS,
            ha='center', va='bottom', transform=ax.transAxes)


def draw_bev(ax, pts, box_sets):
    """box_sets: list of (boxes_array, color) drawn in order."""
    scatter_pts(ax, pts, xidx=0, yidx=1, xlim=BEV_XLIM, ylim=BEV_YLIM)
    for boxes, col in box_sets:
        for box in boxes:
            c = np.vstack([bev_corners(box), bev_corners(box)[0]])
            ax.plot(c[:, 0], c[:, 1], color=col, lw=BOX_LW, solid_capstyle='round')
    _axis_arrows_bev(ax)
    ax.set_facecolor(BG)
    ax.set_xlim(*BEV_XLIM); ax.set_ylim(*BEV_YLIM)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_side(ax, pts, box_sets):
    """box_sets: list of (boxes_array, color) drawn in order."""
    scatter_pts(ax, pts, xidx=1, yidx=2, xlim=SIDE_XLIM, ylim=SIDE_YLIM)
    for boxes, col in box_sets:
        for box in boxes:
            ax.add_patch(Rectangle((box[1] - box[3] / 2, box[2]),
                                   box[3], box[5],
                                   fill=False, edgecolor=col, lw=BOX_LW))
    _axis_arrows_side(ax)
    ax.set_facecolor(BG)
    ax.set_xlim(*SIDE_XLIM); ax.set_ylim(*SIDE_YLIM)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── main ───────────────────────────────────────────────────────────────────────

def _get_gt(gi, gt_car):
    G = (gt_boxes_to_nus(gi['cam_boxes'], gi['lidar2cam'])[gi['labels'] == gt_car]
         if len(gi['cam_boxes']) else np.zeros((0, 7), np.float32)).astype(np.float32)
    return G[in_pc_range(G)] if len(G) else G


def _save_bev(ax, scene, legend_elems):
    ax.legend(handles=legend_elems, loc='upper right', frameon=False,
              labelcolor='white', fontsize=8, handlelength=1.4)
    fig = ax.get_figure()
    fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.04)
    path = os.path.join(OUT, f'bev_{scene}.png')
    fig.savefig(path, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'    → {path}')


def _save_side(ax, scene):
    fig = ax.get_figure()
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.12)
    path = os.path.join(OUT, f'side_{scene}.png')
    fig.savefig(path, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'    → {path}')


def main():
    os.makedirs(OUT, exist_ok=True)

    # adapted model
    cfg = Config.fromfile(CFG); cfg['class_names'] = ['Car']
    model = init_model(cfg, CKPT, device='cuda:0')

    # source-only model (loaded once, used only for SCENE_3WAY)
    so_cfg = Config.fromfile(SO_CFG); so_cfg['class_names'] = ['Car']
    so_model = init_model(so_cfg, SO_CKPT, device='cuda:0')

    gt_lookup, gt_car = build_gt_lookup(VAL)

    leg_2 = [plt.Line2D([0], [0], color=GT_C, lw=1.2, label='Ground truth'),
             plt.Line2D([0], [0], color=PR_C, lw=1.2, label='Adapted')]
    leg_3 = [plt.Line2D([0], [0], color=GT_C, lw=1.2, label='Ground truth'),
             plt.Line2D([0], [0], color=PR_C, lw=1.2, label='Adapted'),
             plt.Line2D([0], [0], color=SO_C, lw=1.2, label='Source-only')]

    # ── regular 2-way scenes ──────────────────────────────────────────────────
    for scene in SCENES:
        gi  = gt_lookup[scene]
        pts = load_points_nus(os.path.join(VELO, scene + '.bin'))
        G   = _get_gt(gi, gt_car)
        boxes, cls_sc, _iou, labels = run_baseline_raw(model, pts, 'cuda:0')
        P   = filter_preds(boxes, cls_sc, labels, pts)
        print(f'  {scene}: GT={len(G)}  pred={len(P)}')

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0), facecolor=BG)
        draw_bev(ax, pts, [(G, GT_C), (P, PR_C)])
        _save_bev(ax, scene, leg_2)

        fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.2), facecolor=BG)
        draw_side(ax, pts, [(G, GT_C), (P, PR_C)])
        _save_side(ax, scene)

    # ── 3-way scene: GT + adapted + source-only ───────────────────────────────
    scene = SCENE_3WAY
    gi  = gt_lookup[scene]
    pts = load_points_nus(os.path.join(VELO, scene + '.bin'))
    G   = _get_gt(gi, gt_car)

    boxes, cls_sc, _iou, labels = run_baseline_raw(model, pts, 'cuda:0')
    P   = filter_preds(boxes, cls_sc, labels, pts)

    so_boxes, so_cls, _iou2, so_labels = run_baseline_raw(so_model, pts, 'cuda:0')
    SO  = filter_preds(so_boxes, so_cls, so_labels, pts)

    print(f'  {scene} (3-way): GT={len(G)}  adapted={len(P)}  source-only={len(SO)}')

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0), facecolor=BG)
    draw_bev(ax, pts, [(G, GT_C), (P, PR_C), (SO, SO_C)])
    _save_bev(ax, scene, leg_3)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.2), facecolor=BG)
    draw_side(ax, pts, [(G, GT_C), (P, PR_C), (SO, SO_C)])
    _save_side(ax, scene)


if __name__ == '__main__':
    main()
