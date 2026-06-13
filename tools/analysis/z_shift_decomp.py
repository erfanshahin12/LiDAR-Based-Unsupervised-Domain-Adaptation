"""z_shift_decomp.py — Matched-pair z-shift decomposition for CenterPoint KITTI eval.

For every KITTI val frame, match each prediction to its nearest KITTI GT Car
in BEV (Euclidean distance of center), then decompose the center-z residual into:

    center_residual = pred_cz  − gt_cz
                    = bottom_residual + Δh/2
    bottom_residual = pred_bot − gt_bot
    Δh              = pred_h   − gt_h

All quantities are in KITTI LiDAR frame.

Usage
-----
Requires the ps_label / prediction pkl produced by filter_teacher_predictions
(nuScenes frame, bottom-center).  Supply the path to that pkl plus the KITTI
val info pkl.

    cd ~/mmdetection3d
    conda activate thesis
    python tools/analysis/z_shift_decomp.py \\
        --pred-pkl work_dirs/<run>/<ts>/ps_labels/ps_label_e0.pkl \\
        --kitti-info data/kitti/kitti_infos_val.pkl \\
        [--bev-match-thr 2.0] [--min-h 1.0]

The script prints aggregate statistics and per-bin breakdowns.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

# ── Setup path so the script works when run from mmdetection3d/ root ──────────
import sys, os
sys.path.insert(0, os.getcwd())

from mmdet3d.structures import CameraInstance3DBoxes, Coord3DMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_kitti_gt(info_path: str):
    """Return dict: lidar_path -> (N,7) array in KITTI LiDAR frame (bottom-center)."""
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    dl = info['data_list']
    out = {}
    for s in dl:
        lp = s['lidar_points']['lidar_path']
        l2c = np.array(s['images']['CAM2']['lidar2cam'])
        cam_boxes = [it['bbox_3d'] for it in s['instances'] if it['bbox_label_3d'] == 2]
        if not cam_boxes:
            out[lp] = np.zeros((0, 7), dtype=np.float32)
            continue
        cb = CameraInstance3DBoxes(
            torch.tensor(np.array(cam_boxes, dtype=np.float32)), box_dim=7)
        lb = cb.convert_to(Coord3DMode.LIDAR, np.linalg.inv(l2c))
        out[lp] = lb.tensor.numpy()   # (N, 7), bottom-center in KITTI LiDAR
    return out


def nus_preds_to_kitti(boxes_nus: np.ndarray) -> np.ndarray:
    """Convert nuScenes-frame bottom-center preds → KITTI LiDAR frame.

    Mirrors NusOnKittiMetric._nus_to_kitti_boxes:
        x_kitti =  y_nus
        y_kitti = -x_nus
        z_kitti =  z_nus + 0.11          (sensor-height undo)
        yaw_kitti = yaw_nus - π/2
        l, w unchanged
    """
    t = boxes_nus.copy()
    x = t[:, 0].copy()
    y = t[:, 1].copy()
    t[:, 0] = y
    t[:, 1] = -x
    t[:, 2] = t[:, 2] + 0.11
    t[:, 6] = t[:, 6] - np.pi / 2
    return t   # still bottom-center


def bev_match(pred_xy: np.ndarray, gt_xy: np.ndarray, thr: float):
    """Greedy nearest-neighbour BEV match. Returns (pred_idx, gt_idx) arrays."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Distance matrix (M_pred x N_gt)
    diff = pred_xy[:, None, :] - gt_xy[None, :, :]   # (M, N, 2)
    dist = np.sqrt((diff ** 2).sum(axis=-1))           # (M, N)

    pred_idxs, gt_idxs = [], []
    matched_gt = set()

    # Sort predictions by their minimum distance to any GT
    order = np.argsort(dist.min(axis=1))
    for pi in order:
        gi = int(np.argmin(dist[pi]))
        if gi in matched_gt:
            continue
        if dist[pi, gi] <= thr:
            pred_idxs.append(pi)
            gt_idxs.append(gi)
            matched_gt.add(gi)

    return np.array(pred_idxs, dtype=int), np.array(gt_idxs, dtype=int)


def print_stats(name: str, arr: np.ndarray):
    if len(arr) == 0:
        print(f"  {name}: (no data)")
        return
    print(f"  {name}: n={len(arr):5d}  mean={arr.mean():+.3f}  "
          f"median={np.median(arr):+.3f}  std={arr.std():.3f}  "
          f"[p5={np.percentile(arr,5):+.3f}  p95={np.percentile(arr,95):+.3f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred-pkl', required=True,
                    help='ps_label_e*.pkl (keys = KITTI lidar paths; '
                         'gt_boxes in nuScenes frame, bottom-center)')
    ap.add_argument('--kitti-info', default='data/kitti/kitti_infos_val.pkl')
    ap.add_argument('--bev-match-thr', type=float, default=2.0,
                    help='Max BEV distance (m) for a valid match')
    ap.add_argument('--min-h', type=float, default=1.0,
                    help='Min predicted height to include in stats')
    args = ap.parse_args()

    print(f"\nLoading KITTI GT from {args.kitti_info} …")
    gt_by_frame = load_kitti_gt(args.kitti_info)

    print(f"Loading predictions from {args.pred_pkl} …")
    with open(args.pred_pkl, 'rb') as f:
        pred_store = pickle.load(f)

    # Normalise prediction keys to bare filename stem so they match GT keys
    # (GT keys: '000001.bin'; pred keys: 'data/kitti/training/.../000001.bin')
    pred_by_stem = {Path(k).name: v for k, v in pred_store.items()}
    print(f"  pred frames: {len(pred_by_stem)}  GT frames: {len(gt_by_frame)}")
    overlap = sum(1 for k in gt_by_frame if k in pred_by_stem)
    print(f"  overlapping frames: {overlap}")

    bottom_res_all = []
    dh_all = []
    center_res_all = []
    pred_h_all = []
    gt_h_all = []
    n_frames_matched = 0

    for lp, gt_boxes in gt_by_frame.items():
        # Match by bare filename
        entry = pred_by_stem.get(lp)
        if entry is None or len(entry['gt_boxes']) == 0 or len(gt_boxes) == 0:
            continue

        pred_nus = entry['gt_boxes']                       # (M, 7) nuScenes frame
        pred_h_filter = pred_nus[:, 5] >= args.min_h
        pred_nus = pred_nus[pred_h_filter]
        if len(pred_nus) == 0:
            continue

        pred_kitti = nus_preds_to_kitti(pred_nus)          # (M, 7) KITTI LiDAR

        pred_idx, gt_idx = bev_match(
            pred_kitti[:, :2], gt_boxes[:, :2], args.bev_match_thr)

        if len(pred_idx) == 0:
            continue
        n_frames_matched += 1

        p = pred_kitti[pred_idx]   # matched preds
        g = gt_boxes[gt_idx]       # matched GTs

        # Bottom-center z convention: z = bottom z for both
        p_bot = p[:, 2]            # z IS the bottom for bottom-center
        g_bot = g[:, 2]
        p_h   = p[:, 5]
        g_h   = g[:, 5]
        p_cz  = p_bot + p_h / 2   # gravity center z
        g_cz  = g_bot + g_h / 2

        bottom_res_all.append(p_bot - g_bot)
        dh_all.append(p_h - g_h)
        center_res_all.append(p_cz - g_cz)
        pred_h_all.append(p_h)
        gt_h_all.append(g_h)

    if not bottom_res_all:
        print("\nNo matched pairs found — check that pred pkl keys match KITTI lidar paths.")
        return

    bottom_res = np.concatenate(bottom_res_all)
    dh         = np.concatenate(dh_all)
    center_res = np.concatenate(center_res_all)
    pred_h     = np.concatenate(pred_h_all)
    gt_h       = np.concatenate(gt_h_all)

    print(f"\n{'='*60}")
    print(f"Matched-pair z-shift decomposition")
    print(f"  Frames with ≥1 match : {n_frames_matched}")
    print(f"  Total matched pairs  : {len(center_res)}")
    print(f"  BEV match threshold  : {args.bev_match_thr} m")
    print(f"{'='*60}")
    print(f"\n-- Box HEIGHT (h) --")
    print_stats("pred_h     ", pred_h)
    print_stats("gt_h       ", gt_h)
    print_stats("Δh  (p−g)  ", dh)
    print(f"\n-- Z decomposition (KITTI LiDAR frame, bottom-center) --")
    print_stats("bottom_res (p_bot−g_bot)", bottom_res)
    print_stats("Δh/2                    ", dh / 2)
    print_stats("center_res (p_cz−g_cz)  ", center_res)
    print(f"\n-- Decomposition check --")
    reconstructed = bottom_res + dh / 2
    err = center_res - reconstructed
    print_stats("bottom_res + Δh/2 (reconstructed center_res)", reconstructed)
    print_stats("center_res − reconstructed  (should be ≈0)  ", err)

    print(f"\n{'='*60}")
    print(f"SUMMARY (medians):")
    print(f"  center_res  = {np.median(bottom_res):+.3f} (bottom) "
          f"+ {np.median(dh/2):+.3f} (Δh/2) "
          f"= {np.median(bottom_res) + np.median(dh/2):+.3f} "
          f"  [direct: {np.median(center_res):+.3f}]")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
