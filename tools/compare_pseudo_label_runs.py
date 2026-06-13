#!/usr/bin/env python3
"""
Cross-run pseudo-label quality comparison.

Loads all ps_label_eN.pkl files for two or more MT training runs, matches
pseudo-boxes against KITTI GT at BEV IoU 0.25 and 0.50, and produces:
  - <out-dir>/comparison_overview.png  — multi-panel evolution line plot
  - <out-dir>/comparison_stats.csv    — one row per (run, epoch)

Usage (from ~/mmdetection3d/):
    python tools/compare_pseudo_label_runs.py \\
        --runs "Teacher:work_dirs/mt_centerpoint_top0.3_minPts10_8jun/20260608_155316/ps_labels" \\
               "Student:work_dirs/selfTrain_centerpoint_top0.3_minPts5_9jun/20260609_095355/ps_labels" \\
        --kitti-info data/kitti/kitti_infos_train.pkl \\
        --out-dir compare_ps_quality
"""

import argparse
import csv
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import helpers from the sibling script — avoids duplicating GT-loading and
# matching logic that already exists and is tested there.
sys.path.insert(0, os.path.dirname(__file__))
from visualize_pseudo_labels import (
    build_gt_lookup,
    gt_boxes_to_nus,
    match_boxes,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Compare pseudo-label quality across runs and epochs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True,
                   metavar='NAME:PKL_DIR',
                   help='One or more run specs in the form "Label:path/to/ps_labels/"')
    p.add_argument('--kitti-info', default='data/kitti/kitti_infos_train.pkl',
                   help='KITTI train info pkl (GT source)')
    p.add_argument('--out-dir', default='compare_ps_quality',
                   help='Directory for output PNG and CSV')
    p.add_argument('--iou-thrs', default='0.25,0.50',
                   help='Comma-separated BEV IoU thresholds for precision/recall')
    return p.parse_args()


def parse_runs(run_specs: list[str]) -> list[tuple[str, str]]:
    """Parse ["Label:path", ...] → [(label, abs_path), ...]."""
    result = []
    for spec in run_specs:
        if ':' not in spec:
            raise ValueError(f'--runs entries must be "Name:path", got: {spec!r}')
        name, path = spec.split(':', 1)
        result.append((name.strip(), os.path.abspath(path.strip())))
    return result


def discover_epochs(pkl_dir: str) -> list[tuple[int, str]]:
    """Return sorted [(epoch_int, abs_pkl_path), ...] from a ps_labels dir."""
    files = sorted(os.listdir(pkl_dir))
    epochs = []
    for f in files:
        m = re.fullmatch(r'ps_label_e(\d+)\.pkl', f)
        if m:
            epochs.append((int(m.group(1)), os.path.join(pkl_dir, f)))
    return epochs


def detect_ps_car_label(ps_data: dict) -> int:
    """Return the integer label used for Car in this pkl.

    If only one unique label exists, assume it is Car.
    If multiple exist, warn and return 2 (KITTI convention).
    """
    all_labels = []
    for v in ps_data.values():
        if isinstance(v, dict) and 'gt_labels' in v:
            all_labels.extend(v['gt_labels'].tolist())
    unique = sorted(set(all_labels))
    if len(unique) == 1:
        print(f'  PS Car label = {unique[0]}  (single unique label → assumed Car)')
        return unique[0]
    print(f'  WARNING: multiple PS labels {unique} — using 2 (KITTI Car). '
          'Pass --ps-car-label if this is wrong.')
    return 2


def compute_epoch_stats(pkl_path: str,
                        gt_lookup: dict,
                        gt_car_label: int,
                        iou_thrs: list[float]) -> dict:
    """Compute aggregate stats for one epoch's ps_label pkl vs KITTI GT.

    Returns a flat dict; keys:
      epoch, n_boxes, n_scenes_with_boxes, n_scenes_total, scene_coverage_pct,
      mean_score, median_score, std_score, frac_score_lt_0p3,
      pt_median, pt_frac_le5_pct, pt_frac_le10_pct,
      precision_{thr}, recall_{thr}, f1_{thr}  for each thr in iou_thrs.
    """
    with open(pkl_path, 'rb') as f:
        ps_data = pickle.load(f)

    epoch = int(re.search(r'e(\d+)', os.path.basename(pkl_path)).group(1))
    ps_car_label = detect_ps_car_label(ps_data)

    totals = {thr: {'tp': 0, 'fp': 0, 'fn': 0} for thr in iou_thrs}
    all_scores: list[float] = []
    all_pts: list[float] = []
    n_scenes_with_boxes = 0

    for pkl_key, entry in ps_data.items():
        if not isinstance(entry, dict) or 'gt_boxes' not in entry:
            continue

        # Match pkl key → GT scene id (e.g. "data/kitti/.../000042.bin" → "000042")
        scene_id = os.path.splitext(os.path.basename(pkl_key))[0]

        ps_boxes  = entry['gt_boxes']
        ps_labels = entry['gt_labels']
        ps_scores = entry['scores']
        ps_pts    = entry['pt_counts']

        # Car-only subset
        car_mask      = ps_labels == ps_car_label
        ps_boxes_car  = ps_boxes[car_mask].astype(np.float32)
        ps_scores_car = ps_scores[car_mask]
        ps_pts_car    = ps_pts[car_mask]

        if len(ps_boxes_car) > 0:
            n_scenes_with_boxes += 1
            all_scores.extend(ps_scores_car.tolist())
            all_pts.extend(ps_pts_car.tolist())

        # GT Car boxes for this scene
        gt_info = gt_lookup.get(scene_id)
        if gt_info is not None and len(gt_info['cam_boxes']) > 0:
            gt_all = gt_boxes_to_nus(gt_info['cam_boxes'], gt_info['lidar2cam'])
            gt_lbl = gt_info['labels']
            gt_car = gt_all[gt_lbl == gt_car_label].astype(np.float32)
        else:
            gt_car = np.zeros((0, 7), dtype=np.float32)

        for thr in iou_thrs:
            tp, fp, fn = match_boxes(ps_boxes_car, gt_car, thr)
            totals[thr]['tp'] += tp
            totals[thr]['fp'] += fp
            totals[thr]['fn'] += fn

    scores = np.array(all_scores, dtype=np.float32)
    pts    = np.array(all_pts,    dtype=np.float32)
    n_tot  = len(ps_data)

    stats: dict = {
        'epoch':               epoch,
        'n_boxes':             int(len(scores)),
        'n_scenes_with_boxes': n_scenes_with_boxes,
        'n_scenes_total':      n_tot,
        'scene_coverage_pct':  100.0 * n_scenes_with_boxes / max(n_tot, 1),
        'mean_score':          float(scores.mean())          if len(scores) else 0.0,
        'median_score':        float(np.median(scores))      if len(scores) else 0.0,
        'std_score':           float(scores.std())           if len(scores) else 0.0,
        'frac_score_lt_0p3':   float((scores < 0.3).mean())  if len(scores) else 0.0,
        'pt_median':           float(np.median(pts))         if len(pts)    else 0.0,
        'pt_frac_le5_pct':     100.0 * float((pts <= 5).mean())  if len(pts) else 0.0,
        'pt_frac_le10_pct':    100.0 * float((pts <= 10).mean()) if len(pts) else 0.0,
    }

    for thr in iou_thrs:
        tp   = totals[thr]['tp']
        fp   = totals[thr]['fp']
        fn   = totals[thr]['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        key  = f'{thr:.2f}'
        stats[f'precision_{key}'] = prec
        stats[f'recall_{key}']    = rec
        stats[f'f1_{key}']        = f1

    return stats


# Colour cycle for runs
_RUN_COLOURS = ['#4db8ff', '#ff7f50', '#7fff7f', '#ffcc44']


def plot_comparison(all_run_stats: dict[str, list[dict]],
                    iou_thrs: list[float],
                    out_path: str) -> None:
    """Save a 2×4 multi-panel comparison figure (dark theme)."""
    fig, axes = plt.subplots(2, 4, figsize=(26, 10))
    fig.patch.set_facecolor('#0a0a0a')

    def _style_ax(ax, title, ylabel, ylim=None):
        ax.set_facecolor('#111111')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        ax.set_title(title, color='white', fontsize=10, pad=6)
        ax.set_xlabel('Epoch', color='white', fontsize=9)
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        ax.grid(True, alpha=0.15, color='white', linewidth=0.5)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, facecolor='#222222',
                  edgecolor='white', labelcolor='white')

    thr_025 = f'{iou_thrs[0]:.2f}'
    thr_050 = f'{iou_thrs[1]:.2f}' if len(iou_thrs) > 1 else '0.50'

    # (row, col, stat_key, title, ylabel, ylim)
    panels = [
        (0, 0, f'precision_{thr_025}', f'Precision @ BEV IoU {thr_025}', 'Precision', (0, 1)),
        (0, 1, f'recall_{thr_025}',    f'Recall @ BEV IoU {thr_025}',    'Recall',    (0, 1)),
        (0, 2, f'precision_{thr_050}', f'Precision @ BEV IoU {thr_050}', 'Precision', (0, 1)),
        (0, 3, f'recall_{thr_050}',    f'Recall @ BEV IoU {thr_050}',    'Recall',    (0, 1)),
        (1, 0, 'mean_score',           'Mean PS score',                   'Score',     (0, 1)),
        (1, 1, 'n_boxes',              'Total PS boxes',                  '# boxes',   None),
        (1, 2, 'pt_frac_le10_pct',     'PS boxes with ≤10 pts (%)',       '%',         (0, None)),
        (1, 3, 'scene_coverage_pct',   'Scene coverage (%)',              '%',         (0, 100)),
    ]

    for colour, (run_name, stats_list) in zip(_RUN_COLOURS, all_run_stats.items()):
        epochs = [s['epoch'] for s in stats_list]
        for row, col, key, title, ylabel, ylim in panels:
            values = [s[key] for s in stats_list]
            axes[row][col].plot(epochs, values, marker='o', linewidth=1.8,
                                markersize=5, color=colour, label=run_name)

    for row, col, key, title, ylabel, ylim in panels:
        _style_ax(axes[row][col], title, ylabel, ylim)

    fig.suptitle('Pseudo-Label Quality Evolution', color='white',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close(fig)
    print(f'\nComparison plot saved → {out_path}')


def save_csv(all_run_stats: dict[str, list[dict]],
             iou_thrs: list[float],
             out_path: str) -> None:
    """Write comparison_stats.csv with one row per (run, epoch)."""
    thr_keys = [f'{t:.2f}' for t in iou_thrs]
    fieldnames = (
        ['run', 'epoch', 'n_boxes', 'n_scenes_with_boxes', 'n_scenes_total',
         'scene_coverage_pct', 'mean_score', 'median_score', 'std_score',
         'frac_score_lt_0p3', 'pt_median', 'pt_frac_le5_pct', 'pt_frac_le10_pct']
        + [f'precision_{k}' for k in thr_keys]
        + [f'recall_{k}'    for k in thr_keys]
        + [f'f1_{k}'        for k in thr_keys]
    )
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, stats_list in all_run_stats.items():
            for s in stats_list:
                row = {'run': run_name}
                row.update({
                    k: f'{s[k]:.4f}' if isinstance(s[k], float) else s[k]
                    for k in fieldnames if k != 'run'
                })
                writer.writerow(row)
    print(f'CSV saved → {out_path}')


def main():
    args = parse_args()
    iou_thrs = [float(x) for x in args.iou_thrs.split(',')]
    runs = parse_runs(args.runs)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading GT from {args.kitti_info} ...')
    gt_lookup, gt_car_label = build_gt_lookup(args.kitti_info)
    print(f'  GT Car label = {gt_car_label}, {len(gt_lookup)} scenes in GT lookup')

    all_run_stats: dict[str, list[dict]] = {}

    for run_name, pkl_dir in runs:
        epochs = discover_epochs(pkl_dir)
        print(f'\n[{run_name}]  processing {len(epochs)} epoch pkl(s) ...')
        run_stats = []
        for ep, path in epochs:
            print(f'  epoch {ep} ...', end=' ', flush=True)
            s = compute_epoch_stats(path, gt_lookup, gt_car_label, iou_thrs)
            run_stats.append(s)
            thr_str = '  '.join(
                f'P@{thr:.2f}={s[f"precision_{thr:.2f}"]:.3f}  '
                f'R@{thr:.2f}={s[f"recall_{thr:.2f}"]:.3f}'
                for thr in iou_thrs)
            print(f'boxes={s["n_boxes"]:5d}  score_mean={s["mean_score"]:.3f}  {thr_str}')
        all_run_stats[run_name] = run_stats

    plot_path = os.path.join(args.out_dir, 'comparison_overview.png')
    plot_comparison(all_run_stats, iou_thrs, plot_path)

    csv_path = os.path.join(args.out_dir, 'comparison_stats.csv')
    save_csv(all_run_stats, iou_thrs, csv_path)


if __name__ == '__main__':
    main()
