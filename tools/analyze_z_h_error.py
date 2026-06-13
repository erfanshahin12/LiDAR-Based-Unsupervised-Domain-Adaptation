#!/usr/bin/env python3
"""
Per-epoch z / h MAE analysis for a Mean Teacher CenterPoint run.

Loads each epoch checkpoint, runs the teacher on the KITTI val set
(converted to nuScenes frame by KittiToNuscenes inside val_pipeline),
matches predictions to GT by BEV centre L2 distance, and reports:

  z_MAE   mean |z_gravity_pred − z_gravity_gt|   [m]
  h_MAE   mean |h_pred − h_gt|                   [m]
  z_p50   median z error                          [m]
  h_p50   median h error                          [m]
  match%  fraction of GT boxes matched

Both prediction and GT boxes are in nuScenes frame throughout
(KittiToNuscenes has already been applied by the val_pipeline).
Comparison uses gravity-centre z, so the nuScenes origin convention
(0.5, 0.5, 0.5) vs bottom-centre (0.5, 0.5, 0) cancels out
through .gravity_center.

Usage (from ~/mmdetection3d, conda activate thesis):
  python tools/analyze_z_h_error.py \\
      configs/mean_teacher/mean_teacher_centerpoint_config.py \\
      work_dirs/mt_centerpoint_top0.3_minPts10_8jun \\
      --pretrained work_dirs/baseline_centerpoint_18may/epoch_20.pth

  # Specific epochs only:
  python tools/analyze_z_h_error.py ... --epochs 1 2 3 4
"""

import argparse
import os
import os.path as osp

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader

import mmdet3d  # noqa — registers all mmdet3d components
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Per-epoch z/h MAE for a Mean Teacher CenterPoint run')
    p.add_argument('config',
                   help='MT training config (mean_teacher_centerpoint_config.py)')
    p.add_argument('work_dir',
                   help='Directory containing epoch_N.pth checkpoints')
    p.add_argument('--epochs', type=int, nargs='+', default=None,
                   help='Epochs to evaluate (default: every epoch_*.pth found)')
    p.add_argument('--pretrained', default=None,
                   help='Pretrained checkpoint to label "baseline" (epoch 0)')
    p.add_argument('--max-dist', type=float, default=2.0,
                   help='Max BEV L2 distance (m) for pred→GT matching (default 2.0)')
    p.add_argument('--score-thr', type=float, default=0.05,
                   help='Min teacher score to keep a prediction (default 0.05)')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=4)
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def build_val_loader(cfg, batch_size, num_workers):
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=pseudo_collate,   # returns list[sample_dict] — handled below
        shuffle=False,
        drop_last=False,
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(cfg, device):
    return MODELS.build(cfg.model).to(device)


def smart_load_checkpoint(model, ckpt_path, device):
    """
    Load a checkpoint into a MeanTeacher3DDetector.

    Handles two checkpoint formats:
      - MT checkpoint  (keys start with 'student.' / 'teacher.')  → direct load
      - Plain model checkpoint (keys start with 'pts_*' etc.)     → remap to
        both student.* and teacher.* before loading

    Always uses strict=False so extra keys (roi_extractor, etc.) are silently
    ignored.
    """
    raw = torch.load(ckpt_path, map_location=device)
    state_dict = raw.get('state_dict', raw)

    first_key = next(iter(state_dict), '')
    is_mt_ckpt = first_key.startswith('student.') or first_key.startswith('teacher.')

    if not is_mt_ckpt:
        # Plain pretrain checkpoint: broadcast weights into both student and teacher
        remapped = {}
        for k, v in state_dict.items():
            remapped[f'student.{k}'] = v
            remapped[f'teacher.{k}'] = v
        state_dict = remapped

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f'  [info] {len(unexpected)} unexpected keys (roi_extractor etc.) — ok')
    # Missing keys after remapping means the checkpoint didn't cover those layers.
    core = [k for k in missing if 'roi_extractor' not in k and 'num_batches_tracked' not in k]
    if core:
        print(f'  [warn] {len(core)} unmatched core keys — checkpoint may be incompatible')


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_teacher_inference(model, loader, score_thr, device):
    """
    Iterate the val loader, run the teacher, and return two parallel lists:
      preds_list[i]  LiDARInstance3DBoxes  teacher predictions for scene i
      gts_list[i]    LiDARInstance3DBoxes  GT boxes for scene i
    Both are in nuScenes frame.
    """
    model.eval()
    preds_list, gts_list = [], []
    n_scenes = len(loader.dataset)
    processed = 0

    for batch in loader:
        # pseudo_collate transposes list[dict] → dict[list], so batch is:
        #   batch['inputs']['points']  = list[LiDARPoints]   (one per scene)
        #   batch['data_samples']      = list[Det3DDataSample]
        batch_data_samples = batch['data_samples']
        batch_inputs       = {'points': batch['inputs']['points']}

        # Capture GT before predict (safe even if predict modifies data_samples in-place)
        batch_gts = [ds.gt_instances_3d.bboxes_3d for ds in batch_data_samples]

        # MeanTeacher3DDetector.predict handles DSNorm switching, eval mode,
        # and subnet.data_preprocessor (voxelization) internally.
        results = model.predict(batch_inputs, batch_data_samples)

        for result, gt_boxes in zip(results, batch_gts):
            pred_boxes = result.pred_instances_3d.bboxes_3d
            scores     = result.pred_instances_3d.scores_3d.cpu()
            keep       = scores >= score_thr
            preds_list.append(pred_boxes[keep])
            gts_list.append(gt_boxes)

        processed += len(batch)
        if processed % 500 == 0 or processed == n_scenes:
            print(f'    {processed}/{n_scenes} scenes', end='\r', flush=True)

    print()
    return preds_list, gts_list


# ── Matching ──────────────────────────────────────────────────────────────────

def match_bev(pred_boxes, gt_boxes, max_dist):
    """
    Greedy nearest-neighbour BEV matching.

    Process GT boxes in ascending order of their minimum BEV distance to any
    prediction.  Assign each GT to its nearest unmatched prediction within
    max_dist.  Returns list of (pred_idx, gt_idx) integer pairs.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []

    gt_xy = gt_boxes.gravity_center[:, :2].cpu().numpy()    # (N_gt,  2)
    pr_xy = pred_boxes.gravity_center[:, :2].cpu().numpy()  # (N_pred, 2)

    dist = np.linalg.norm(gt_xy[:, None] - pr_xy[None], axis=-1)  # (N_gt, N_pred)

    used_preds = set()
    matches    = []
    for gi in np.argsort(dist.min(axis=1)):
        row      = dist[gi].copy()
        row[list(used_preds)] = np.inf
        pi       = int(np.argmin(row))
        if row[pi] <= max_dist:
            matches.append((pi, gi))
            used_preds.add(pi)
    return matches


# ── Error computation ─────────────────────────────────────────────────────────

def collect_errors(preds_list, gts_list, max_dist):
    """
    Match all predictions to GT and aggregate per-pair z and h absolute errors.

    Returns:
      z_errs      np.float32 array of |z_gravity_pred - z_gravity_gt| values
      h_errs      np.float32 array of |h_pred - h_gt| values
      n_matched   total matched pairs
      n_gt        total GT boxes
    """
    z_errs, h_errs = [], []
    n_gt = n_matched = 0

    for preds, gts in zip(preds_list, gts_list):
        n_gt     += len(gts)
        matches   = match_bev(preds, gts, max_dist)
        n_matched += len(matches)
        if not matches:
            continue

        pr_gz = preds.gravity_center[:, 2].cpu().numpy()  # predicted gravity-centre z
        pr_h  = preds.tensor[:, 5].cpu().numpy()           # predicted box height
        gt_gz = gts.gravity_center[:, 2].cpu().numpy()    # GT gravity-centre z
        gt_h  = gts.tensor[:, 5].cpu().numpy()             # GT box height

        for pi, gi in matches:
            z_errs.append(abs(float(pr_gz[pi]) - float(gt_gz[gi])))
            h_errs.append(abs(float(pr_h[pi])  - float(gt_h[gi])))

    return (np.array(z_errs, dtype=np.float32),
            np.array(h_errs, dtype=np.float32),
            n_matched, n_gt)


# ── Reporting helpers ─────────────────────────────────────────────────────────

COL = 12  # label column width

HEADER = (f'{"label":<{COL}}  {"z_MAE":>8}  {"z_p50":>8}  '
          f'{"h_MAE":>8}  {"h_p50":>8}  {"match%":>7}  {"n_match":>8}')
UNITS  = (f'{"":^{COL}}  {"(m)":>8}  {"(m)":>8}  '
          f'{"(m)":>8}  {"(m)":>8}  {"":>7}  {"":>8}')
SEP    = '─' * len(HEADER)


def fmt_row(r):
    return (f'{r["label"]:<{COL}}  {r["z_mae"]:>8.4f}  {r["z_p50"]:>8.4f}  '
            f'{r["h_mae"]:>8.4f}  {r["h_p50"]:>8.4f}  '
            f'{r["match_pct"]:>6.1f}%  {r["n_match"]:>8}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    register_all_modules()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg    = Config.fromfile(args.config)

    # ── Build val loader once; reuse across checkpoints ───────────────────────
    print('Building val dataset …')
    loader = build_val_loader(cfg, args.batch_size, args.num_workers)
    print(f'  {len(loader.dataset)} scenes, {len(loader)} batches '
          f'(batch_size={args.batch_size})')

    # ── Collect (label, path) pairs to evaluate ───────────────────────────────
    ckpts = []
    if args.pretrained:
        ckpts.append(('baseline', args.pretrained))

    epoch_nums = args.epochs
    if epoch_nums is None:
        epoch_nums = sorted(
            int(f[6:-4])
            for f in os.listdir(args.work_dir)
            if f.startswith('epoch_') and f.endswith('.pth')
        )
    for ep in sorted(epoch_nums):
        path = osp.join(args.work_dir, f'epoch_{ep}.pth')
        if osp.exists(path):
            ckpts.append((f'epoch_{ep}', path))
        else:
            print(f'  [skip] {path} not found')

    if not ckpts:
        print('No checkpoints to evaluate. '
              'Check --work-dir path or --pretrained argument.')
        return

    # ── Build model once; reload weights for each checkpoint ─────────────────
    print('\nBuilding model …')
    model = build_model(cfg, device)

    # ── Print table header ────────────────────────────────────────────────────
    print(f'\n{HEADER}\n{UNITS}\n{SEP}')

    results = []

    for label, ckpt_path in ckpts:
        print(f'\n[{label}]  {ckpt_path}')
        smart_load_checkpoint(model, ckpt_path, device)

        preds_list, gts_list = run_teacher_inference(
            model, loader, args.score_thr, device)

        z_e, h_e, n_match, n_gt = collect_errors(
            preds_list, gts_list, args.max_dist)

        if len(z_e) == 0:
            print(f'  ← no matched pairs  '
                  f'(score_thr={args.score_thr}, max_dist={args.max_dist} m)')
            print('    Try lowering --score-thr or raising --max-dist.')
            continue

        r = dict(
            label     = label,
            z_mae     = float(np.mean(z_e)),
            z_p50     = float(np.median(z_e)),
            h_mae     = float(np.mean(h_e)),
            h_p50     = float(np.median(h_e)),
            match_pct = 100.0 * n_match / max(n_gt, 1),
            n_match   = n_match,
        )
        results.append(r)
        print(fmt_row(r))

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(results) < 2:
        return

    print(f'\n{"=" * len(SEP)}')
    print(f'SUMMARY  —  nuScenes frame  ·  gravity-centre z  '
          f'·  BEV match ≤ {args.max_dist} m')
    print(f'{HEADER}\n{UNITS}\n{SEP}')
    for r in results:
        print(fmt_row(r))
    print('=' * len(SEP))

    # ── Drift trend (epoch_1 → epoch_N) ──────────────────────────────────────
    ep_res = [r for r in results if r['label'] != 'baseline']
    if len(ep_res) >= 2:
        first, last = ep_res[0], ep_res[-1]
        dz = last['z_mae'] - first['z_mae']
        dh = last['h_mae'] - first['h_mae']
        dominant = 'z' if abs(dz) > abs(dh) else 'h'
        print(f'\nDrift  {first["label"]} → {last["label"]}:')
        print(f'  z_MAE  {first["z_mae"]:.4f} → {last["z_mae"]:.4f}'
              f'  (Δ = {dz:+.4f} m)')
        print(f'  h_MAE  {first["h_mae"]:.4f} → {last["h_mae"]:.4f}'
              f'  (Δ = {dh:+.4f} m)')
        print(f'\n  → dominant drift: {dominant}  '
              f'({dominant} drifted {max(abs(dz), abs(dh)):.4f} m vs '
              f'{"h" if dominant == "z" else "z"} '
              f'{min(abs(dz), abs(dh)):.4f} m)')

    # ── Decision guidance ─────────────────────────────────────────────────────
    if ep_res:
        base = next((r for r in results if r['label'] == 'baseline'), None)
        ep1  = ep_res[0]
        if base:
            print(f'\nBaseline vs epoch_1:')
            print(f'  z_MAE  {base["z_mae"]:.4f} → {ep1["z_mae"]:.4f}'
                  f'  (Δ = {ep1["z_mae"] - base["z_mae"]:+.4f} m)')
            print(f'  h_MAE  {base["h_mae"]:.4f} → {ep1["h_mae"]:.4f}'
                  f'  (Δ = {ep1["h_mae"] - base["h_mae"]:+.4f} m)')
        print()
        print('Recommendation:')
        print('  If z_MAE grows >> h_MAE:')
        print('    pseudo_code_weights=[1,1,0, 1,1,1, 1,1]  (freeze z only)')
        print('  If both grow proportionally:')
        print('    pseudo_code_weights=[1,1,0, 1,1,0, 1,1]  (freeze z and h)')
        print()


if __name__ == '__main__':
    main()
