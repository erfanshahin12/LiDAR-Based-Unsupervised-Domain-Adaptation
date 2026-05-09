#!/usr/bin/env python3
"""Visualize predicted 3D boxes on a few validation/test scenes.

Usage:
    python tools/misc/visualize_predictions.py CONFIG CHECKPOINT [options]

Saves one Open3D .ply file per scene to --out-dir (default: <ckpt_dir>/visualizations/).

Colour legend
  Point cloud : grey
  Predictions : Car=vivid blue  Pedestrian=vivid orange  Cyclist=vivid green
  Ground truth: Car=light blue  Pedestrian=light orange  Cyclist=light green
"""

import argparse
from os import path as osp

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.utils import mkdir_or_exist
from mmengine.visualization.utils import tensor2ndarray

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes

MAX_SCENES = 10

# ---------------------------------------------------------------------------
# Per-class colours — predictions use vivid tones, GT boxes use muted tones.
# Index matches: 0=Car  1=Pedestrian  2=Cyclist
# ---------------------------------------------------------------------------
_PRED_COLORS = {
    0: np.array([0.15, 0.45, 1.00]),   # Car        — vivid blue
    1: np.array([1.00, 0.35, 0.00]),   # Pedestrian — vivid orange
    2: np.array([0.00, 0.85, 0.25]),   # Cyclist    — vivid green
}
_GT_COLORS = {
    0: np.array([0.60, 0.80, 1.00]),   # Car GT        — light blue
    1: np.array([1.00, 0.75, 0.55]),   # Pedestrian GT — light orange
    2: np.array([0.65, 1.00, 0.70]),   # Cyclist GT    — light green
}
_FALLBACK_PRED = np.array([1.00, 1.00, 1.00])  # white  (unknown label)
_FALLBACK_GT   = np.array([0.80, 0.80, 0.80])  # light grey


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize model predictions on point cloud scenes.')
    parser.add_argument('config', help='Config file (.py)')
    parser.add_argument('checkpoint', help='Checkpoint file (.pth)')
    parser.add_argument(
        '--num-scenes', type=int, default=5,
        help=f'How many scenes to visualise (default 5, hard cap {MAX_SCENES})')
    parser.add_argument(
        '--indices', type=str, default=None,
        help='Comma-separated dataset indices (e.g. "0,17,42"). '
             'Overrides --num-scenes.')
    parser.add_argument(
        '--split', choices=['val', 'test'], default='val',
        help='Which dataloader split to pull scenes from (default: val)')
    parser.add_argument(
        '--score-thr', type=float, default=0.3,
        help='Minimum confidence for predicted boxes (default: 0.3)')
    parser.add_argument(
        '--out-dir', type=str, default=None,
        help='Output directory.  Defaults to <checkpoint_dir>/visualizations/')
    parser.add_argument(
        '--show', action='store_true',
        help='Open interactive Open3D window (requires $DISPLAY)')
    parser.add_argument(
        '--no-gt', action='store_true',
        help='Skip drawing ground-truth boxes')
    parser.add_argument(
        '--filter-gt-range', action='store_true',
        help='Remove GT boxes whose gravity centres are outside the model '
             'point_cloud_range (suppresses floating-box artefacts caused by '
             'the val pipeline having no ObjectRangeFilter)')
    parser.add_argument(
        '--use-teacher', action='store_true',
        help='MeanTeacher3DDetector: infer with teacher instead of student')
    parser.add_argument(
        '--device', default=None,
        help='Inference device, e.g. "cuda:0" or "cpu". '
             'Defaults to "cuda:0" if a GPU is available, otherwise "cpu".')
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU inference (equivalent to --device cpu)')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Override config values in key=value format')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline augmentation helpers
# ---------------------------------------------------------------------------

def _inject_gt_pipeline(data_cfg, cfg):
    """Return data_cfg with LoadAnnotations3D spliced into the pipeline.

    If the pipeline already loads annotations, or if we cannot locate a
    LoadAnnotations3D transform in the train pipeline, the original cfg is
    returned unchanged.
    """
    pipeline = list(data_cfg.get('pipeline', []))
    if any(t.get('type', '') == 'LoadAnnotations3D' for t in pipeline):
        return data_cfg  # already present

    train_pipeline = list(cfg.get('train_pipeline', []))
    if not train_pipeline:
        train_pipeline = list(
            cfg.get('train_dataloader', {}).get('dataset', {}).get('pipeline', []))

    ann_transform = None
    pack_transform = None
    for t in train_pipeline:
        if t.get('type') == 'LoadAnnotations3D':
            ann_transform = t
        if t.get('type') == 'Pack3DDetInputs':
            pack_transform = t

    if ann_transform is None:
        return data_cfg

    new_pipeline = []
    for t in pipeline:
        if t.get('type') == 'Pack3DDetInputs':
            new_pipeline.append(ann_transform)
            new_pipeline.append(pack_transform if pack_transform is not None else t)
        else:
            new_pipeline.append(t)

    data_cfg = dict(data_cfg)
    data_cfg['pipeline'] = new_pipeline
    return data_cfg


def _unwrap_dataset_cfg(data_cfg):
    """Strip RepeatDataset / ConcatDataset / CBGSDataset wrappers."""
    while True:
        dtype = data_cfg.get('type', '')
        if dtype == 'RepeatDataset':
            data_cfg = data_cfg['dataset']
        elif dtype == 'ConcatDataset':
            data_cfg = data_cfg['datasets'][0]
        elif dtype == 'CBGSDataset':
            data_cfg = data_cfg['dataset']
        else:
            break
    return data_cfg


def build_dataset(cfg, split, draw_gt):
    if split == 'val':
        data_cfg = cfg.val_dataloader.dataset
    else:
        data_cfg = cfg.test_dataloader.dataset

    data_cfg = _unwrap_dataset_cfg(data_cfg)

    if draw_gt:
        data_cfg = _inject_gt_pipeline(data_cfg, cfg)

    try:
        return DATASETS.build(data_cfg, default_args=dict(filter_empty_gt=False))
    except TypeError:
        return DATASETS.build(data_cfg)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _to_lidar_corners(bboxes_3d) -> np.ndarray:
    """Return (N, 8, 3) box corners in the LIDAR/sensor frame.

    Points loaded from disk are always in LIDAR coordinates, so we keep the
    corners in the same frame — no coordinate-system conversion needed.
    Any non-LIDAR box type is converted to LIDAR first.
    """
    if not isinstance(bboxes_3d, LiDARInstance3DBoxes):
        bboxes_3d = bboxes_3d.convert_to(Box3DMode.LIDAR)
    return bboxes_3d.corners.numpy()  # (N, 8, 3) in LIDAR coords


# 12 edges of a box as pairs of corner indices (Open3D convention)
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # vertical pillars
]


def _add_wireframe(bboxes_3d, labels, colors_map, fallback_color,
                   all_pts, all_cols, n_samples=15):
    """Sample n_samples points per edge; append to all_pts / all_cols."""
    corners = _to_lidar_corners(bboxes_3d)  # (N, 8, 3)
    labels_np = (labels.numpy()
                 if hasattr(labels, 'numpy') else np.asarray(labels))

    pts_acc, col_acc = [], []
    for i, box_corners in enumerate(corners):
        color = colors_map.get(int(labels_np[i]), fallback_color)
        for a, b in _BOX_EDGES:
            for t in np.linspace(0, 1, n_samples):
                pts_acc.append(box_corners[a] * (1 - t) + box_corners[b] * t)
                col_acc.append(color)

    if pts_acc:
        all_pts.append(np.array(pts_acc, dtype=np.float64))
        all_cols.append(np.array(col_acc, dtype=np.float64))


# ---------------------------------------------------------------------------
# Per-scene debug output
# ---------------------------------------------------------------------------

def _print_debug(pts, pred_sample, score_thr, pcl_range):
    xmin, ymin, zmin, xmax, ymax, zmax = pcl_range
    print(f'    PCL : {len(pts):6d} pts  '
          f'x[{pts[:, 0].min():6.1f}, {pts[:, 0].max():6.1f}]  '
          f'y[{pts[:, 1].min():6.1f}, {pts[:, 1].max():6.1f}]  '
          f'z[{pts[:, 2].min():5.1f}, {pts[:, 2].max():5.1f}]')

    if hasattr(pred_sample, 'pred_instances_3d'):
        pred = pred_sample.pred_instances_3d
        mask = pred.scores_3d >= score_thr
        n = int(mask.sum())
        if n > 0:
            gc = pred.bboxes_3d[mask].gravity_center.numpy()
            out = ~((gc[:, 0] >= xmin) & (gc[:, 0] <= xmax) &
                    (gc[:, 1] >= ymin) & (gc[:, 1] <= ymax))
            lbls = pred.labels_3d[mask].numpy()
            by_cls = {k: int((lbls == k).sum()) for k in range(3)}
            print(f'    Pred: {n:4d} boxes '
                  f'(Car={by_cls[0]}, Ped={by_cls[1]}, Cyc={by_cls[2]})  '
                  f'{int(out.sum())} outside PCL range')

    if hasattr(pred_sample, 'gt_instances_3d'):
        gt = pred_sample.gt_instances_3d
        if hasattr(gt, 'bboxes_3d') and len(gt.bboxes_3d) > 0:
            gc = gt.bboxes_3d.gravity_center.numpy()
            out = ~((gc[:, 0] >= xmin) & (gc[:, 0] <= xmax) &
                    (gc[:, 1] >= ymin) & (gc[:, 1] <= ymax))
            lbls = gt.labels_3d.numpy()
            by_cls = {k: int((lbls == k).sum()) for k in range(3)}
            print(f'    GT  : {len(gt.bboxes_3d):4d} boxes '
                  f'(Car={by_cls[0]}, Ped={by_cls[1]}, Cyc={by_cls[2]})  '
                  f'{int(out.sum())} outside PCL range')


# ---------------------------------------------------------------------------
# Headless PLY writer
# ---------------------------------------------------------------------------

def _save_scene_ply(data_input, pred_sample, o3d_path,
                    score_thr, draw_gt, pcl_range=None,
                    filter_gt_range=False, show=False):
    """Write point cloud + per-class 3D box wireframes to a .ply file.

    All geometry is kept in the LIDAR/sensor coordinate frame — the same
    frame the raw points arrive in — so there is no axis-swap artefact.
    """
    import open3d as o3d

    raw_pts = data_input.get('points', None)
    if raw_pts is None:
        print('(no lidar points in data_input — skipping)')
        return
    pts = tensor2ndarray(raw_pts).astype(np.float64)[:, :3]

    if pcl_range is not None:
        _print_debug(pts, pred_sample, score_thr, pcl_range)

    all_pts = [pts]
    all_cols = [np.full((len(pts), 3), 0.6)]  # grey point cloud

    # --- Predicted boxes: per-class vivid colours ---
    if hasattr(pred_sample, 'pred_instances_3d'):
        pred = pred_sample.pred_instances_3d
        mask = pred.scores_3d >= score_thr
        if mask.sum() > 0:
            _add_wireframe(
                pred.bboxes_3d[mask], pred.labels_3d[mask],
                _PRED_COLORS, _FALLBACK_PRED, all_pts, all_cols)

    # --- GT boxes: per-class muted colours ---
    if draw_gt and hasattr(pred_sample, 'gt_instances_3d'):
        gt = pred_sample.gt_instances_3d
        if hasattr(gt, 'bboxes_3d') and len(gt.bboxes_3d) > 0:
            bboxes = gt.bboxes_3d
            labels = gt.labels_3d
            if filter_gt_range and pcl_range is not None:
                xmin, ymin, zmin, xmax, ymax, zmax = pcl_range
                gc = bboxes.gravity_center.numpy()
                in_r = ((gc[:, 0] >= xmin) & (gc[:, 0] <= xmax) &
                        (gc[:, 1] >= ymin) & (gc[:, 1] <= ymax) &
                        (gc[:, 2] >= zmin) & (gc[:, 2] <= zmax))
                mask_t = torch.from_numpy(in_r)
                bboxes = bboxes[mask_t]
                labels = labels[mask_t]
            if len(bboxes) > 0:
                _add_wireframe(
                    bboxes, labels,
                    _GT_COLORS, _FALLBACK_GT, all_pts, all_cols)

    combined_pts = np.concatenate(all_pts, axis=0)
    combined_cols = np.clip(np.concatenate(all_cols, axis=0), 0.0, 1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pts)
    pcd.colors = o3d.utility.Vector3dVector(combined_cols)
    o3d.io.write_point_cloud(o3d_path, pcd)

    if show:
        o3d.visualization.draw_geometries(
            [pcd], window_name=osp.basename(o3d_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.cpu:
        args.device = 'cpu'
    elif args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {args.device}')

    if args.out_dir is None:
        args.out_dir = osp.join(
            osp.dirname(osp.abspath(args.checkpoint)), 'visualizations')
    mkdir_or_exist(args.out_dir)

    num_scenes = min(args.num_scenes, MAX_SCENES)

    print(f'Loading model from {args.checkpoint} ...')
    model = init_model(
        args.config, args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options)
    model.eval()

    try:
        from mmdet3d.models.detectors.mean_teacher_detector import (
            MeanTeacher3DDetector)
        if args.use_teacher and isinstance(model, MeanTeacher3DDetector):
            print('Swapping to teacher submodule for inference.')
            teacher = model.teacher
            teacher.dataset_meta = model.dataset_meta
            model = teacher
            model.eval()
        elif args.use_teacher:
            print('Warning: --use-teacher given but model is not '
                  'MeanTeacher3DDetector — flag ignored.')
    except ImportError:
        if args.use_teacher:
            print('Warning: could not import MeanTeacher3DDetector.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    # Extract point_cloud_range for debug output and optional GT filtering
    pcl_range = cfg.get('point_cloud_range', None)
    if pcl_range is None:
        # fall back to data_preprocessor voxel_layer
        try:
            pcl_range = (cfg.model.data_preprocessor
                         .voxel_layer.point_cloud_range)
        except (AttributeError, KeyError):
            pass
    if pcl_range is not None:
        pcl_range = list(pcl_range)
        print(f'Point cloud range: {pcl_range}')

    draw_gt = not args.no_gt
    print(f'Building {args.split} dataset ...')
    dataset = build_dataset(cfg, args.split, draw_gt)
    print(f'Dataset size: {len(dataset)} samples.')

    class_names = model.dataset_meta.get('classes', [])
    print(f'Classes: {class_names}')
    print('Colour legend:')
    print('  Predictions — Car: vivid blue  Pedestrian: vivid orange  '
          'Cyclist: vivid green')
    print('  Ground truth — Car: light blue  Pedestrian: light orange  '
          'Cyclist: light green')

    if args.indices is not None:
        indices = [int(x.strip()) for x in args.indices.split(',')]
        indices = indices[:MAX_SCENES]
    else:
        indices = list(range(min(num_scenes, len(dataset))))

    saved = []
    for step, idx in enumerate(indices):
        if idx >= len(dataset):
            print(f'  Index {idx} out of range (dataset has {len(dataset)} '
                  'samples) — skipping.')
            continue

        print(f'  [{step + 1}/{len(indices)}] scene {idx} ...', flush=True)
        item = dataset[idx]
        data_input = item['inputs']
        gt_sample = item['data_samples']

        with torch.no_grad():
            results = model.test_step(pseudo_collate([item]))
        pred_sample = results[0]

        if draw_gt and hasattr(gt_sample, 'gt_instances_3d'):
            pred_sample.gt_instances_3d = gt_sample.gt_instances_3d

        scene_name = f'scene_{idx:04d}'
        o3d_path = osp.join(args.out_dir, f'{scene_name}.ply')

        _save_scene_ply(
            data_input, pred_sample, o3d_path,
            score_thr=args.score_thr,
            draw_gt=draw_gt,
            pcl_range=pcl_range,
            filter_gt_range=args.filter_gt_range,
            show=args.show)
        saved.append(o3d_path)
        print(f'    -> saved {o3d_path}')

    print(f'\nSaved {len(saved)} scene(s) to: {args.out_dir}')


if __name__ == '__main__':
    main()
