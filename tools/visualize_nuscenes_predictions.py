#!/usr/bin/env python3
"""
Visualize MMDetection3D predictions on nuScenes dataset.
Supports both PNG image export and Open3D-compatible PLY export.
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mmcv
try:
    from mmengine.config import Config
except ImportError:
    from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import init_model, inference_detector

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. Only PNG export will work.")


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize 3D detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out-dir', default='./vis_output', help='output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='number of samples to visualize')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'], help='dataset split')
    parser.add_argument('--score-thr', type=float, default=0.3, help='score threshold for visualization')
    parser.add_argument('--format', default='both', choices=['png', 'ply', 'both'], 
                        help='output format: png (2D images), ply (Open3D), or both')
    parser.add_argument('--device', default='cuda:0', help='device used for inference')
    return parser.parse_args()


def create_open3d_bbox(center, size, rotation, color=[1, 0, 0]):
    """Create Open3D bounding box from center, size, and rotation."""
    # Create bbox
    bbox = o3d.geometry.OrientedBoundingBox(center, rotation, size)
    bbox.color = color
    return bbox


def draw_lidar_bbox3d_on_img(points, gt_bboxes, pred_bboxes, scores, 
                              score_thr=0.3, save_path=None):
    """
    Draw 3D bounding boxes on BEV (Bird's Eye View) image.
    
    Args:
        points: (N, 4) point cloud [x, y, z, intensity]
        gt_bboxes: (M, 7) ground truth boxes [x, y, z, dx, dy, dz, rot]
        pred_bboxes: (K, 7) predicted boxes
        scores: (K,) prediction scores
        score_thr: score threshold for filtering predictions
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    
    # Draw point cloud (BEV)
    ax.scatter(points[:, 0], points[:, 1], s=0.1, c=points[:, 2], 
               cmap='viridis', alpha=0.5, label='Point Cloud')
    
    # Helper function to draw box corners
    def draw_box_bev(box, color, label, linestyle='-'):
        """Draw box in BEV."""
        x, y, z, dx, dy, dz, rot = box
        
        # Get corners in local coordinate
        corners = np.array([
            [-dx/2, -dy/2],
            [dx/2, -dy/2],
            [dx/2, dy/2],
            [-dx/2, dy/2],
            [-dx/2, -dy/2],  # close the box
        ])
        
        # Rotation matrix
        rot_mat = np.array([
            [np.cos(rot), -np.sin(rot)],
            [np.sin(rot), np.cos(rot)]
        ])
        
        # Rotate and translate
        corners_rot = corners @ rot_mat.T
        corners_global = corners_rot + np.array([x, y])
        
        ax.plot(corners_global[:, 0], corners_global[:, 1], 
                color=color, linewidth=2, label=label, linestyle=linestyle)
    
    # Draw ground truth boxes
    if gt_bboxes is not None and len(gt_bboxes) > 0:
        for i, box in enumerate(gt_bboxes):
            draw_box_bev(box, 'green', 'GT' if i == 0 else '', linestyle='--')
    
    # Draw predicted boxes (filter by score)
    if pred_bboxes is not None and len(pred_bboxes) > 0:
        mask = scores >= score_thr
        filtered_boxes = pred_bboxes[mask]
        filtered_scores = scores[mask]
        
        for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
            label = f'Pred (s={score:.2f})' if i == 0 else ''
            draw_box_bev(box, 'red', label, linestyle='-')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('3D Detection Results (Bird\'s Eye View)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved BEV image to {save_path}")
    plt.close()


def save_open3d_ply(points, gt_bboxes, pred_bboxes, scores, 
                     score_thr=0.3, save_path=None):
    """
    Save point cloud and bounding boxes in PLY format for Open3D visualization.
    
    Args:
        points: (N, 4) point cloud [x, y, z, intensity]
        gt_bboxes: (M, 7) ground truth boxes
        pred_bboxes: (K, 7) predicted boxes
        scores: (K,) prediction scores
        save_path: output path (without extension)
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available, skipping PLY export")
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color by height (z-coordinate) normalized
    z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-6)
    colors = plt.cm.viridis(z_norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create list of geometries
    geometries = [pcd]
    
    # Add ground truth bounding boxes (green)
    if gt_bboxes is not None and len(gt_bboxes) > 0:
        for box in gt_bboxes:
            x, y, z, dx, dy, dz, rot = box
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, rot])
            bbox = o3d.geometry.OrientedBoundingBox([x, y, z], R, [dx, dy, dz])
            bbox.color = [0, 1, 0]  # Green for GT
            geometries.append(bbox)
    
    # Add predicted bounding boxes (red), filtered by score
    if pred_bboxes is not None and len(pred_bboxes) > 0:
        mask = scores >= score_thr
        filtered_boxes = pred_bboxes[mask]
        
        for box in filtered_boxes:
            x, y, z, dx, dy, dz, rot = box
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, rot])
            bbox = o3d.geometry.OrientedBoundingBox([x, y, z], R, [dx, dy, dz])
            bbox.color = [1, 0, 0]  # Red for predictions
            geometries.append(bbox)
    
    # Save as PLY
    if save_path:
        # Save point cloud
        o3d.io.write_point_cloud(f"{save_path}_points.ply", pcd)
        print(f"Saved point cloud to {save_path}_points.ply")
        
        # Create a combined scene
        # Note: We can't directly save OrientedBoundingBox, so we convert to LineSet
        combined = o3d.geometry.PointCloud()
        combined.points = pcd.points
        combined.colors = pcd.colors
        
        for geom in geometries[1:]:  # Skip the point cloud itself
            if isinstance(geom, o3d.geometry.OrientedBoundingBox):
                lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(geom)
                lineset.paint_uniform_color(geom.color)
                # We'll save the full scene visualization info
        
        # Save visualization script
        vis_script = f"""
import open3d as o3d

# Load point cloud
pcd = o3d.io.read_point_cloud("{save_path}_points.ply")

# Visualize
o3d.visualization.draw_geometries([pcd],
                                   window_name="3D Detection Results",
                                   width=1024, height=768,
                                   point_show_normal=False)
"""
        with open(f"{save_path}_visualize.py", 'w') as f:
            f.write(vis_script)
        print(f"Saved visualization script to {save_path}_visualize.py")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Build dataset
    if args.split == 'train':
        dataset = build_dataset(cfg.data.train)
    elif args.split == 'val':
        dataset = build_dataset(cfg.data.val)
    else:
        dataset = build_dataset(cfg.data.test)
    
    # Initialize model
    print(f"Loading model from {args.checkpoint}...")
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    # Sample indices
    num_samples = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    print(f"Visualizing {num_samples} samples from {args.split} set...")
    
    for idx_num, idx in enumerate(indices):
        print(f"\nProcessing sample {idx_num+1}/{num_samples} (dataset index {idx})...")
        
        # Get data
        data = dataset[idx]
        
        # Run inference
        result = inference_detector(model, data)
        
        # Extract points - handle different data formats
        if hasattr(data, 'get'):
            points = data.get('points', data.get('inputs', {}).get('points'))
        else:
            points = data['points']
        
        if hasattr(points, '_data'):
            points = points._data.numpy()
        elif hasattr(points, 'numpy'):
            points = points.numpy()
        elif isinstance(points, dict):
            points = points['points'].numpy()
        else:
            points = points
        
        # Extract ground truth boxes (if available)
        gt_bboxes = None
        if hasattr(data, 'get'):
            gt_data = data.get('gt_bboxes_3d', data.get('data_samples', {}).get('gt_instances_3d', {}).get('bboxes_3d'))
        else:
            gt_data = data.get('gt_bboxes_3d')
        
        if gt_data is not None:
            if hasattr(gt_data, '_data'):
                gt_bboxes = gt_data._data.tensor.numpy()
            elif hasattr(gt_data, 'tensor'):
                gt_bboxes = gt_data.tensor.numpy()
            else:
                gt_bboxes = gt_data
        
        # Extract predictions - handle different result formats
        if isinstance(result, dict):
            if 'boxes_3d' in result:
                pred_bboxes = result['boxes_3d'].tensor.numpy()
                scores = result['scores_3d'].numpy()
                labels = result['labels_3d'].numpy()
            elif 'pts_bbox' in result:
                pred_bboxes = result['pts_bbox']['boxes_3d'].tensor.numpy()
                scores = result['pts_bbox']['scores_3d'].numpy()
                labels = result['pts_bbox']['labels_3d'].numpy()
        else:
            # Handle tuple or list results
            pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
            scores = result[0]['scores_3d'].numpy()
            labels = result[0]['labels_3d'].numpy()
        
        # Create output filename
        sample_name = f"sample_{idx:04d}"
        
        # Save PNG visualization
        if args.format in ['png', 'both']:
            png_path = os.path.join(args.out_dir, f"{sample_name}.png")
            draw_lidar_bbox3d_on_img(points, gt_bboxes, pred_bboxes, scores, 
                                     args.score_thr, png_path)
        
        # Save PLY for Open3D
        if args.format in ['ply', 'both']:
            ply_path = os.path.join(args.out_dir, sample_name)
            save_open3d_ply(points, gt_bboxes, pred_bboxes, scores, 
                           args.score_thr, ply_path)
        
        print(f"  - GT boxes: {len(gt_bboxes) if gt_bboxes is not None else 0}")
        print(f"  - Predicted boxes: {len(pred_bboxes)} (filtered: {(scores >= args.score_thr).sum()})")
    
    print(f"\nDone! Results saved to {args.out_dir}")
    print(f"\nTo visualize PLY files locally with Open3D:")
    print(f"  python {args.out_dir}/sample_XXXX_visualize.py")


if __name__ == '__main__':
    main()