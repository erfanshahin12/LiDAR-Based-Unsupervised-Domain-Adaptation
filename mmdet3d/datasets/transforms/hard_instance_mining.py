"""
Hard Instance Mining for Domain Adaptive 3D Object Detection

This module implements hard instance mining by:
1. Building a bank of "hard" source instances (objects with few points)
2. Injecting these into target domain scenes during training
3. Enriching target distribution with challenging source examples
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import torch
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmcv.transforms import BaseTransform

class HardInstanceBank:
    """
    Bank for storing and sampling hard instances from source domain.
    Hard instances are defined as objects with fewer points (more challenging).
    """
    def __init__(self,
                 source_db_infos: Dict[str, List],
                 quantile_threshold: float = 50.0,
                 classes: List[str] = None,
                 target_db_infos: Optional[Dict[str, List]] = None):
        """
        Initialize hard instance bank.
        Args:
            source_db_infos: Source domain database (from create_data.py)
            quantile_threshold: Percentile threshold γ (0-100)
            classes: List of class names to consider
            target_db_infos: Optional target domain db for adaptive thresholding
        """
        self.quantile_threshold = quantile_threshold
        self.classes = classes or list(source_db_infos.keys())
        
        # Build hard instance bank
        self.hard_instances = {}
        
        for cls_name in self.classes:
            if cls_name not in source_db_infos:
                print(f"Warning: {cls_name} not found in source database")
                continue
            
            source_samples = source_db_infos[cls_name]
            
            if len(source_samples) == 0:
                self.hard_instances[cls_name] = []
                continue
            
            # Calculate point count threshold
            # Adaptive: use target domain quantile, if available
            if target_db_infos and cls_name in target_db_infos:
                threshold = self._calculate_adaptive_threshold(
                    source_samples, target_db_infos[cls_name], quantile_threshold
                )
            else:
                # Static: use source domain quantile
                threshold = self._calculate_static_threshold(
                    source_samples, quantile_threshold
                )
            
            # Select hard instances: |b_i| < γ
            hard_samples = [
                sample for sample in source_samples
                if sample['num_points_in_gt'] < threshold
            ]
            
            self.hard_instances[cls_name] = hard_samples
            
            print(f"Hard Instance Bank - {cls_name}: "
                  f"{len(hard_samples)}/{len(source_samples)} samples "
                  f"(threshold: {threshold:.1f} points)")
    
    def _calculate_static_threshold(self, samples: List[Dict], 
                                   quantile: float) -> float:
        """Calculate threshold based on source domain quantile """
        
        point_counts = [s['num_points_in_gt'] for s in samples]
        return np.percentile(point_counts, quantile)
    
    def _calculate_adaptive_threshold(self, source_samples: List[Dict],
                                     target_samples: List[Dict],
                                     quantile: float) -> float:
        """ Calculate threshold based on target domain statistics """

        target_point_counts = [s['num_points_in_gt'] for s in target_samples]
        return np.percentile(target_point_counts, quantile)
    
    def sample(self, cls_name: str, num_samples: int = 1) -> List[Dict]:
        """
        Sample hard instances for a given class.
        Args:
            cls_name: Class name
            num_samples: Number of samples to draw
        
        Returns:
            List of sampled instance dictionaries
        """
        if cls_name not in self.hard_instances:
            return []
        
        available = self.hard_instances[cls_name]
        if len(available) == 0:
            return []
        
        # Random sampling with replacement
        num_samples = min(num_samples, len(available))
        indices = np.random.choice(len(available), num_samples, replace=False)
        return [available[i] for i in indices]
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics about the hard instance bank."""
        stats = {}
        for cls_name, samples in self.hard_instances.items():
            if len(samples) > 0:
                point_counts = [s['num_points_in_gt'] for s in samples]
                stats[cls_name] = {
                    'count': len(samples),
                    'mean_points': np.mean(point_counts),
                    'median_points': np.median(point_counts),
                    'min_points': np.min(point_counts),
                    'max_points': np.max(point_counts)
                }
        return stats

@TRANSFORMS.register_module()
class HardInstanceSampling(BaseTransform):
    """
    Transform to inject hard instances from source domain into target domain scenes
    to enrich the target domain distribution with challenging source objects.
    Works with pseudo-labels or GT labels for collision detection.
    """
    
    def __init__(self,
                 hard_instance_bank: HardInstanceBank,
                 sample_groups: Dict[str, int],
                 use_pred_boxes_for_collision: bool = True,
                 iou_thresh: float = 0.3,
                 points_loader: Optional[Dict] = None,
                 class_names: List[str] = ['Car', 'Pedestrian', 'Cyclist']):
        """
        Initialize hard instance sampling.
        
        Args:
            hard_instance_bank: Pre-built hard instance bank
            sample_groups: Dict mapping class names to num samples
                Example: {'Car': 5, 'Pedestrian': 3, 'Cyclist': 3}
            use_pred_boxes_for_collision: Use predictions for collision detection
                (needed for unlabeled target data)
            iou_thresh: IoU threshold for collision detection
            points_loader: Config for loading points from database
        """
        self.hard_instance_bank = hard_instance_bank
        self.sample_groups = sample_groups
        self.use_pred_boxes_for_collision = use_pred_boxes_for_collision
        self.iou_thresh = iou_thresh
        self.points_loader = points_loader
        self.class_names = class_names

        #  Build the actual loader transform
        if points_loader:
            self.points_loader_transform = TRANSFORMS.build(points_loader)
        else:
            self.points_loader_transform = None        
    
    def transform(self, results: Dict) -> Dict:
        """
        Apply hard instance sampling to the input data.
        Args:
            results: Dict containing:
                - 'points': Point cloud
                - 'gt_bboxes_3d': Ground truth boxes (if available)
                - 'gt_labels_3d': Ground truth labels (if available)
                - 'pred_bboxes_3d': Predicted boxes (for collision detection)
                - 'pred_labels_3d': Predicted labels
        
        Returns:
            Modified results with injected hard instances
        """
        # Get existing boxes for collision detection
        if self.use_pred_boxes_for_collision and 'pred_bboxes_3d' in results:
            existing_boxes = results['pred_bboxes_3d']

        elif 'gt_bboxes_3d' in results:
            existing_boxes = results['gt_bboxes_3d']

        else:
            # No boxes to check collision with - can freely add instances
            existing_boxes = None
        
        # Sample and inject hard instances
        sampled_instances = []
        sampled_boxes = []
        sampled_labels = []
        sampled_points_list = []
        
        for cls_name, num_samples in self.sample_groups.items():
            # Sample from hard instance bank
            samples = self.hard_instance_bank.sample(cls_name, num_samples)
            
            for sample in samples:
                # Load points for this instance
                instance_box = self._get_instance_box(sample)
                instance_label = self._get_instance_label(sample, cls_name)
                instance_points = self._load_instance_points(sample)
                
                # Check collision with existing boxes
                if existing_boxes is not None:
                    if self._has_collision(instance_box, existing_boxes):
                        continue  # Skip this instance
                
                # Accept this instance
                sampled_instances.append(sample)
                sampled_boxes.append(instance_box)
                sampled_labels.append(instance_label)
                sampled_points_list.append(instance_points)
        
        # Merge sampled instances into the scene
        if len(sampled_instances) > 0:
            results = self._merge_instances(
                results, sampled_boxes, sampled_labels, sampled_points_list
            )
        
        return results
    
    def _load_instance_points(self, sample: Dict) -> np.ndarray:
        """Load point cloud for an instance from database using the configured loader."""

        if self.points_loader_transform:        # Use the configured loader
        # Create a minimal results dict for the transform
            results = {
            'lidar_path': sample['path'],
            'num_pts_feats': self.points_loader['use_dim']}
        
        # Apply the LoadPointsFromFile transform
            results = self.points_loader_transform(results)
            return results['points']
        
        else:
            # Fallback: direct loading from file
            db_path = Path(sample['path'])
            points = np.fromfile(db_path, dtype=np.float32)
            
            # Reshape based on point dimension
            if 'num_points_in_gt' in sample:
                num_points = sample['num_points_in_gt']
                point_dim = len(points) // num_points
                points = points.reshape(-1, point_dim)
            else:
                # Assume 4D points (x, y, z, intensity)
                points = points.reshape(-1, 4)
            
            return points
    
    def _transform_points_to_global(self, instance_points, box):
        """
        Transform instance points from object-local to global coordinates.
        Used in merging instances into the scene.
        Args:
            instance_points: [N, 4] array in object-local coordinates
            box: [7] array [x, y, z, l, w, h, yaw]
        
        Returns:
            global_points: [N, 4] array in global coordinates
        """
        # Extract box parameters
        x, y, z, l, w, h, yaw = box
        
        # Separate xyz and intensity
        local_xyz = instance_points[:, :3]  # [N, 3]
        intensity = instance_points[:, 3:]   # [N, 1]
        
        # Create rotation matrix around z-axis
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [   0,              0,      1]
        ])
        
        # Apply rotation
        rotated_xyz = local_xyz @ rotation_matrix.T  # [N, 3]
        
        # Apply translation (move to box center)
        translation = np.array([x, y, z])
        global_xyz = rotated_xyz + translation  # [N, 3]
        
        # Combine xyz with intensity
        global_points = np.concatenate([global_xyz, intensity], axis=1)
        
        return global_points

    def _get_instance_box(self, sample: Dict) -> np.ndarray:
        """Extract bounding box from sample."""
        box = sample['box3d_lidar']
        return np.array(box).reshape(1, -1)  # [1, 7]
    
    def _get_instance_label(self, sample: Dict, cls_name: str) -> int:
        """Get label index for the instance."""
        # Map class name to label index
        # This should match your dataset's class mapping
        if 'label' in sample:
            return sample['label']
        else:
        # Use provided class names
            if cls_name in self.class_names:
                return self.class_names.index(cls_name)
            else:
                print(f"Warning: {cls_name} not in class_names {self.class_names}, using 0")
                return 0
    
    def _has_collision(self, new_box: np.ndarray, 
                      existing_boxes: LiDARInstance3DBoxes) -> bool:
        """
        Check if new box collides with existing boxes.
        
        Args:
            new_box: [1, 7] array
            existing_boxes: LiDARInstance3DBoxes object
        
        Returns:
            True if collision detected
        """
        if len(existing_boxes) == 0:
            return False
        
        # Use the same origin as existing boxes for consistency
        origin = getattr(existing_boxes, 'origin', (0.5, 0.5, 0.5))
    
        # Convert to same format for IoU calculation
        new_box_obj = LiDARInstance3DBoxes(
            torch.from_numpy(new_box).float(),
            box_dim=new_box.shape[-1],
            origin=origin
        )
        
        # Calculate IoU with existing boxes
        ious = LiDARInstance3DBoxes.overlaps(new_box_obj, existing_boxes, mode='iou')
        
        # Check if any IoU exceeds threshold
        return (ious > self.iou_thresh).any().item()
    
    def _merge_instances(self, results: Dict,
                        new_boxes: List[np.ndarray],
                        new_labels: List[int],
                        new_points_list: List[np.ndarray]) -> Dict:
        """
        Merge sampled instances into the scene.
        
        This adds:
        1. Instance points to the global point cloud
        2. Instance boxes to gt_bboxes_3d
        3. Instance labels to gt_labels_3d
        """
        # Concatenate new boxes
        new_boxes = np.concatenate(new_boxes, axis=0)  # [N, 7]
        new_labels = np.array(new_labels)  # [N]
        
        # Merge points
        for i, instance_points in enumerate(new_points_list):
            # Transform from object-local to global coordinates
            box = new_boxes[i]  # [7]
            global_points = self._transform_points_to_global(instance_points, box)
            
            # Concatenate to scene points, account for tensor/ndarray types
            if isinstance(results['points'], torch.Tensor):

                current_points = results['points'].numpy()
                merged_points = np.concatenate([current_points, global_points], axis=0)
                results['points'] = torch.from_numpy(merged_points).float()

            else:
                results['points'] = np.concatenate(
                    [results['points'], global_points], axis=0)
        
        # Determine origin from existing boxes or use nuScenes default
        if 'gt_bboxes_3d' in results and hasattr(results['gt_bboxes_3d'], 'origin'):
            origin = results['gt_bboxes_3d'].origin
        else:
            # After KittiToNuscenes transform, should be nuScenes origin
            origin = (0.5, 0.5, 0.5)

        # Merge boxes and labels       
        new_boxes_obj = LiDARInstance3DBoxes(
            torch.from_numpy(new_boxes).float(),
            box_dim=new_boxes.shape[-1],
            origin=origin
        )
        
        if 'gt_bboxes_3d' in results:
            # Concatenate with existing GT boxes
            results['gt_bboxes_3d'] = results['gt_bboxes_3d'].cat(
                [results['gt_bboxes_3d'], new_boxes_obj])
            
            # Handle both numpy and tensor types (avoid potential mismatch)
            if isinstance(results['gt_labels_3d'], torch.Tensor):
                results['gt_labels_3d'] = torch.cat([
                    results['gt_labels_3d'],
                    torch.from_numpy(new_labels).long()], dim=0)

            else:
                results['gt_labels_3d'] = np.concatenate(
                    [results['gt_labels_3d'], new_labels], axis=0)
                
        else:
            # No existing GT - create new
            results['gt_bboxes_3d'] = new_boxes_obj
            results['gt_labels_3d'] = new_labels
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(sample_groups={self.sample_groups}, '
        repr_str += f'use_pred_boxes={self.use_pred_boxes_for_collision}, '
        repr_str += f'iou_thresh={self.iou_thresh})'
        return repr_str


def build_hard_instance_bank(source_db_path: str,
                             target_db_path: Optional[str] = None,
                             quantile_threshold: float = 50.0,
                             classes: List[str] = None,
                             source_class_mapping: Optional[Dict[str, str]] = None) -> HardInstanceBank:
    """
    Build a hard instance bank from database files.
    Args:
        source_db_path: Path to source domain dbinfos pkl file
        target_db_path: Optional path to target domain dbinfos pkl file
        quantile_threshold: Percentile threshold (0-100)
        classes: List of class names
        source_class_mapping: Dict mapping target names to source names

    Returns:    HardInstanceBank object
    """
    # Load source database
    with open(source_db_path, 'rb') as f:
        source_db_infos = pickle.load(f)
    
    # Load target database if provided
    target_db_infos = None
    if target_db_path:
        with open(target_db_path, 'rb') as f:
            target_db_infos = pickle.load(f)

    # Remap source database classes if mapping provided
    if source_class_mapping:
        print("Remapping source database classes:")
        remapped_db = {}
    
        for target_name, source_names in source_class_mapping.items():
            remapped_db[target_name] = []

             # Handle both single string and list of strings
            if isinstance(source_names, str):
                source_names = [source_names]

            for src_name in source_names:
                if src_name in source_db_infos:
                    samples = source_db_infos[src_name]
                    remapped_db[target_name].extend(samples)
                    print(f"  Mapped '{src_name}' → '{target_name}': {len(samples)} samples")
                else:
                    print(f"  Warning: Source class '{src_name}' not found in database")

        source_db_infos = remapped_db
    
    # Build bank
    bank = HardInstanceBank(
        source_db_infos=source_db_infos,
        quantile_threshold=quantile_threshold,
        classes=classes,
        target_db_infos=target_db_infos
    )
    
    # Print statistics to ensure successful creation
    print("\n" + "="*60)
    print("Hard Instance Bank Statistics:")
    print("="*60)
    stats = bank.get_stats()
    for cls_name, cls_stats in stats.items():
        print(f"\n{cls_name}:")
        print(f"  Count: {cls_stats['count']}")
        print(f"  Mean points: {cls_stats['mean_points']:.1f}")
        print(f"  Median points: {cls_stats['median_points']:.1f}")
        print(f"  Range: [{cls_stats['min_points']}, {cls_stats['max_points']}]")
    print("="*60 + "\n")
    
    return bank