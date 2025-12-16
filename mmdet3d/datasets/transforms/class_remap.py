import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ClassRemap(BaseTransform):
    """Remap class names in 3D object detection annotations.
    
    This transform remaps ground truth class names from source dataset naming
    convention to target dataset naming convention. Useful for domain adaptation
    where datasets have different class naming schemes.
    
    Args:
        mapping (dict): Dictionary mapping source class names to target class names.
            Classes not in mapping will be filtered out.
        keep_unmapped (bool): If True, keep boxes with unmapped classes. 
            If False, filter them out. Default: False.
    
    Example:
        >>> transform = ClassRemap(
        ...     mapping={
        ...         'car': 'Car',
        ...         'bicycle': 'Cyclist',
        ...         'pedestrian': 'Pedestrian',
        ...         'truck': 'Car',  # Map truck to Car
        ...         'bus': 'Car',    # Map bus to Car
        ...     },
        ...     keep_unmapped=False
        ... )
    """
    
    def __init__(self, mapping, keep_unmapped=False):
        self.mapping = mapping
        self.keep_unmapped = keep_unmapped
    
    def transform(self, results):
        """Transform function to remap class names.
        
        Args:
            results (dict): Result dict containing annotations.
            
        Returns:
            dict: Updated result dict with remapped class names.
        """
        if 'gt_labels_3d' not in results or 'gt_bboxes_3d' not in results:
            return results
        
        # Get original annotations
        gt_names = results.get('gt_names', None)
        if gt_names is None:
            return results
        
        # Remap class names
        new_names = []
        keep_indices = []
        
        for idx, name in enumerate(gt_names):
            if name in self.mapping:
                new_names.append(self.mapping[name])
                keep_indices.append(idx)
            elif self.keep_unmapped:
                new_names.append(name)
                keep_indices.append(idx)
            # else: filter out this box
        
        if len(keep_indices) == 0:
            # No valid boxes after remapping
            results['gt_bboxes_3d'] = results['gt_bboxes_3d'][np.array([], dtype=np.int64)]
            results['gt_labels_3d'] = np.array([], dtype=np.int64)
            results['gt_names'] = np.array([], dtype=str)
            return results
        
        keep_indices = np.array(keep_indices)
        
        # Filter annotations
        results['gt_bboxes_3d'] = results['gt_bboxes_3d'][keep_indices]
        results['gt_labels_3d'] = results['gt_labels_3d'][keep_indices]
        results['gt_names'] = np.array(new_names)
        
        # Handle optional fields
        if 'gt_velocities' in results and results['gt_velocities'] is not None:
            results['gt_velocities'] = results['gt_velocities'][keep_indices]
        
        if 'attr_labels' in results and results['attr_labels'] is not None:
            results['attr_labels'] = results['attr_labels'][keep_indices]
        
        if 'centers_2d' in results and results['centers_2d'] is not None:
            results['centers_2d'] = results['centers_2d'][keep_indices]
        
        if 'depths' in results and results['depths'] is not None:
            results['depths'] = results['depths'][keep_indices]
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mapping={self.mapping}, '
        repr_str += f'keep_unmapped={self.keep_unmapped})'
        return repr_str


@TRANSFORMS.register_module()
class ClassRemapWithLabel(BaseTransform):
    """Remap class names AND update label indices based on new class list.
    
    This is more robust as it also updates gt_labels_3d to match the new
    class order defined in metainfo.
    
    Args:
        mapping (dict): Dictionary mapping source class names to target class names.
        class_names (list): Target class names list (should match metainfo classes).
        keep_unmapped (bool): If True, keep boxes with unmapped classes.
            Default: False.
    
    Example:
        >>> transform = ClassRemapWithLabel(
        ...     mapping={
        ...         'car': 'Car',
        ...         'bicycle': 'Cyclist',
        ...         'pedestrian': 'Pedestrian',
        ...     },
        ...     class_names=['Car', 'Pedestrian', 'Cyclist']  # KITTI order
        ... )
    """
    
    def __init__(self, mapping, class_names, keep_unmapped=False):
        self.mapping = mapping
        self.class_names = class_names
        self.keep_unmapped = keep_unmapped
        
        # Build name to index mapping for efficient lookup
        self.name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    def transform(self, results):
        """Transform function to remap class names and labels."""
        if 'gt_labels_3d' not in results or 'gt_bboxes_3d' not in results:
            return results
        
        gt_names = results.get('gt_names', None)
        if gt_names is None:
            return results
        
        new_names = []
        new_labels = []
        keep_indices = []
        
        for idx, name in enumerate(gt_names):
            if name in self.mapping:
                remapped_name = self.mapping[name]
                if remapped_name in self.name_to_idx:
                    new_names.append(remapped_name)
                    new_labels.append(self.name_to_idx[remapped_name])
                    keep_indices.append(idx)
            elif self.keep_unmapped and name in self.name_to_idx:
                new_names.append(name)
                new_labels.append(self.name_to_idx[name])
                keep_indices.append(idx)
        
        if len(keep_indices) == 0:
            results['gt_bboxes_3d'] = results['gt_bboxes_3d'][np.array([], dtype=np.int64)]
            results['gt_labels_3d'] = np.array([], dtype=np.int64)
            results['gt_names'] = np.array([], dtype=str)
            return results
        
        keep_indices = np.array(keep_indices)
        
        # Update all annotation fields
        results['gt_bboxes_3d'] = results['gt_bboxes_3d'][keep_indices]
        results['gt_labels_3d'] = np.array(new_labels, dtype=np.int64)
        results['gt_names'] = np.array(new_names)
        
        # Handle optional fields
        if 'gt_velocities' in results and results['gt_velocities'] is not None:
            results['gt_velocities'] = results['gt_velocities'][keep_indices]
        
        if 'attr_labels' in results and results['attr_labels'] is not None:
            results['attr_labels'] = results['attr_labels'][keep_indices]
        
        if 'centers_2d' in results and results['centers_2d'] is not None:
            results['centers_2d'] = results['centers_2d'][keep_indices]
        
        if 'depths' in results and results['depths'] is not None:
            results['depths'] = results['depths'][keep_indices]
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mapping={self.mapping}, '
        repr_str += f'class_names={self.class_names}, '
        repr_str += f'keep_unmapped={self.keep_unmapped})'
        return repr_str