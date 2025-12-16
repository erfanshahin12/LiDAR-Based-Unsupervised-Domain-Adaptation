import torch
from torch import Tensor
from typing import Tuple, Dict, List, Optional
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import VoxelNet
from mmdet3d.structures import Det3DDataSample


@MODELS.register_module()
class VoxelNetWithBEV(VoxelNet):
    """
    Extended VoxelNet (PointPillars base) that exposes BEV features.
    This is needed for Mean-Teacher consistency loss which operates on BEV features.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Flag to control whether to return BEV features
        self.return_bev_features = False
    
    def extract_feat(self, batch_inputs_dict: dict, 
                    return_bev: bool = False) -> Tuple[Tensor]:
        """
        Extract features from points.
        
        Args:
            batch_inputs_dict: Dict containing voxel information
            return_bev: If True, return (features, bev_features) tuple
        
        Returns:
            If return_bev=False: tuple of feature maps from neck (same as original VoxelNet)
            If return_bev=True: (neck_features, bev_feature_map)
        """
        voxel_dict = batch_inputs_dict['voxels']
        
        # Voxel encoding: sparse voxels → voxel features
        voxel_features = self.voxel_encoder(
            voxel_dict['voxels'],
            voxel_dict['num_points'],
            voxel_dict['coors'])
        
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        
        # Scatter to BEV: sparse voxel features → dense BEV map
        # Here is the BEV feature map
        bev_features = self.middle_encoder(
            voxel_features,
            voxel_dict['coors'],
            batch_size)
        
        # Backbone: extract multi-scale features
        x = self.backbone(bev_features)
        
        # Neck: fuse multi-scale features
        if self.with_neck:
            x = self.neck(x)
        
        if return_bev:
            # Return both neck features (for detection) and BEV features (for consistency)
            return x, bev_features
        else:
            return x
    
    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: List[Det3DDataSample],
                return_bev_features: bool = False,
                **kwargs) -> List[Det3DDataSample]:
        """
        Predict results from a batch of inputs and data samples.
        
        Args:
            batch_inputs_dict: Input point clouds
            batch_data_samples: Data samples
            return_bev_features: If True, add BEV features to results
        
        Returns:
            List of Det3DDataSample with predictions (and optionally BEV features)
        """
        # Extract features
        if return_bev_features or self.return_bev_features:
            x, bev_features = self.extract_feat(batch_inputs_dict, return_bev=True)
        else:
            x = self.extract_feat(batch_inputs_dict, return_bev=False)
            bev_features = None
        
        # Get predictions from bbox_head
        results_list = self.bbox_head.predict(
            x, batch_data_samples, **kwargs)
        
        # Add BEV features to results if requested
        if return_bev_features or self.return_bev_features:
            batch_size = len(results_list)
            for i in range(batch_size):
                # Add BEV features to the prediction result
                # bev_features shape: [B, C, H, W]
                results_list[i].bev_features = bev_features[i]  # [C, H, W]
        
        return results_list
    
def extract_bev_features_from_predictions(predictions: List[Det3DDataSample]) -> List[Tensor]:
    """
    Helper function to extract BEV features from prediction results.

    Args:
        predictions: List of Det3DDataSample with bev_features attribute
    
    Returns:
        List of BEV feature tensors [C, H, W]
    """
    bev_features = []
    for pred in predictions:
        if hasattr(pred, 'bev_features'):
            bev_features.append(pred.bev_features)
        else:
            raise ValueError(
                "BEV features not found in predictions. "
                "Make sure to call predict() with return_bev_features=True"
            )
    return bev_features