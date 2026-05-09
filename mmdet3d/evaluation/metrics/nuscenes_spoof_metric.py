import torch
from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics import NuScenesMetric

@METRICS.register_module()
class NuScenesSpoofVelocityMetric(NuScenesMetric):
    def process(self, data_batch: dict, data_samples: list) -> None:
        """Intercept data_samples and pad 7-DoF boxes to 9-DoF with zeros."""
        
        for data_sample in data_samples:
            # Safely get the 3D predictions (MMEngine uses dict-like access here)
            if 'pred_instances_3d' in data_sample:
                pred_3d = data_sample['pred_instances_3d']
                
                if 'bboxes_3d' in pred_3d:
                    bboxes_3d = pred_3d['bboxes_3d']
                    
                    # If the boxes are 7-DoF, pad them to 9-DoF (vx=0, vy=0)
                    if bboxes_3d.tensor.shape[-1] == 7:
                        # Create a [N, 2] tensor of zeros matching the device of your boxes
                        zeros = bboxes_3d.tensor.new_zeros((bboxes_3d.tensor.shape[0], 2))
                        padded_tensor = torch.cat([bboxes_3d.tensor, zeros], dim=-1)
                        
                        # Dynamically recreate the exact same box type (LiDARInstance3DBoxes)
                        # but with 9 dimensions so NuScenes doesn't crash
                        new_bboxes_3d = type(bboxes_3d)(
                            padded_tensor, 
                            box_dim=9, 
                            origin=(0.5, 0.5, 0.5)
                        )
                        
                        # Overwrite the old 7-DoF boxes
                        pred_3d['bboxes_3d'] = new_bboxes_3d
                        
        # Now pass the padded 9-DoF data to the original NuScenes process function
        super().process(data_batch, data_samples)