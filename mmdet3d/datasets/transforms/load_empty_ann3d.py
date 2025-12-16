from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import BaseTransform
import torch

@TRANSFORMS.register_module()
class LoadEmptyAnnotations3D(BaseTransform):
    """Set empty ground truth for unlabeled point clouds."""

    def transform(self, input_dict: dict) -> dict:
        input_dict['gt_bboxes_3d'] = torch.zeros((0, 7), dtype=torch.float32)  # empty boxes
        input_dict['gt_labels_3d'] = torch.zeros((0,), dtype=torch.long)       # empty labels
        return input_dict