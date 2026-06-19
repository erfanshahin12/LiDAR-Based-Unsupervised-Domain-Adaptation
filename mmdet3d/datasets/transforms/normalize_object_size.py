# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.random_object_scaling import (
    scale_objects_in_place)


@TRANSFORMS.register_module()
class NormalizeObjectSize(BaseTransform):
    """Normalize Object Size (NOS).

    Deterministically shifts each eligible object's ``[l, w, h]`` by the fixed
    residual ``size_res`` (negative ⇒ shrink), scaling the interior points with
    the box so the geometry stays consistent. ``z`` is shifted by
    ``(new_h - old_h) / 2`` to keep the object grounded.

    Use it on the **source** pipeline to map source-domain object sizes onto the
    target-domain mean (``size_res = source_mean - target_mean``), removing the
    systematic size bias at its origin. Pair it with a *symmetric*
    ``RandomObjectScaling`` (e.g. ``[0.95, 1.05]``) so ROS adds only variance and
    does not re-shift the mean NOS just corrected.

    Port of ST3D's ``normalize_object_size``. Shares the per-object scaling core
    with :class:`RandomObjectScaling` via ``scale_objects_in_place``.

    Required Keys: ``points``, ``gt_bboxes_3d``, ``gt_labels_3d``
    Modified Keys: ``points``, ``gt_bboxes_3d``
    """

    def __init__(self,
                 size_res: Sequence[float],
                 floor: float = 0.1,
                 class_names: Optional[List[str]] = None) -> None:
        assert len(size_res) == 3, 'size_res must be [dl, dw, dh]'
        self.size_res = np.asarray(size_res, dtype=np.float32)
        self.floor = float(floor)
        self.class_names = list(class_names) if class_names is not None else None

    def transform(self, input_dict: dict) -> dict:
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']

        if len(gt_bboxes_3d) == 0:
            return input_dict

        boxes_np = gt_bboxes_3d.tensor.numpy().copy()
        points_np = points.tensor.numpy().copy()

        eligible = self._eligible_mask(input_dict, n=boxes_np.shape[0])
        if not eligible.any():
            return input_dict

        size_res = self.size_res.astype(boxes_np.dtype)
        floor = self.floor

        def get_scale(boxes: np.ndarray, k: int) -> Optional[np.ndarray]:
            lwh = boxes[k, 3:6]
            new_dims = np.maximum(lwh + size_res, floor)
            return new_dims / np.maximum(lwh, 1e-6)

        points_np, boxes_np = scale_objects_in_place(
            boxes_np, points_np, eligible, get_scale)

        input_dict['points'] = points.new_point(points_np)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(boxes_np)
        return input_dict

    def _eligible_mask(self, input_dict: dict, n: int) -> np.ndarray:
        if self.class_names is None:
            return np.ones(n, dtype=bool)
        gt_labels = np.asarray(input_dict['gt_labels_3d'])
        all_classes = None
        if 'box_type_3d' in input_dict and 'metainfo' in input_dict:
            all_classes = input_dict['metainfo'].get('classes', None)
        if all_classes is None:
            return gt_labels >= 0
        keep_ids = {
            i for i, c in enumerate(all_classes) if c in self.class_names
        }
        return np.array([int(l) in keep_ids for l in gt_labels], dtype=bool)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'size_res={self.size_res.tolist()}, '
                f'floor={self.floor}, '
                f'class_names={self.class_names})')
