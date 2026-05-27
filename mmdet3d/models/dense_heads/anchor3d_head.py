# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.utils import multi_apply, select_single_mlvl
from mmdet.utils.memory import cast_tensor_type
from mmengine.runner import amp
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.models.task_modules import PseudoSampler
from mmdet3d.models.test_time_augs import merge_aug_bboxes_3d
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import limit_period, xywhr2xyxyr
from mmdet3d.structures.ops.iou3d_calculator import bbox_overlaps_nearest_3d
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)
from .base_3d_dense_head import Base3DDenseHead
from .train_mixins import AnchorTrainMixin


@MODELS.register_module()
class Anchor3DHead(Base3DDenseHead, AnchorTrainMixin):
    """Anchor-based head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int = 256,
                 use_direction_classifier: bool = True,
                 anchor_generator: ConfigType = dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[3.9, 1.6, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size: bool = False,
                 assign_per_class: bool = False,
                 diff_rad_by_sin: bool = True,
                 dir_offset: float = -np.pi / 2,
                 dir_limit_offset: int = 0,
                 bbox_coder: ConfigType = dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 loss_dir: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss', loss_weight=0.2),
                 predict_iou: bool = False,
                 loss_iou_weight: float = 1.0,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.predict_iou = predict_iou
        self.loss_iou_weight = loss_iou_weight
        warnings.warn(
            'dir_offset and dir_limit_offset will be depressed and be '
            'incorporated into box coder in the future')

        # build anchor generator
        self.prior_generator = TASK_UTILS.build(anchor_generator)
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.prior_generator.num_base_anchors
        # build box coder
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'mmdet.FocalLoss', 'mmdet.GHMC'
        ]
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dir = MODELS.build(loss_dir)

        self._init_layers()
        self._init_assigner_sampler()

        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                TASK_UTILS.build(res) for res in self.train_cfg.assigner
            ]

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                          self.num_anchors * 2, 1)
        if self.predict_iou:
            # Per-anchor IoU regression head. Predicts a logit per
            # anchor that is sigmoided into an estimated 3D IoU
            # between the decoded prediction and its assigned GT.
            self.conv_iou = nn.Conv2d(self.feat_channels, self.num_anchors, 1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function on a single-scale feature map.

        When ``predict_iou`` is True, returns a 4-tuple ``(cls_score,
        bbox_pred, dir_cls_pred, iou_pred)``; otherwise the original
        3-tuple ``(cls_score, bbox_pred, dir_cls_pred)``.  ``iou_pred`` has
        shape ``[B, num_anchors, H, W]`` — one logit per anchor location.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = None
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(x)
        if self.predict_iou:
            iou_pred = self.conv_iou(x)
            return cls_score, bbox_pred, dir_cls_pred, iou_pred
        return cls_score, bbox_pred, dir_cls_pred

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], ...]:
        """Forward pass.

        Returns a tuple of per-level lists matching ``forward_single``'s
        return arity: 3 lists when ``predict_iou`` is False, 4 lists when
        True (the last being ``iou_preds``).
        """
        return multi_apply(self.forward_single, x)

    # TODO: Support augmentation test
    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_input_metas,
                 rescale=False,
                 **kwargs):
        aug_bboxes = []
        # only support aug_test for one sample
        for x, input_meta in zip(aug_batch_feats, aug_batch_input_metas):
            outs = self.forward(x)
            bbox_list = self.get_results(*outs, [input_meta], rescale=rescale)
            bbox_dict = dict(
                bboxes_3d=bbox_list[0].bboxes_3d,
                scores_3d=bbox_list[0].scores_3d,
                labels_3d=bbox_list[0].labels_3d)
            aug_bboxes.append(bbox_dict)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, aug_batch_input_metas,
                                            self.test_cfg)
        return [merged_bboxes]

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    input_metas: List[dict],
                    device: str = 'cuda') -> list:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            input_metas (list[dict]): contain pcd and img's meta info.
            device (str): device of current module.

        Returns:
            list[list[torch.Tensor]]: Anchors of each image, valid flags
                of each image.
        """
        num_imgs = len(input_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_anchors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        return anchor_list

    def _loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                             dir_cls_pred: Tensor,
                             iou_pred: Optional[Tensor],
                             anchors: Tensor,
                             labels: Tensor, label_weights: Tensor,
                             bbox_targets: Tensor, bbox_weights: Tensor,
                             dir_targets: Tensor, dir_weights: Tensor,
                             num_total_samples: int):
        """Calculate loss of single-level results.

        Always 4-tuple in / 4-tuple out for shape consistency under
        ``multi_apply``: when ``predict_iou`` is False, ``iou_pred`` is
        passed as None and ``loss_iou`` is returned as None.
        """
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        assert labels.max().item() <= self.num_classes
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(
                        as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        # ── IoU regression loss. Computed BEFORE add_sin_difference
        # below mutates pos_bbox_pred / pos_bbox_targets in place. ──
        loss_iou = None
        if self.predict_iou and iou_pred is not None:
            iou_pred_flat = iou_pred.permute(0, 2, 3, 1).reshape(-1)
            if num_pos > 0:
                anchors_flat = anchors.reshape(-1, anchors.shape[-1])
                pos_anchors = anchors_flat[pos_inds]
                with torch.no_grad():
                    decoded_pred = self.bbox_coder.decode(
                        pos_anchors, pos_bbox_pred)
                    decoded_gt = self.bbox_coder.decode(
                        pos_anchors, pos_bbox_targets)
                    iou_target = bbox_overlaps_nearest_3d(
                        decoded_pred, decoded_gt,
                        mode='iou', is_aligned=True,
                        coordinate='lidar').clamp(0.0, 1.0)
                pos_iou_pred = iou_pred_flat[pos_inds]
                loss_iou = F.binary_cross_entropy_with_logits(
                    pos_iou_pred, iou_target,
                    reduction='mean') * self.loss_iou_weight
            else:
                # Zero loss but keep graph connected so DDP sees the param.
                loss_iou = iou_pred_flat.sum() * 0.0

        # dir loss
        if self.use_direction_classifier:
            dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            dir_targets = dir_targets.reshape(-1)
            dir_weights = dir_weights.reshape(-1)
            pos_dir_cls_pred = dir_cls_pred[pos_inds]
            pos_dir_targets = dir_targets[pos_inds]
            pos_dir_weights = dir_weights[pos_inds]

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            if self.diff_rad_by_sin:
                pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                    pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=num_total_samples)

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_pred,
                    pos_dir_targets,
                    pos_dir_weights,
                    avg_factor=num_total_samples)
        else:
            loss_bbox = pos_bbox_pred.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_pred.sum()

        return loss_cls, loss_bbox, loss_dir, loss_iou

    @staticmethod
    def add_sin_difference(boxes1: Tensor, boxes2: Tensor) -> tuple:
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            dir_cls_preds: List[Tensor],
            *args,
            **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the head.

        Positional signature is variadic so that this method accepts both
        the legacy 3-output call (``cls, bbox, dir, gt, metas, ignore``) and
        the hybrid iou-class 4-output call (``cls, bbox, dir, iou, gt, metas, ignore``);
        ``self.predict_iou`` selects which form ``*args`` carries.

        When ``predict_iou`` is True, the returned dict also has
        ``loss_iou`` (per-level list).
        """
        if self.predict_iou:
            iou_preds = args[0]
            batch_gt_instances_3d = args[1]
            batch_input_metas = args[2]
            batch_gt_instances_ignore = args[3] if len(args) > 3 else \
                kwargs.get('batch_gt_instances_ignore', None)
        else:
            iou_preds = None
            batch_gt_instances_3d = args[0]
            batch_input_metas = args[1]
            batch_gt_instances_ignore = args[2] if len(args) > 2 else \
                kwargs.get('batch_gt_instances_ignore', None)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list = self.get_anchors(
            featmap_sizes, batch_input_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            batch_gt_instances_3d,
            batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            num_classes=self.num_classes,
            label_channels=label_channels,
            sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # Build per-level flat anchors (concatenated across batch) so the
        # hybrid IoU target can be decoded against the matching anchor.
        num_imgs = len(batch_input_metas)
        num_levels = len(cls_scores)
        mlvl_anchors_flat = [
            torch.cat([anchor_list[b][lvl_idx] for b in range(num_imgs)],
                      dim=0)
            for lvl_idx in range(num_levels)
        ]

        iou_preds_input = (
            cast_tensor_type(iou_preds, dst_type=torch.float32)
            if iou_preds is not None else [None] * num_levels)

        # num_total_samples = None
        with amp.autocast(enabled=False):
            losses_cls, losses_bbox, losses_dir, losses_iou = multi_apply(
                self._loss_by_feat_single,
                cast_tensor_type(cls_scores, dst_type=torch.float32),
                cast_tensor_type(bbox_preds, dst_type=torch.float32),
                cast_tensor_type(dir_cls_preds, dst_type=torch.float32),
                iou_preds_input,
                mlvl_anchors_flat,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                dir_targets_list,
                dir_weights_list,
                num_total_samples=num_total_samples)
        out = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)
        if self.predict_iou:
            out['loss_iou'] = losses_iou
        return out

    # ------------------------------------------------------------------
    # Predict path: overrides Base3DDenseHead to optionally route the
    # per-anchor IoU prediction through NMS and attach it to the result
    # InstanceData as ``iou_scores_3d``. Backward-compatible: when
    # ``predict_iou`` is False this falls back to the base behaviour.
    # ------------------------------------------------------------------

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        dir_cls_preds: List[Tensor],
                        *args,
                        batch_input_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigType] = None,
                        rescale: bool = False,
                        **kwargs) -> InstanceList:
        if self.predict_iou and len(args) > 0:
            iou_preds = args[0]
        else:
            iou_preds = None

        if iou_preds is None:
            return super().predict_by_feat(
                cls_scores, bbox_preds, dir_cls_preds,
                batch_input_metas=batch_input_metas,
                cfg=cfg, rescale=rescale, **kwargs)

        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) \
            == len(iou_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_anchors(
            featmap_sizes, device=cls_scores[0].device)
        mlvl_priors = [
            prior.reshape(-1, self.box_code_size) for prior in mlvl_priors
        ]
        result_list = []
        for input_id in range(len(batch_input_metas)):
            input_meta = batch_input_metas[input_id]
            cls_score_list = select_single_mlvl(cls_scores, input_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, input_id)
            dir_cls_pred_list = select_single_mlvl(dir_cls_preds, input_id)
            iou_pred_list = select_single_mlvl(iou_preds, input_id)
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                dir_cls_pred_list=dir_cls_pred_list,
                mlvl_priors=mlvl_priors,
                input_meta=input_meta,
                cfg=cfg,
                rescale=rescale,
                iou_pred_list=iou_pred_list,
                **kwargs)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                dir_cls_pred_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                input_meta: dict,
                                cfg: ConfigType,
                                rescale: bool = False,
                                iou_pred_list: Optional[List[Tensor]] = None,
                                **kwargs) -> InstanceData:
        if iou_pred_list is None:
            return super()._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                dir_cls_pred_list=dir_cls_pred_list,
                mlvl_priors=mlvl_priors,
                input_meta=input_meta,
                cfg=cfg,
                rescale=rescale,
                **kwargs)

        cfg = self.test_cfg if cfg is None else cfg
        # Hybrid-IoU ranking at val/test (ST3D / CMT
        # ``POST_PROCESSING.NMS_CONFIG.SCORE_TYPE`` analog).  Defaults to
        # ``hybrid`` when this override runs (i.e. when ``predict_iou=True``)
        # so the IoU head's score actually influences the AP-ranking the
        # metric reads from ``scores_3d``.  Set ``score_type='cls'`` in
        # ``test_cfg`` to opt out.  Weights are independent from
        # ``mean_teacher_cfg['hybrid_w_iou']`` (which gates pseudo-labels).
        score_type = cfg.get('score_type', 'hybrid')
        score_weights = cfg.get('score_weights', dict(iou=0.5, cls=0.5))
        w_iou = float(score_weights['iou'])
        w_cls = float(score_weights['cls'])
        assert abs(w_iou + w_cls - 1.0) < 1e-6, \
            f'score_weights iou+cls must sum to 1, got {w_iou} + {w_cls}'
        assert score_type in ('cls', 'iou', 'hybrid'), \
            f"score_type must be 'cls' | 'iou' | 'hybrid', got {score_type!r}"

        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_priors) \
            == len(iou_pred_list)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_iou_scores = []
        for cls_score, bbox_pred, dir_cls_pred, iou_pred, priors in zip(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                iou_pred_list, mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            assert cls_score.size()[-2:] == iou_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                cls_probs = cls_score.sigmoid()
            else:
                cls_probs = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)
            iou_score = iou_pred.permute(1, 2, 0).reshape(-1).sigmoid()

            # Compose the ranking score that flows through topk + NMS.  ``iou_score``
            # is per-anchor (shape [N]); ``cls_probs`` is per-class [N, C], so
            # broadcast iou across the class dim to preserve per-class NMS.
            if score_type == 'hybrid':
                scores = w_iou * iou_score.unsqueeze(1) + w_cls * cls_probs
            elif score_type == 'iou':
                scores = iou_score.unsqueeze(1).expand_as(cls_probs)
            else:  # 'cls'
                scores = cls_probs

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                priors = priors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
                iou_score = iou_score[topk_inds]

            bboxes = self.bbox_coder.decode(priors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_iou_scores.append(iou_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        mlvl_iou_scores = torch.cat(mlvl_iou_scores)

        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        # Route iou_score through the existing ``mlvl_attr_scores`` slot of
        # ``box3d_multiclass_nms`` so it follows the same class-wise gather
        # and topk re-ordering as the boxes.
        results = box3d_multiclass_nms(
            mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores, score_thr,
            cfg.max_num, cfg, mlvl_dir_scores,
            mlvl_attr_scores=mlvl_iou_scores)
        bboxes, scores, labels, dir_scores, iou_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels
        results.iou_scores_3d = iou_scores
        return results
