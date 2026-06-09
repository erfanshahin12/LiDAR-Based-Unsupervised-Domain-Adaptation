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
from mmdet3d.structures.ops.iou3d_calculator import bbox_overlaps_3d
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
                 roi_extractor_cfg: OptConfigType = None,
                 loss_iou_weight: float = 1.0,
                 iou_sample_cfg: OptConfigType = None,
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
        self.roi_extractor_cfg = roi_extractor_cfg
        # IoU MLP is active only when both the flag and an extractor config
        # are provided.  predict_iou=True alone (legacy configs without
        # roi_extractor_cfg) leaves the head in a no-op state.
        self.has_iou_mlp = predict_iou and (roi_extractor_cfg is not None)
        self.loss_iou_weight = loss_iou_weight
        # Default sampling config: 256 anchors per image split equally across
        # 4 IoU bins [0,0.1) [0.1,0.3) [0.3,0.5) [0.5,1.0].
        self.iou_sample_cfg = iou_sample_cfg if iou_sample_cfg is not None \
            else dict(num_per_img=256, bins=[0.0, 0.1, 0.3, 0.5, 1.0])
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
        if self.has_iou_mlp:
            # Two-stage RoI IoU head: a ROIFeatureExtractor pools a
            # rotation-aware 7×7 BEV crop at each sampled decoded-box location,
            # and iou_mlp maps the resulting per-box descriptor to a scalar IoU
            # logit.  Lazy import avoids a hard detector→dense_head dependency.
            from mmdet3d.models.detectors.voxelnet_bev_roi import (
                ROIFeatureExtractor)
            roi_out_ch = self.roi_extractor_cfg.get('out_channels', 128)
            self.iou_roi_extractor = ROIFeatureExtractor(
                in_channels=self.roi_extractor_cfg.get('in_channels', 64),
                out_channels=roi_out_ch,
                roi_size=self.roi_extractor_cfg.get('roi_size', 7),
                voxel_size=self.roi_extractor_cfg.get('voxel_size', 0.2),
                point_cloud_range=self.roi_extractor_cfg.get(
                    'point_cloud_range'),
            )
            self.iou_mlp = nn.Sequential(
                nn.Linear(roi_out_ch, roi_out_ch),
                nn.ReLU(inplace=True),
                nn.Linear(roi_out_ch, 1),
            )

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function on a single-scale feature map.

        Returns ``(cls_score, bbox_pred, dir_cls_pred)``.  IoU quality
        estimation is now done post-NMS by ``iou_mlp`` on RoI features and
        does not produce a per-anchor dense output here.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = None
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_pred

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], ...]:
        """Forward pass.

        Returns a tuple of 3 per-level lists: cls_scores, bbox_preds, dir_cls_preds.
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

    def _compute_iou_targets_level(
            self,
            anchors_flat: Tensor,
            bbox_pred_level: Tensor,
            batch_gt_instances_3d: list) -> Tensor:
        """Compute per-anchor true-3D-IoU targets for one feature-map level.

        For every anchor, decodes the predicted box and computes its maximum
        IoU with all GT boxes in the scene using *rotated* 3D IoU (not the
        nearest-BEV proxy).  A BEV-center-distance prefilter limits the
        true-3D-IoU calls to anchors that are geometrically close to at least
        one GT, keeping the computation tractable on the full feature map.

        Args:
            anchors_flat (Tensor): ``[B*H*W*A, 7]`` flat anchors for this
                level (already concatenated across the batch).
            bbox_pred_level (Tensor): ``[B, A*code_size, H, W]`` raw
                regression predictions for this level.
            batch_gt_instances_3d (list): per-image GT instance data.

        Returns:
            Tensor: ``[B*H*W*A]`` float32 soft IoU targets in ``[0, 1]``.
        """
        device = anchors_flat.device
        num_imgs = len(batch_gt_instances_3d)
        # anchors_flat may be multi-dim (e.g. [H, W, A, 7] per image,
        # concatenated along dim-0). Flatten to [B*H*W*A, 7] before decoding.
        n_anchor_dim = anchors_flat.shape[-1]  # 7
        n_total = anchors_flat.numel() // n_anchor_dim    # B*H*W*A
        n_per_img = n_total // num_imgs                   # H*W*A
        anchors_2d = anchors_flat.float().reshape(n_total, n_anchor_dim)  # [B*N, 7]

        # Decode all predicted boxes for this level (float32 for stability).
        bbox_pred_flat = bbox_pred_level.float().permute(
            0, 2, 3, 1).reshape(-1, self.box_code_size)  # [B*H*W*A, code_size]
        decoded_preds = self.bbox_coder.decode(
            anchors_2d, bbox_pred_flat)                    # [B*H*W*A, 7]

        iou_targets_list = []
        for b_idx in range(num_imgs):
            start = b_idx * n_per_img
            end = (b_idx + 1) * n_per_img
            decoded_b = decoded_preds[start:end]            # [N, 7]

            gt_instances = batch_gt_instances_3d[b_idx]
            has_gt = (hasattr(gt_instances, 'bboxes_3d')
                      and len(gt_instances.bboxes_3d) > 0)
            if not has_gt:
                iou_targets_list.append(decoded_b.new_zeros(n_per_img))
                continue

            # GT boxes: z at bottom-center (LiDARInstance3DBoxes internal rep).
            gt_boxes = gt_instances.bboxes_3d.tensor[:, :7].float().to(device)

            # ── BEV-center distance prefilter ──────────────────────────────
            # Only compute rotated-3D-IoU for anchors near at least one GT.
            pred_xy = decoded_b[:, :2]    # [N, 2]
            gt_xy = gt_boxes[:, :2]       # [M, 2]
            # search radius = half GT diagonal + 2 m margin
            max_gt_diag = torch.sqrt(
                (gt_boxes[:, 3] ** 2 + gt_boxes[:, 4] ** 2)
            ).max().item()
            search_r = max_gt_diag / 2.0 + 2.0

            dists = torch.cdist(pred_xy, gt_xy)           # [N, M]
            near_mask = dists.min(dim=1).values < search_r  # [N]

            iou_targets_b = decoded_b.new_zeros(n_per_img)
            if near_mask.sum() > 0:
                near_inds = near_mask.nonzero(as_tuple=False).squeeze(1)
                near_preds = decoded_b[near_inds]          # [K, 7]
                # True rotated-3D-IoU matrix: [K, M]
                iou_mat = bbox_overlaps_3d(
                    near_preds, gt_boxes, mode='iou', coordinate='lidar')
                max_ious = iou_mat.max(dim=1).values.clamp(0.0, 1.0)
                iou_targets_b[near_inds] = max_ious

            iou_targets_list.append(iou_targets_b)

        return torch.cat(iou_targets_list)  # [B*N]

    @staticmethod
    def _balanced_iou_sample(
            iou_targets: Tensor,
            pos_inds: Tensor,
            sample_cfg: dict) -> Tensor:
        """Return indices of anchors selected for the IoU loss.

        Positives (assigned by the assigner) are always included.  The
        remaining budget is distributed equally across IoU bins so that the
        loss sees the full quality spectrum, not just high-scoring positives.

        Args:
            iou_targets (Tensor): ``[N]`` float, soft IoU targets.
            pos_inds (Tensor): ``[P]`` long, indices of assigned positives.
            sample_cfg (dict): keys:
                * ``num_per_img`` (int) — total non-positive sampling budget.
                * ``bins`` (list[float]) — monotone bin boundaries, e.g.
                  ``[0.0, 0.1, 0.3, 0.5, 1.0]``.

        Returns:
            Tensor: 1-D LongTensor of selected anchor indices.
        """
        device = iou_targets.device
        bins = sample_cfg.get('bins', [0.0, 0.1, 0.3, 0.5, 1.0])
        num_per_img = sample_cfg.get('num_per_img', 256)
        N = len(iou_targets)

        # Mark positives so they are excluded from bin sampling.
        pos_mask = iou_targets.new_zeros(N, dtype=torch.bool)
        if len(pos_inds) > 0:
            pos_mask[pos_inds] = True

        all_inds = torch.arange(N, device=device)
        neg_inds = all_inds[~pos_mask]          # non-positives
        if len(neg_inds) == 0:
            return pos_inds

        neg_iou = iou_targets[neg_inds]
        n_bins = len(bins) - 1
        quota = max(1, num_per_img // n_bins)

        sampled_parts = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            # Last bin is inclusive on both ends.
            in_bin = ((neg_iou >= lo) & (neg_iou < hi)) if i < n_bins - 1 \
                else ((neg_iou >= lo) & (neg_iou <= hi))
            bin_inds = neg_inds[in_bin]
            if len(bin_inds) == 0:
                continue
            n_take = min(quota, len(bin_inds))
            perm = torch.randperm(len(bin_inds), device=device)[:n_take]
            sampled_parts.append(bin_inds[perm])

        sampled_negs = torch.cat(sampled_parts) if sampled_parts \
            else neg_inds.new_empty(0)

        parts = []
        if len(pos_inds) > 0:
            parts.append(pos_inds)
        if len(sampled_negs) > 0:
            parts.append(sampled_negs)
        return torch.cat(parts) if parts else all_inds.new_empty(0)

    def _loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                             dir_cls_pred: Tensor,
                             labels: Tensor, label_weights: Tensor,
                             bbox_targets: Tensor, bbox_weights: Tensor,
                             dir_targets: Tensor, dir_weights: Tensor,
                             cls_soft_targets: Tensor,
                             quality_weights: Tensor,
                             num_total_samples: int,
                             use_soft_cls: bool = False,
                             use_quality: bool = False):
        """Calculate cls/bbox/dir loss for a single feature-map level.

        IoU loss is computed separately in ``_iou_mlp_loss`` because it
        requires the BEV feature map (not available per-level here).
        """
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        cls_soft_targets = cls_soft_targets.reshape(-1)
        quality_weights   = quality_weights.reshape(-1)
        assert labels.max().item() <= self.num_classes

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(
                        as_tuple=False).reshape(-1)

        if use_soft_cls and len(pos_inds) > 0:
            # ── Soft-BCE for pseudo positives ─────────────────────────────
            # Zero out focal-loss weight at positives so they are excluded
            # from the hard-label path. Negatives / background keep focal.
            label_weights_hard = label_weights.clone()
            label_weights_hard[pos_inds] = 0.0
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights_hard,
                avg_factor=num_total_samples)

            # Soft BCE: student cls logit at assigned class vs teacher cls score.
            # cls_score[pos, label[pos]] gives the logit for the correct class.
            pos_labels = labels[pos_inds]                       # [P]
            pos_logits = cls_score[pos_inds, pos_labels]        # [P]
            pos_targets = cls_soft_targets[pos_inds]            # [P]
            pos_qw      = quality_weights[pos_inds]             # [P]
            loss_cls_soft = F.binary_cross_entropy_with_logits(
                pos_logits, pos_targets,
                weight=pos_qw,
                reduction='sum') / max(num_total_samples, 1)
            loss_cls_soft = loss_cls_soft * self.loss_cls.loss_weight
            loss_cls = loss_cls + loss_cls_soft
        else:
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)

        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

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

            # ── Quality-weighted bbox / dir for pseudo-label positives ────
            # Multiplying weights by the hybrid quality score means that
            # low-confidence pseudo boxes contribute a smaller gradient than
            # high-confidence ones.  Source GT quality_weights are all 1.0,
            # so this is a strict no-op on the source path.
            if use_quality:
                pos_quality = quality_weights[pos_inds]             # [P]
                pos_bbox_weights = pos_bbox_weights * pos_quality.unsqueeze(-1)
                if self.use_direction_classifier:
                    pos_dir_weights = pos_dir_weights * pos_quality

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

        return loss_cls, loss_bbox, loss_dir

    def _iou_mlp_loss(self,
                      bbox_preds: List[Tensor],
                      mlvl_anchors_flat: List[Tensor],
                      mlvl_iou_targets: List[Tensor],
                      labels_list: List[Tensor],
                      bev_features: List[Tensor],
                      num_imgs: int) -> Optional[Tensor]:
        """Compute the balanced-bin BCE IoU loss for the RoI-based iou_mlp.

        For every level and every image, a balanced subset of anchor indices
        is selected (same ``_balanced_iou_sample`` as before), the
        corresponding decoded boxes are passed through the rotation-aware
        ``iou_roi_extractor``, and the resulting per-box descriptors are
        scored by ``iou_mlp``.

        Args:
            bbox_preds: Per-level ``[B, A*code_size, H, W]`` regression preds.
            mlvl_anchors_flat: Per-level ``[B*H*W*A, 7]`` flat anchors.
            mlvl_iou_targets: Per-level ``[B*H*W*A]`` true-3D-IoU targets.
            labels_list: Per-level ``[B, H*W*A]`` anchor labels.
            bev_features: Per-image ``[C, H_bev, W_bev]`` BEV feature maps.
            num_imgs: Batch size.

        Returns:
            Scalar IoU loss tensor, or None if no valid samples were found.
        """
        num_levels = len(bbox_preds)
        total_loss = None
        n_samples = 0

        for lvl_idx in range(num_levels):
            anchors_flat = mlvl_anchors_flat[lvl_idx]   # [B*N, 7]
            iou_targets_lvl = mlvl_iou_targets[lvl_idx] # [B*N]
            labels_flat = labels_list[lvl_idx].reshape(-1)  # [B*N]
            n_per_img = anchors_flat.shape[0] // num_imgs

            for b_idx in range(num_imgs):
                s = b_idx * n_per_img
                e = s + n_per_img
                anchors_b = anchors_flat[s:e]            # [N, 7]
                iou_tgts_b = iou_targets_lvl[s:e]        # [N]
                labels_b = labels_flat[s:e]              # [N]

                # Positive anchors (assigned by bbox_assigner)
                pos_inds = (
                    (labels_b >= 0) & (labels_b < self.num_classes)
                ).nonzero(as_tuple=False).squeeze(1)

                sample_inds = self._balanced_iou_sample(
                    iou_tgts_b, pos_inds, self.iou_sample_cfg)
                if len(sample_inds) == 0:
                    continue

                # Decode sampled anchor predictions (no gradient through boxes)
                bbox_pred_b = (bbox_preds[lvl_idx][b_idx]
                               .permute(1, 2, 0)
                               .reshape(-1, self.box_code_size))
                decoded_b = self.bbox_coder.decode(
                    anchors_b.float(),
                    bbox_pred_b.float().detach())         # [N, 7]
                sampled_boxes = decoded_b[sample_inds]   # [K, 7]
                sampled_tgts = iou_tgts_b[sample_inds]  # [K]

                # RoI features → iou_mlp → logit
                roi_feats = self.iou_roi_extractor.extract_roi_features(
                    bev_features[b_idx], sampled_boxes)  # [K, out_ch]
                iou_logits = self.iou_mlp(roi_feats).squeeze(-1).float()  # [K]

                loss = F.binary_cross_entropy_with_logits(
                    iou_logits, sampled_tgts,
                    reduction='sum') * self.loss_iou_weight

                total_loss = loss if total_loss is None else total_loss + loss
                n_samples += len(sample_inds)

        if total_loss is None:
            return None
        return total_loss / max(n_samples, 1)

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
        # forward() now always returns (cls, bbox, dir) — no dense iou_preds.
        # BEV features for the RoI IoU head arrive via kwarg from VoxelNetBEVRoI.
        bev_features = kwargs.pop('bev_features', None)
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
         num_total_neg, cls_soft_targets_list,
         quality_weights_list) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # Detect whether this is a pseudo-label batch (any GT carries metadata).
        # These bools propagate per-call so the source path is never affected.
        use_soft_cls = any(
            hasattr(g, 'cls_scores_3d') for g in batch_gt_instances_3d)
        use_quality  = any(
            hasattr(g, 'quality_weights_3d') for g in batch_gt_instances_3d)

        num_imgs = len(batch_input_metas)
        num_levels = len(cls_scores)

        # Build per-level flat anchors (concatenated across batch) for
        # decoding in the IoU target computation below.
        mlvl_anchors_flat = [
            torch.cat([anchor_list[b][lvl_idx] for b in range(num_imgs)],
                      dim=0)
            for lvl_idx in range(num_levels)
        ]

        # ── True-3D-IoU target computation (no gradient needed) ───────────
        # For each level decode all predicted boxes and compute, for every
        # anchor, the maximum *rotated* 3D IoU with the scene's GT boxes.
        # These flat targets are shared by both the _loss_by_feat_single
        # (unused slot — kept for potential future use) and _iou_mlp_loss.
        if self.has_iou_mlp and bev_features is not None:
            with torch.no_grad():
                mlvl_iou_targets = [
                    self._compute_iou_targets_level(
                        mlvl_anchors_flat[lvl_idx],
                        bbox_preds[lvl_idx],
                        batch_gt_instances_3d)
                    for lvl_idx in range(num_levels)
                ]
        else:
            mlvl_iou_targets = None

        with amp.autocast(enabled=False):
            losses_cls, losses_bbox, losses_dir = multi_apply(
                self._loss_by_feat_single,
                cast_tensor_type(cls_scores, dst_type=torch.float32),
                cast_tensor_type(bbox_preds, dst_type=torch.float32),
                cast_tensor_type(dir_cls_preds, dst_type=torch.float32),
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                dir_targets_list,
                dir_weights_list,
                cls_soft_targets_list,
                quality_weights_list,
                num_total_samples=num_total_samples,
                use_soft_cls=use_soft_cls,
                use_quality=use_quality)
        out = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)

        if self.has_iou_mlp and bev_features is not None \
                and mlvl_iou_targets is not None:
            with amp.autocast(enabled=False):
                loss_iou = self._iou_mlp_loss(
                    cast_tensor_type(bbox_preds, dst_type=torch.float32),
                    mlvl_anchors_flat,
                    mlvl_iou_targets,
                    labels_list,
                    bev_features,
                    num_imgs)
            if loss_iou is not None:
                out['loss_iou'] = loss_iou
        return out

    # ------------------------------------------------------------------
    # Predict path: standard NMS using cls scores; iou_mlp runs post-NMS
    # on surviving boxes if bev_features are available.
    # ------------------------------------------------------------------

    def predict(self,
                x,
                batch_data_samples,
                rescale: bool = False,
                bev_features=None,
                **kwargs):
        """Forward predict, forwarding bev_features to predict_by_feat.

        Overrides the base class predict() which lacks **kwargs forwarding,
        so bev_features passed by VoxelNetBEVRoI would otherwise be rejected.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples]
        outs = self(x)
        return self.predict_by_feat(
            *outs,
            batch_input_metas=batch_input_metas,
            rescale=rescale,
            bev_features=bev_features,
            **kwargs)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        dir_cls_preds: List[Tensor],
                        batch_input_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigType] = None,
                        rescale: bool = False,
                        bev_features: Optional[List[Tensor]] = None,
                        **kwargs) -> InstanceList:
        """Standard NMS followed by post-NMS RoI IoU scoring.

        NMS ranking uses classification scores only.  After NMS, if
        ``bev_features`` are provided and ``iou_mlp`` is configured, each
        surviving box is scored by the rotation-aware RoI IoU head and the
        result is attached as ``iou_scores_3d`` on the InstanceData.
        """
        result_list = super().predict_by_feat(
            cls_scores, bbox_preds, dir_cls_preds,
            batch_input_metas=batch_input_metas,
            cfg=cfg, rescale=rescale, **kwargs)

        if self.has_iou_mlp and bev_features is not None:
            for i, results in enumerate(result_list):
                boxes = results.bboxes_3d
                if len(boxes) > 0:
                    with torch.no_grad():
                        roi_feats = self.iou_roi_extractor.extract_roi_features(
                            bev_features[i], boxes)
                        iou_scores = self.iou_mlp(roi_feats).squeeze(-1).sigmoid()
                else:
                    iou_scores = bev_features[i].new_zeros(0)
                results.iou_scores_3d = iou_scores

        return result_list
