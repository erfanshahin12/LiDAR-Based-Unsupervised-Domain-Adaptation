import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.structures import LiDARInstance3DBoxes

# DSNorm
from mmdet3d.models.layers.dsnorm import DSNorm
from mmdet3d.models.layers.dsnorm import set_ds_source, set_ds_target


@MODELS.register_module()
class MeanTeacher3DDetector(Base3DDetector):
    """Mean-Teacher wrapper for 3D detectors in MMDetection3D.

    Trains a student detector with:
    - Supervised loss on labeled source-domain data.
    - Pseudo-label loss on unlabeled target-domain data using teacher predictions.
    - Optional InfoNCE BEV-feature consistency loss between student and teacher.

    The teacher is never directly trained; it accumulates student knowledge via
    exponential moving average (EMA) after every iteration.

    Args:
        detector (dict): Config for the base detector (shared architecture for
            both student and teacher).
        mean_teacher_cfg (dict): Hyper-parameters for the Mean Teacher setup.
            Keys and their defaults:

            - ``ema_momentum`` (float, 0.999): EMA decay.
            - ``update_teacher_buffers`` (bool, True): Also EMA BN buffers.
            - ``point_cloud_range`` (list): [x_min, y_min, z_min, x_max, y_max, z_max].
              Used to normalise box centres to [-1, 1] for BEV feature sampling.
            - ``source_loss_weight`` (float, 1.0): Weight on the supervised source loss.
            - ``target_loss_weight`` (float, 0.5): Weight on the pseudo-label loss.
            - ``contrastive_weight`` (float, 0.1): Weight on the contrastive loss.
            - ``use_bev_consistency`` (bool, True): Enable the contrastive loss term.
            - ``tau`` (float, 0.07): InfoNCE temperature.
            - ``verbose`` (bool, False): Enable per-iter debug logging.
            - ``hybrid_w_iou`` (float, 0.0): Target weight on the IoU head in
              the hybrid pseudo-label score ``w_iou * iou + (1 - w_iou) * cls``.
              0 disables hybrid scoring (cls-only filtering).
            - ``iou_warmup_iters`` (int, 0): Number of student iterations to
              run cls-only filtering before the IoU head's score gates
              pseudo-labels — gives the randomly-initialised IoU head time to
              learn before it gets a say.

            Note: ``conf_threshold`` is NOT a user-facing config key.
            ``PseudoLabelRefreshHook`` writes the kneedle + count-floor
            threshold into ``mean_teacher_cfg['conf_threshold']`` at the start
            of every refresh epoch; the detector reads it from there. Setting
            it in the config has no lasting effect — it will be overwritten
            before the first training iter of epoch 0.

        pretrained_ckpt (str, optional): Path to a checkpoint to initialise the
            student (and thus the teacher) before training begins.
        train_cfg (dict, optional): Passed through; not used directly here.
        test_cfg (dict, optional): Passed through; not used directly here.
        init_cfg: Passed to :class:`Base3DDetector`.
    """

    def __init__(self,
                 detector,
                 mean_teacher_cfg=dict(
                     point_cloud_range=None,
                     ema_momentum=0.999,
                     update_teacher_buffers=False,
                     use_bev_consistency=True,
                     tau=0.07,
                     source_loss_weight=1.0,
                     target_loss_weight=0.5,
                     contrastive_weight=0.1,
                     verbose=False,
                     eval_use_teacher=True,
                     use_dsnorm=False,
                 ),
                 pretrained_ckpt=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        self.student = MODELS.build(copy.deepcopy(detector))
        self.teacher = MODELS.build(copy.deepcopy(detector))

        if mean_teacher_cfg.get('use_dsnorm', False):
            self.student = DSNorm.convert_dsnorm(self.student)
            self.teacher = DSNorm.convert_dsnorm(self.teacher)

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student.train()
        self.teacher.train()

        self.mean_teacher_cfg = mean_teacher_cfg
        self.pretrained_ckpt = pretrained_ckpt
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._ema_update_count = 0
        self._last_param_norm = None
        self._train_iter = 0

        # {lidar_path → entry}: populated by PseudoLabelRefreshHook;
        # empty → fall back to per-iteration teacher predictions.
        self.pseudo_label_store: dict = {}

    def init_weights(self):
        super().init_weights()

        if self.pretrained_ckpt is None:
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.pretrained_ckpt, map_location=device)
        state_dict = state_dict.get('state_dict', state_dict)

        missing, unexpected = self.student.load_state_dict(state_dict, strict=False)
        self.teacher.load_state_dict(state_dict, strict=False)

        MMLogger.get_current_instance().info(
            f'[init_weights] {self.pretrained_ckpt} → student+teacher: '
            f'missing={len(missing)} unexpected={len(unexpected)}')

        # Fail fast on core-layer mismatches (num_classes, anchor config, etc.).
        critical = (
            'bbox_head.', 'voxel_encoder.', 'middle_encoder.', 'backbone.', 'neck.',
            # CenterPoint (MVXTwoStageDetector) uses pts_* prefixes for the same layers.
            'pts_bbox_head.', 'pts_voxel_encoder.', 'pts_middle_encoder.',
            'pts_backbone.', 'pts_neck.',
        )
        skip = 'bbox_head.conv_iou.'   # IoU head is always newly added
        bad_missing    = [k for k in missing    if k.startswith(critical) and not k.startswith(skip)]
        bad_unexpected = [k for k in unexpected if k.startswith(critical)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f'Pretrained checkpoint mismatch in core layers — refusing to train.\n'
                f'  missing: {bad_missing}\n  unexpected: {bad_unexpected}\n'
                f'Check num_classes / anchor config vs {self.pretrained_ckpt}.')

    @torch.no_grad()
    def ema_update(self):
        """Update teacher weights (and optionally BN buffers) as EMA of student."""
        self._ema_update_count += 1
        alpha = self.mean_teacher_cfg.get('ema_momentum', 0.999)
        update_buffers = self.mean_teacher_cfg.get('update_teacher_buffers', True)

        param_updated = 0
        total_param_norm = 0.0

        for (t_name, t_param), (s_name, s_param) in zip(
                self.teacher.named_parameters(),
                self.student.named_parameters()):
            assert t_name == s_name, f'Parameter mismatch: {t_name} vs {s_name}'
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)
            param_updated += 1
            total_param_norm += t_param.data.norm().item()

        if update_buffers:
            for (t_name, t_buf), (s_name, s_buf) in zip(
                    self.teacher.named_buffers(),
                    self.student.named_buffers()):
                assert t_name == s_name, f'Buffer mismatch: {t_name} vs {s_name}'
                if t_buf.dtype.is_floating_point:
                    t_buf.data.mul_(alpha).add_(s_buf.data, alpha=1 - alpha)

        if self._ema_update_count % 50 == 0:
            avg_norm = total_param_norm / max(param_updated, 1)
            delta = ''
            if self._last_param_norm is not None:
                delta = f' (Δ: {avg_norm - self._last_param_norm:+.2e})'
            MMLogger.get_current_instance().info(
                f'[EMA #{self._ema_update_count}] avg param norm: {avg_norm:.4f}{delta}')
            self._last_param_norm = avg_norm


    # ------------------------------------------------------------------
    # Pseudo-label quality metadata helpers
    # ------------------------------------------------------------------

    def _pseudo_loss_cfg(self) -> dict:
        """Return pseudo_loss_cfg with defaults filled in."""
        defaults = dict(
            enable_soft_quality=True,   # master on/off for soft-quality weighting
            use_soft_cls_targets=True,
            min_quality_weight=0.0,
            normalize_quality_weights=False,
            weight_bbox_by_quality=True,
            weight_dir_by_quality=True,
        )
        cfg = self.mean_teacher_cfg.get('pseudo_loss_cfg', {})
        return {**defaults, **cfg}

    def _attach_pseudo_meta(self, gt, cls_scores_t, iou_scores_t, device):
        """Attach soft-target metadata to a pseudo GT InstanceData.

        The quality weight is the teacher **classification score only** — the
        single confidence signal shared by both detector heads (CenterPoint has
        no IoU head, so its ``iou_scores`` just mirror ``scores``). The former
        ``0.5·cls + 0.5·iou`` hybrid is dropped. Attaches ``cls_scores_3d``,
        ``iou_scores_3d`` and ``quality_weights_3d`` per ``pseudo_loss_cfg``.

        Master switch: when ``enable_soft_quality=False`` nothing is attached,
        so both heads fall back to hard targets — Anchor3D via its ``hasattr``
        checks, CenterHead via its default-1.0 quality. This is the single point
        that toggles soft-quality across both architectures.

        Args:
            gt: InstanceData whose ``bboxes_3d`` is already set.
            cls_scores_t: float32 Tensor [N] of teacher cls scores.
            iou_scores_t: float32 Tensor [N] of teacher IoU scores.
            device: torch device.
        """
        pcfg = self._pseudo_loss_cfg()
        if not pcfg.get('enable_soft_quality', True):
            return  # soft-quality disabled → attach nothing (hard targets)

        min_q  = float(pcfg['min_quality_weight'])
        norm_q = bool(pcfg['normalize_quality_weights'])

        if cls_scores_t is None:
            cls_scores_t = iou_scores_t  # fall back

        # Ensure tensors on the right device
        cls_t = cls_scores_t.to(device=device, dtype=torch.float32)
        iou_t = iou_scores_t.to(device=device, dtype=torch.float32)

        # Cls-only quality (hybrid cls+iou dropped per Stage-2 decision).
        quality = cls_t.clamp(min=min_q)
        if norm_q and quality.numel() > 0 and quality.max() > 0:
            quality = quality / quality.max()

        # Attach metadata selectively so downstream code can detect which
        # features are active via hasattr().
        # - cls_scores_3d present ↔ head will use soft-BCE for pseudo positives
        # - quality_weights_3d present ↔ head will scale bbox/dir weights
        # iou_scores_3d is always attached for reference / future use.
        if pcfg.get('use_soft_cls_targets', True):
            gt.cls_scores_3d = cls_t
        gt.iou_scores_3d = iou_t
        attach_quality = (pcfg.get('weight_bbox_by_quality', True)
                          or pcfg.get('weight_dir_by_quality', True)
                          or pcfg.get('use_soft_cls_targets', True))
        if attach_quality:
            gt.quality_weights_3d = quality

    def set_pseudo_labels(self, d: dict) -> None:
        """Replace the pseudo-label store with a new dict.

        Args:
            d: Mapping from ``lidar_path`` to ``{'gt_boxes': np.ndarray(N,7),
               'gt_labels': np.ndarray(N,), 'scores': np.ndarray(N,)}`` in
               canonical (weak/no-aug) lidar frame.  Stale entries from the
               previous refresh round are discarded.
        """
        self.pseudo_label_store.clear()
        self.pseudo_label_store.update(d)

    def load_pseudo_labels_from_pkl(self, path: str) -> None:
        """Load a previously dumped pseudo-label pkl file into the store."""
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.set_pseudo_labels(d)

    def _create_pseudo_labels_from_store(self, target_samp_strong, device):
        """Build pseudo-labeled strong samples from the cached store.

        Store boxes are in canonical (weak-aug) frame; they are transformed to
        the strong-aug frame via ``_transform_boxes``. Missing entries get empty GT.

        Injected hard source instances (placed by HardInstanceSampling in the
        strong pipeline) are already in the strong-aug frame and are merged here
        with full reliability (cls = iou = 1.0 → quality weight = 1.0).
        """
        pseudo_labeled_samples = copy.deepcopy(target_samp_strong)
        total_boxes = 0
        logger = MMLogger.get_current_instance()

        for samp_strong, pseudo_samp in zip(target_samp_strong, pseudo_labeled_samples):
            key = samp_strong.metainfo.get('lidar_path')
            entry = self.pseudo_label_store.get(key)

            if entry is not None and len(entry['gt_boxes']) > 0:
                boxes_t  = torch.from_numpy(entry['gt_boxes'].astype(np.float32)).to(device)
                labels_t = torch.from_numpy(entry['gt_labels'].astype(np.int64)).to(device)
                scores_t = torch.from_numpy(entry['scores'].astype(np.float32)).to(device)
                boxes_3d = self._transform_boxes(
                    LiDARInstance3DBoxes(boxes_t), samp_strong.metainfo)

                # Soft-target metadata from the refresh store.
                iou_np = entry.get('iou_scores')
                cls_np = entry.get('cls_scores')
                iou_t = (torch.from_numpy(iou_np.astype(np.float32)).to(device)
                         if iou_np is not None else scores_t)
                cls_t = (torch.from_numpy(cls_np.astype(np.float32)).to(device)
                         if cls_np is not None else scores_t)
            else:
                boxes_3d = LiDARInstance3DBoxes(torch.zeros(0, 7, device=device))
                labels_t = torch.zeros(0, dtype=torch.long, device=device)
                scores_t = torch.zeros(0, device=device)
                iou_t    = torch.zeros(0, device=device)
                cls_t    = torch.zeros(0, device=device)

            # ── Merge reliable hard-instance GT from the strong pipeline ──────
            # HardInstanceSampling places injected source instances into
            # samp_strong.gt_instances_3d in the strong-aug frame (they were
            # augmented together with their points by GlobalRotScaleTrans /
            # RandomFlip3D).  They need no _transform_boxes and are assigned
            # full quality weight (cls = iou = 1.0) as reliable GT.
            hard_gt = getattr(samp_strong, 'gt_instances_3d', None)
            if (hard_gt is not None
                    and hasattr(hard_gt, 'bboxes_3d')
                    and len(hard_gt.bboxes_3d) > 0):
                h_boxes  = hard_gt.bboxes_3d.to(device)
                h_labels = hard_gt.labels_3d.to(device)
                n_h = len(h_boxes)
                if len(boxes_3d) > 0:
                    try:
                        boxes_3d = LiDARInstance3DBoxes.cat([boxes_3d, h_boxes])
                        labels_t = torch.cat([labels_t, h_labels])
                        ones = torch.ones(n_h, device=device)
                        scores_t = torch.cat([scores_t, ones])
                        iou_t    = torch.cat([iou_t,    ones])
                        cls_t    = torch.cat([cls_t,    ones])
                    except Exception as e:
                        logger.warning(
                            f'[HardInstances] merge failed, skipping: {e}')
                else:
                    boxes_3d = h_boxes
                    labels_t = h_labels
                    ones = torch.ones(n_h, device=device)
                    scores_t, iou_t, cls_t = ones, ones, ones

            total_boxes += len(boxes_3d)
            gt = InstanceData(bboxes_3d=boxes_3d, labels_3d=labels_t)
            gt.scores_3d = scores_t
            self._attach_pseudo_meta(gt, cls_t, iou_t, device)
            pseudo_samp.gt_instances_3d = gt

        if total_boxes == 0:
            logger.warning(
                '[PseudoLabels] store produced 0 boxes (all frames missing or empty)')

        return pseudo_labeled_samples


    def _hybrid_effective_w_iou(self) -> float:
        """Effective IoU weight in the hybrid pseudo-label score.

        Linearly: ``hybrid = w_iou * iou + (1 - w_iou) * cls``.  Returns 0
        (cls-only filtering) until ``self._train_iter`` reaches
        ``iou_warmup_iters``, then the configured ``hybrid_w_iou`` target.
        This way the randomly-initialised IoU head can train before its
        score actually gates pseudo-labels.

        Both ``hybrid_w_iou`` and ``iou_warmup_iters`` live in
        ``mean_teacher_cfg``; the absence of ``hybrid_w_iou`` (or value 0)
        disables hybrid scoring entirely.
        """
        target = float(self.mean_teacher_cfg.get('hybrid_w_iou', 0.0))
        if target <= 0.0:
            return 0.0
        warmup = int(self.mean_teacher_cfg.get('iou_warmup_iters', 0))
        if self._train_iter < warmup:
            return 0.0
        return target

    def _iou_distill_weight(self) -> float:
        """Effective weight for the IoU-head distillation loss on target.

        Returns 0 until ``_train_iter`` passes ``iou_distill_warmup_iters``
        (gives the EMA teacher time to partially adapt before its KITTI IoU
        estimates are used as distillation targets), then the configured
        ``iou_distill_weight``.  0 (default) disables distillation entirely.
        """
        w = float(self.mean_teacher_cfg.get('iou_distill_weight', 0.0))
        if w <= 0.0:
            return 0.0
        warmup = int(self.mean_teacher_cfg.get('iou_distill_warmup_iters', 0))
        return 0.0 if self._train_iter < warmup else w

    def _compute_iou_distill_loss(self, x_student, x_teacher_strong):
        """BCE distillation: student IoU logits → teacher IoU logits.

        Both student and teacher see the same strongly-augmented KITTI scene
        so anchor (h, w) correspondence is exact.  No pseudo-label boxes are
        involved — the loss is purely feature-level quality consistency that
        adapts the IoU head (and the backbone/neck) to KITTI's point density
        and sensor characteristics.

        Args:
            x_student: Neck features from the student on strong-aug target.
            x_teacher_strong: Neck features from the teacher on strong-aug
                target (computed under ``torch.no_grad()``).

        Returns:
            Scalar loss tensor.
        """
        if not getattr(self.student.bbox_head, 'predict_iou', False):
            return torch.tensor(0., device=x_student[0].device)

        # student forward on x_student (second pass through head convs;
        # gradients accumulate correctly with those from bbox_head.loss()).
        student_outs = self.student.bbox_head.forward(x_student)
        with torch.no_grad():
            teacher_outs = self.teacher.bbox_head.forward(x_teacher_strong)

        # predict_iou=True → forward returns (cls_list, bbox_list, dir_list, iou_list)
        if len(student_outs) < 4:
            return torch.tensor(0., device=x_student[0].device)

        quality_weight = self.mean_teacher_cfg.get('iou_distill_quality_weight', False)
        device = x_student[0].device
        total = torch.tensor(0., device=device)
        n = 0
        for s_iou, t_iou in zip(student_outs[3], teacher_outs[3]):
            soft = t_iou.detach().sigmoid()
            if quality_weight:
                # Weight each anchor's BCE loss by the teacher's own IoU
                # quality at that location: high-confidence regions dominate
                # the distillation gradient.
                per_elem = F.binary_cross_entropy_with_logits(
                    s_iou.float(), soft.float(), reduction='none')
                # soft is in [0,1]; use it as per-element weight.
                denom = soft.sum().clamp(min=1.0)
                total += (per_elem * soft).sum() / denom
            else:
                total += F.binary_cross_entropy_with_logits(
                    s_iou.float(), soft.float(), reduction='mean')
            n += 1
        return total / max(n, 1)

    def filter_teacher_predictions(self, teacher_pred):
        """Keep only high-confidence teacher detections for pseudo-labelling.

        Hybrid thresholding: when the bbox head emits ``iou_scores_3d`` and
        the effective IoU weight is > 0 (after the warmup period configured
        via ``mean_teacher_cfg``), the threshold is applied to the hybrid
        score ``w_iou * iou + (1 - w_iou) * cls`` instead of cls alone, and
        the hybrid value is written back into ``scores_3d`` so every
        downstream path (online filter, pseudo-label store, kneedle
        threshold) thresholds the same quantity.

        Args:
            teacher_pred: Single-sample prediction object with ``pred_instances_3d``.

        Returns:
            A new prediction object with low-confidence boxes removed.
            ``bev_features`` is preserved unchanged.
        """
        conf_threshold = self.mean_teacher_cfg.get('conf_threshold', 0.6)
        w_iou_eff = self._hybrid_effective_w_iou()

        scores = teacher_pred.pred_instances_3d.scores_3d
        labels = teacher_pred.pred_instances_3d.labels_3d
        bboxes = teacher_pred.pred_instances_3d.bboxes_3d
        iou_scores = getattr(
            teacher_pred.pred_instances_3d, 'iou_scores_3d', None)

        if scores is None:
            return teacher_pred

        if iou_scores is not None and w_iou_eff > 0:
            hybrid = w_iou_eff * iou_scores + (1.0 - w_iou_eff) * scores
        else:
            hybrid = scores

        mask = hybrid >= conf_threshold

        filtered_pred = copy.copy(teacher_pred)
        inst_data = InstanceData(
            bboxes_3d=bboxes[mask],
            scores_3d=hybrid[mask],
            labels_3d=labels[mask],
        )
        if iou_scores is not None:
            inst_data.iou_scores_3d = iou_scores[mask]
        # scores is the pre-hybrid CLS (scores_3d before write-back); preserve for pkl.
        inst_data.cls_scores_3d = scores[mask]
        filtered_pred.pred_instances_3d = inst_data
        if hasattr(teacher_pred, 'bev_features'):
            filtered_pred.bev_features = teacher_pred.bev_features

        return filtered_pred


    def contrastive_loss(self, bev_s, bev_t, boxes_anchor, boxes_pos,
                         all_boxes_t=None, all_scores_t=None,
                         neg_threshold=0.25, tau=0.07):
        """Instance-aligned cross-view InfoNCE on RoI-pooled neck features.

        Anchors   = student RoI features at teacher-fg boxes in the strong frame (grad ON).
        Positives = teacher RoI features at the same fg boxes in the weak frame (no_grad);
                    anchor i is paired with positive i (the same physical object).
        Negatives = the other instances' positives ∪ teacher bg (score <= neg_threshold).

        Loss = cross_entropy(F_a @ [F_pos; F_bg].T / τ, target=i).

        Args:
            bev_s: Student BEV feature map [C, H, W], strong-aug frame, grad ON.
            bev_t: Teacher BEV feature map [C, H, W], weak-aug frame, no grad.
            boxes_anchor: Teacher fg box tensors transformed to strong frame [N, 7].
            boxes_pos:    Teacher fg box tensors in weak frame [N, 7].
            all_boxes_t:  All unfiltered teacher box tensors [M, 7] for bg mining.
            all_scores_t: Corresponding scores [M] for bg mining.
            neg_threshold: Score at-or-below which a prediction is background.
            tau:          InfoNCE temperature.
        """
        device = bev_s.device
        roi = getattr(self.student, 'roi_extractor', None)

        if roi is None or boxes_anchor is None or boxes_anchor.shape[0] == 0:
            return torch.tensor(0., device=device)

        # Anchors: student BEV, strong-frame boxes — gradients ON.
        F_a = F.normalize(
            roi.extract_roi_features(bev_s, LiDARInstance3DBoxes(boxes_anchor)),
            dim=1)   # [N, C]

        N = F_a.shape[0]
        if N == 0:
            return torch.tensor(0., device=device)

        with torch.no_grad():
            # Positives: teacher BEV, weak-frame boxes (same N, same order).
            F_pos = F.normalize(
                roi.extract_roi_features(bev_t, LiDARInstance3DBoxes(boxes_pos)),
                dim=1)   # [N, C]

            # Background negatives: unfiltered teacher preds with score <= neg_threshold.
            F_bg = None
            if all_boxes_t is not None and all_scores_t is not None:
                bg_mask = all_scores_t <= neg_threshold
                bg_boxes = all_boxes_t[bg_mask]
                if bg_boxes.shape[0] > 0:
                    F_bg = F.normalize(
                        roi.extract_roi_features(bev_t, LiDARInstance3DBoxes(bg_boxes)),
                        dim=1)   # [N_bg, C]

        # Instance-aligned InfoNCE: anchor i's positive is the SAME object in the
        # teacher's weak view (column i); every other positive (other instances)
        # and every background RoI are negatives. Preserves per-instance
        # discrimination — pulling each anchor toward ALL foreground would
        # homogenise features and harm box regression now that the loss shapes
        # the detection neck.
        keys = torch.cat([F_pos, F_bg], dim=0) if F_bg is not None else F_pos  # [N+N_bg, C]
        logits = torch.mm(F_a, keys.T) / tau                                  # [N, N+N_bg]
        targets = torch.arange(N, device=device)
        loss = F.cross_entropy(logits, targets)

        return loss


    def _transform_boxes(self, boxes, metainfo_strong):
        """Apply the strong augmentation transforms recorded in ``metainfo_strong``
        to boxes that are currently in the (un-augmented) weak-aug space.

        Transform order mirrors the strong pipeline:
            GlobalRotScaleTrans (rot → scale → trans) → RandomFlip3D (H, V).

        Args:
            boxes: :class:`LiDARInstance3DBoxes` or raw tensor in weak-aug space.
            metainfo_strong: Metainfo dict from strong-aug sample.

        Returns:
            :class:`LiDARInstance3DBoxes` in strong-aug space.
        """
        if isinstance(boxes, LiDARInstance3DBoxes):
            boxes_tensor = boxes.tensor.clone()
        else:
            boxes_tensor = boxes.clone()

        # boxes_tensor is always in LiDARInstance3DBoxes internal format
        # (z = bottom-center, origin=(0.5,0.5,0)).  Do NOT read 'box_origin'
        # from metainfo here: Pack3DDetInputs never stores that key, so the
        # old default (0.5,0.5,0.5) silently applied a spurious -h/2 z-shift
        # on every call, causing pseudo-label z to drift downward during training.
        device = boxes_tensor.device

        # ── rotation ────────────────────────────────────────────────
        pcd_rotation = metainfo_strong.get('pcd_rotation', 0.0)
        if isinstance(pcd_rotation, torch.Tensor):
            if pcd_rotation.numel() == 1:
                pcd_rotation = pcd_rotation.item()
            elif pcd_rotation.numel() == 9:
                rm = pcd_rotation.view(3, 3)
                pcd_rotation = torch.atan2(rm[1, 0], rm[0, 0]).item()
            elif pcd_rotation.numel() == 3:
                pcd_rotation = pcd_rotation[2].item()
            else:
                MMLogger.get_current_instance().warning(
                    f'_transform_boxes: unexpected rotation shape {pcd_rotation.shape}, using 0')
                pcd_rotation = 0.0
        else:
            pcd_rotation = float(pcd_rotation)

        # ── scale ────────────────────────────────────────────────────
        pcd_scale_factor = metainfo_strong.get('pcd_scale_factor', 1.0)
        if isinstance(pcd_scale_factor, torch.Tensor):
            pcd_scale_factor = pcd_scale_factor.item()
        pcd_scale_factor = float(pcd_scale_factor)

        # ── translation ──────────────────────────────────────────────
        pcd_trans = metainfo_strong.get('pcd_trans', None)
        if pcd_trans is None:
            pcd_trans = torch.zeros(3, device=device, dtype=boxes_tensor.dtype)
        elif isinstance(pcd_trans, np.ndarray):
            pcd_trans = torch.from_numpy(pcd_trans).to(device=device, dtype=boxes_tensor.dtype)
        elif isinstance(pcd_trans, (list, tuple)):
            pcd_trans = torch.tensor(pcd_trans, device=device, dtype=boxes_tensor.dtype)
        else:
            pcd_trans = pcd_trans.to(device=device, dtype=boxes_tensor.dtype)

        flip_h = metainfo_strong.get('pcd_horizontal_flip', False)
        flip_v = metainfo_strong.get('pcd_vertical_flip', False)

        # 1. Rotation
        if abs(pcd_rotation) > 1e-6:
            cos_r = torch.cos(torch.tensor(pcd_rotation, device=device))
            sin_r = torch.sin(torch.tensor(pcd_rotation, device=device))
            x_rot = cos_r * boxes_tensor[:, 0] - sin_r * boxes_tensor[:, 1]
            y_rot = sin_r * boxes_tensor[:, 0] + cos_r * boxes_tensor[:, 1]
            boxes_tensor[:, 0] = x_rot
            boxes_tensor[:, 1] = y_rot
            boxes_tensor[:, 6] += pcd_rotation

        # 2. Scale
        if abs(pcd_scale_factor - 1.0) > 1e-6:
            boxes_tensor[:, :3] *= pcd_scale_factor
            boxes_tensor[:, 3:6] *= pcd_scale_factor

        # 3. Translation
        if torch.abs(pcd_trans).sum() > 1e-6:
            boxes_tensor[:, :3] += pcd_trans

        # 4. Horizontal flip (y ← −y)
        if flip_h:
            boxes_tensor[:, 1] = -boxes_tensor[:, 1]
            boxes_tensor[:, 6] = -boxes_tensor[:, 6]

        # 5. Vertical flip (x ← −x)
        if flip_v:
            boxes_tensor[:, 0] = -boxes_tensor[:, 0]
            boxes_tensor[:, 6] = -(boxes_tensor[:, 6] + np.pi)

        # Normalise yaw to [−π, π]
        boxes_tensor[:, 6] = torch.atan2(
            torch.sin(boxes_tensor[:, 6]),
            torch.cos(boxes_tensor[:, 6]))

        # Use default origin=(0.5,0.5,0): boxes_tensor already has z at bottom-center.
        return LiDARInstance3DBoxes(boxes_tensor, box_dim=boxes_tensor.shape[-1])

    # Pseudo-label construction

    def _create_pseudo_labels(self, teacher_predictions, target_samples_strong):
        """Replace ``gt_instances_3d`` in each strong-aug sample with teacher predictions.

        Args:
            teacher_predictions: Filtered + box-transformed teacher preds.
            target_samples_strong: Strongly augmented samples (already preprocessed).

        Returns:
            Deep-copied samples with updated ``gt_instances_3d``.
        """
        verbose = self.mean_teacher_cfg.get('verbose', False)
        logger = MMLogger.get_current_instance()

        pseudo_labeled_samples = copy.deepcopy(target_samples_strong)
        total_boxes = 0

        for pred, samp_strong, sample_strong in zip(
                teacher_predictions, target_samples_strong, pseudo_labeled_samples):
            instance = pred.pred_instances_3d
            boxes  = instance.bboxes_3d
            labels = instance.labels_3d
            scores = instance.scores_3d if hasattr(instance, 'scores_3d') else None
            device = labels.device

            # Attach soft-target metadata produced by filter_teacher_predictions.
            iou_t = getattr(instance, 'iou_scores_3d', scores)
            cls_t = getattr(instance, 'cls_scores_3d', scores)
            if iou_t is None:
                iou_t = scores if scores is not None else torch.zeros(0, device=device)
            if cls_t is None:
                cls_t = scores if scores is not None else torch.zeros(0, device=device)

            # ── Merge reliable hard-instance GT from the strong pipeline ──────
            hard_gt = getattr(samp_strong, 'gt_instances_3d', None)
            if (hard_gt is not None
                    and hasattr(hard_gt, 'bboxes_3d')
                    and len(hard_gt.bboxes_3d) > 0):
                h_boxes  = hard_gt.bboxes_3d.to(device)
                h_labels = hard_gt.labels_3d.to(device)
                n_h = len(h_boxes)
                if len(boxes) > 0:
                    try:
                        boxes  = LiDARInstance3DBoxes.cat([boxes, h_boxes])
                        labels = torch.cat([labels, h_labels])
                        ones   = torch.ones(n_h, device=device)
                        scores = torch.cat([scores, ones]) if scores is not None else ones
                        iou_t  = torch.cat([iou_t, ones])
                        cls_t  = torch.cat([cls_t, ones])
                    except Exception as e:
                        logger.warning(
                            f'[HardInstances] merge failed, skipping: {e}')
                else:
                    boxes, labels = h_boxes, h_labels
                    ones = torch.ones(n_h, device=device)
                    scores, iou_t, cls_t = ones, ones, ones

            total_boxes += len(boxes)
            gt = InstanceData(bboxes_3d=boxes, labels_3d=labels)
            if scores is not None:
                gt.scores_3d = scores
            self._attach_pseudo_meta(gt, cls_t, iou_t, device)

            sample_strong.gt_instances_3d = gt

        if verbose:
            per_sample = [len(p.pred_instances_3d.bboxes_3d) for p in teacher_predictions]
            logger.info(
                f'[PseudoStats] per-sample boxes: '
                f'mean={sum(per_sample)/max(len(per_sample),1):.1f} '
                f'min={min(per_sample) if per_sample else 0} '
                f'max={max(per_sample) if per_sample else 0} '
                f'total={total_boxes}')
        if total_boxes == 0:
            logger.warning(
                '[PseudoLabels] No pseudo-labels kept (all filtered out)')

        return pseudo_labeled_samples

    # Training loss

    def loss(self, batch_inputs_dict, batch_data_samples):
        """Compute the three-term Mean Teacher loss.

        Terms:
          1. Supervised source loss (student on labeled data).
          2. Pseudo-label loss (student on unlabeled target with teacher GT).
          3. InfoNCE BEV contrastive loss (student vs teacher features).
        """
        w_source = self.mean_teacher_cfg.get('source_loss_weight', 1.0)
        w_target = self.mean_teacher_cfg.get('target_loss_weight', 0.5)
        w_cont   = self.mean_teacher_cfg.get('contrastive_weight', 0.1)
        _use_dsnorm = self.mean_teacher_cfg.get('use_dsnorm', False)

        if _use_dsnorm:
            def _ds_switch(net, domain):
                net.apply(set_ds_source if domain == 'source' else set_ds_target)
        else:
            def _ds_switch(net, domain):
                pass

        self._train_iter += 1

        device = next(self.student.parameters()).device
        logger = MMLogger.get_current_instance()
        losses = {}

        # ── Helper: run data_preprocessor and unpack result ─────────
        def preprocess(inputs, data_samples):
            result = self.student.data_preprocessor(
                {'inputs': inputs, 'data_samples': data_samples})
            if isinstance(result, dict):
                return result.get('inputs', result), result.get('data_samples', data_samples)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            return result, data_samples

        # ── Preprocess source and target (weak and strong augmentation) data ────
        source_in, source_samp = preprocess(
            batch_inputs_dict['labeled'], batch_data_samples['labeled'])

        target_weak_in, target_samp_weak = preprocess(
            batch_inputs_dict['unlabeled']['weak'], batch_data_samples['unlabeled']['weak'])
            
        target_strong_in, target_samp_strong = preprocess(
            batch_inputs_dict['unlabeled']['strong'], batch_data_samples['unlabeled']['strong'])

        # ── TERM 1: Source supervised loss ───────────────────────────
        if w_source > 0:
            _ds_switch(self.student, 'source')
            loss_source = self.student.loss(source_in, source_samp)
            for key, value in loss_source.items():
                if isinstance(value, (list, tuple)):
                    if value and isinstance(value[0], torch.Tensor):
                        value = sum(value)
                    else:
                        continue
                losses[f'{key}_source'] = value * w_source

        # ── Teacher forward on weak-augmented target ─────────────────
        # eval() freezes DSNorm/BN running stats so the teacher's target-domain
        # statistics don't drift from mini-batch noise during the training loop.
        # The refresh hook already does this correctly; this mirrors that behaviour.
        _ds_switch(self.teacher, 'target')
        self.teacher.eval()
        with torch.no_grad():
            teacher_pred = self.teacher.predict(
                target_weak_in, target_samp_weak,
                return_bev_features=True)
        self.teacher.train()

        filtered_preds = [self.filter_teacher_predictions(p) for p in teacher_pred]
        transformed_preds = copy.deepcopy(filtered_preds)

        for pred, samp in zip(transformed_preds, target_samp_strong):
            if len(pred.pred_instances_3d.bboxes_3d) > 0:
                pred.pred_instances_3d.bboxes_3d = self._transform_boxes(
                    pred.pred_instances_3d.bboxes_3d, samp.metainfo)

        # ── TERM 2: Pseudo-label loss ──────────────────────────────────
        # Pseudo-labels come from the periodically-refreshed store (populated by
        # PseudoLabelRefreshHook) or fall back to current-iteration teacher preds.
        if self.pseudo_label_store:
            pseudo_samples = self._create_pseudo_labels_from_store(
                target_samp_strong, device)
        else:
            pseudo_samples = self._create_pseudo_labels(
                transformed_preds, target_samp_strong)

        # Extract student neck features once — used for BOTH the pseudo-label
        # loss and the contrastive loss. The contrastive loss now operates on the
        # neck (detection) representation, not the pre-backbone SparseEncoder map,
        # so it shapes the features the box-regression head actually consumes.
        # VoxelNetBEVRoI returns a 2-tuple (pts_feats, bev); CenterPointBEVRoI
        # returns a 3-tuple (img_feats, pts_feats, bev) — unpack accordingly.
        _ds_switch(self.student, 'target')
        _feats = self.student.extract_feat(target_strong_in, return_bev=True)
        if len(_feats) == 3:
            _, x_strong, _ = _feats
        else:
            x_strong, _ = _feats
        # Per-sample neck map for contrastive RoI pooling (SECONDFPN returns a
        # single-level list; unwrap to a [B, C, H, W] tensor before indexing).
        neck_student = x_strong[0] if isinstance(x_strong, (list, tuple)) else x_strong

        # Effective distill weight (0 during warmup or when disabled).
        w_iou_distill = self._iou_distill_weight()

        if w_target > 0:
            loss_target = self.student.bbox_head.loss(x_strong, pseudo_samples)
            # When IoU distillation is active, suppress the pseudo-label IoU
            # loss: distillation provides a cleaner KITTI adaptation signal and
            # training the IoU head on noisy pseudo-label localization would
            # corrupt it via the circular dependency with filter_teacher_predictions.
            if w_iou_distill > 0 and self.mean_teacher_cfg.get(
                    'suppress_target_iou_loss', True):
                loss_target.pop('loss_iou', None)
            for key, value in loss_target.items():
                if isinstance(value, (list, tuple)):
                    losses[f'{key}_target'] = [v * w_target for v in value]
                else:
                    losses[f'{key}_target'] = value * w_target

        # ── TERM 4: IoU head distillation on target ───────────────────
        # Teacher and student both process the strongly-augmented KITTI scene
        # (identical voxelization → exact anchor-to-anchor alignment).
        # BCE(student_iou_logit, sigmoid(teacher_iou_logit)) adapts the IoU head
        # to KITTI's point density and object characteristics without relying on
        # pseudo-label box localization.
        if w_iou_distill > 0:
            _ds_switch(self.teacher, 'target')
            self.teacher.eval()
            with torch.no_grad():
                x_teacher_strong = self.teacher.extract_feat(target_strong_in)
            self.teacher.train()
            loss_iou_distill = self._compute_iou_distill_loss(
                x_strong, x_teacher_strong)
            losses['loss_iou_distill'] = loss_iou_distill * w_iou_distill

        # ── TERM 3: BEV contrastive loss (single-direction multi-positive) ──
        # Only runs when use_bev_consistency is True (default).
        # Iterates over the *unfiltered* teacher_pred so that fg/bg thresholds
        # (fg_threshold / neg_threshold) are applied fresh here, independent of
        # the pseudo-label conf_threshold.  Anchor positions = teacher fg boxes
        # transformed into the student's strong-aug frame.
        loss_cont_total = torch.tensor(0., device=device)
        n_valid = 0

        cont_warmup = int(self.mean_teacher_cfg.get('contrastive_warmup_iters', 0))
        if (self.mean_teacher_cfg.get('use_bev_consistency', True)
                and self._train_iter >= cont_warmup):
            tau     = self.mean_teacher_cfg.get('tau', 0.07)
            fg_thr  = self.mean_teacher_cfg.get('fg_threshold', 0.5)
            neg_thr = self.mean_teacher_cfg.get('neg_threshold', 0.25)

            for i in range(len(teacher_pred)):
                # Teacher BEV is stored on the (possibly filtered) pred object;
                # fall back to teacher_pred in case filtered_preds dropped the frame.
                bev_t = getattr(teacher_pred[i], 'bev_features', None)
                if bev_t is None:
                    bev_t = getattr(filtered_preds[i], 'bev_features', None)
                if bev_t is None:
                    continue

                all_boxes_t  = teacher_pred[i].pred_instances_3d.bboxes_3d.tensor
                all_scores_t = teacher_pred[i].pred_instances_3d.scores_3d

                # foreground: score > fg_threshold (independent of pseudo conf_threshold).
                fg_mask   = all_scores_t > fg_thr
                boxes_pos = all_boxes_t[fg_mask]   # weak frame
                if boxes_pos.shape[0] == 0:
                    continue

                # Same fg boxes transformed into the student's strong-aug frame.
                boxes_anchor = self._transform_boxes(
                    LiDARInstance3DBoxes(boxes_pos),
                    target_samp_strong[i].metainfo).tensor  # strong frame

                loss_i = self.contrastive_loss(
                    neck_student[i], bev_t, boxes_anchor, boxes_pos,
                    all_boxes_t=all_boxes_t, all_scores_t=all_scores_t,
                    neg_threshold=neg_thr, tau=tau,
                )
                loss_cont_total += loss_i
                n_valid += 1

        losses['loss_contrastive'] = (
            loss_cont_total / n_valid * w_cont if n_valid > 0
            else torch.tensor(0., device=device))

        return {k: v for k, v in losses.items() if not k.startswith('_')}


    def predict(self, batch_inputs, batch_data_samples,
                use_teacher=None, **kwargs):
        """Run inference with student or teacher.

        The chosen subnet is temporarily set to eval mode so BN running stats
        are not updated from validation data.

        Args:
            use_teacher (bool | None): If None (default), consults
                ``mean_teacher_cfg['eval_use_teacher']`` (default True).
                Pass True/False explicitly to override.
        """
        if use_teacher is None:
            use_teacher = self.mean_teacher_cfg.get('eval_use_teacher', True)
        subnet = self.teacher if use_teacher else self.student
        if self.mean_teacher_cfg.get('use_dsnorm', False):
            subnet.apply(set_ds_target)
        was_training = subnet.training
        subnet.eval()
        try:
            # The MT wrapper has no data_preprocessor of its own, Voxelize here via the
            # subnet's preprocessor before forwarding.
            data = subnet.data_preprocessor(
                {'inputs': batch_inputs, 'data_samples': batch_data_samples},
                training=False)
            result = subnet.predict(data['inputs'], data['data_samples'], **kwargs)
        finally:
            subnet.train(was_training)
        return result

    def _forward(self, batch_inputs, batch_data_samples=None):
        return self.student._forward(batch_inputs, batch_data_samples)

    def extract_feat(self, batch_inputs: torch.Tensor):
        # Required by Base3DDetector's ABC contract.  Never called on the wrapper directly
        # since it's always called on the student or teacher
        raise NotImplementedError(
            'Call extract_feat on self.student or self.teacher directly.')
