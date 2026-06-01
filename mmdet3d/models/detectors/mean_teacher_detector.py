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
        critical = ('bbox_head.', 'voxel_encoder.', 'middle_encoder.', 'backbone.', 'neck.')
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
        """
        pseudo_labeled_samples = copy.deepcopy(target_samp_strong)
        total_boxes = 0

        for samp_strong, pseudo_samp in zip(target_samp_strong, pseudo_labeled_samples):
            key = samp_strong.metainfo.get('lidar_path')
            entry = self.pseudo_label_store.get(key)

            if entry is not None and len(entry['gt_boxes']) > 0:
                boxes_t  = torch.from_numpy(entry['gt_boxes'].astype(np.float32)).to(device)
                labels_t = torch.from_numpy(entry['gt_labels'].astype(np.int64)).to(device)
                scores_t = torch.from_numpy(entry['scores'].astype(np.float32)).to(device)
                boxes_3d = self._transform_boxes(
                    LiDARInstance3DBoxes(boxes_t), samp_strong.metainfo)
            else:
                boxes_3d = LiDARInstance3DBoxes(torch.zeros(0, 7, device=device))
                labels_t = torch.zeros(0, dtype=torch.long, device=device)
                scores_t = torch.zeros(0, device=device)

            total_boxes += len(boxes_3d)
            gt = InstanceData(bboxes_3d=boxes_3d, labels_3d=labels_t)
            gt.scores_3d = scores_t
            pseudo_samp.gt_instances_3d = gt

        if total_boxes == 0:
            MMLogger.get_current_instance().warning(
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
        verbose = self.mean_teacher_cfg.get('verbose', False)
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


    def contrastive_loss(self, bev_s, bev_t, boxes_t_s, boxes_t,
                         all_boxes_t=None, all_scores_t=None,
                         conf_threshold=0.6, tau=0.07):
        """Symmetric InfoNCE on RoI-pooled BEV features (CMT-style).

        RoI features replace point-sampled BEV features; low-confidence teacher
        predictions serve as explicit background negatives in the denominator.
        """
        device = bev_s.device
        roi_extractor = getattr(self.student, 'roi_extractor', None)

        if roi_extractor is None or boxes_t is None or boxes_t.shape[0] == 0:
            return torch.tensor(0., device=device)

        # RoI features: boxes_t and boxes_t_s share the same ordering → F_s[i] ↔ F_t[i].
        F_s = F.normalize(roi_extractor.extract_roi_features(
            bev_s, LiDARInstance3DBoxes(boxes_t_s)), dim=1)   # [N, C]
        F_t = F.normalize(roi_extractor.extract_roi_features(
            bev_t, LiDARInstance3DBoxes(boxes_t)), dim=1)     # [N, C]

        N = F_s.shape[0]
        if N == 0:
            return torch.tensor(0., device=device)

        # Background negatives from low-confidence teacher predictions.
        F_bg = None
        if all_boxes_t is not None and all_scores_t is not None:
            bg_boxes = all_boxes_t[all_scores_t < conf_threshold]
            if bg_boxes.shape[0] > 0:
                F_bg = F.normalize(roi_extractor.extract_roi_features(
                    bev_t, LiDARInstance3DBoxes(bg_boxes)), dim=1)  # [N_bg, C]

        # s→t: student queries against teacher fg (pos) + teacher bg (neg).
        F_neg = torch.cat([F_t, F_bg], dim=0) if F_bg is not None else F_t
        pos_st = (F_s * F_t).sum(dim=1) / tau
        loss_st = -(pos_st - torch.logsumexp(torch.mm(F_s, F_neg.T) / tau, dim=1)).mean()

        # t→s: teacher fg queries against student fg (fg-only negatives).
        pos_ts = (F_t * F_s).sum(dim=1) / tau
        loss_ts = -(pos_ts - torch.logsumexp(torch.mm(F_t, F_s.T) / tau, dim=1)).mean()

        return (loss_st + loss_ts) * 0.5


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

        for pred, sample_strong in zip(teacher_predictions, pseudo_labeled_samples):
            instance = pred.pred_instances_3d
            boxes  = instance.bboxes_3d
            labels = instance.labels_3d
            scores = instance.scores_3d if hasattr(instance, 'scores_3d') else None

            total_boxes += len(boxes)
            gt = InstanceData(bboxes_3d=boxes, labels_3d=labels)
            if scores is not None:
                gt.scores_3d = scores
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
        verbose  = self.mean_teacher_cfg.get('verbose', False)
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
        _ds_switch(self.teacher, 'target')
        self.teacher.train()
        with torch.no_grad():
            teacher_pred = self.teacher.predict(
                target_weak_in, target_samp_weak,
                return_bev_features=True)

        filtered_preds = [self.filter_teacher_predictions(p) for p in teacher_pred]
        if verbose:
            conf_thr_log = self.mean_teacher_cfg.get('conf_threshold', 0.6)
            total_before = sum(len(p.pred_instances_3d.scores_3d) for p in teacher_pred)
            total_after  = sum(len(p.pred_instances_3d.scores_3d) for p in filtered_preds)
            per_sample   = [len(p.pred_instances_3d.scores_3d) for p in filtered_preds]
            logger.info(
                f'[Filter] kept {total_after}/{total_before} boxes across '
                f'{len(filtered_preds)} samples (thresh={conf_thr_log:.2f})  '
                f'per-sample: {per_sample}')

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

        # Extract student features once — neck features for pseudo-label loss,
        # BEV features for contrastive loss — avoiding a second forward pass.
        _ds_switch(self.student, 'target')
        x_strong, bev_features_student = self.student.extract_feat(
            target_strong_in, return_bev=True)

        if w_target > 0:
            loss_target = self.student.bbox_head.loss(x_strong, pseudo_samples)
            for key, value in loss_target.items():
                if isinstance(value, (list, tuple)):
                    losses[f'{key}_target'] = [v * w_target for v in value]
                else:
                    losses[f'{key}_target'] = value * w_target

        # ── TERM 3: BEV contrastive loss ──────────────────────────────
        loss_cont_total = torch.tensor(0., device=device)
        n_valid = 0
        tau = self.mean_teacher_cfg.get('tau', 0.07)
        conf_thr = self.mean_teacher_cfg.get('conf_threshold', 0.6)

        for i in range(len(filtered_preds)):
            bev_t = getattr(filtered_preds[i], 'bev_features', None)
            if bev_t is None:
                continue
            boxes_t   = filtered_preds[i].pred_instances_3d.bboxes_3d.tensor
            boxes_t_s = transformed_preds[i].pred_instances_3d.bboxes_3d.tensor
            if boxes_t.shape[0] == 0:
                continue
            loss_i = self.contrastive_loss(
                bev_features_student[i], bev_t, boxes_t_s, boxes_t,
                all_boxes_t=teacher_pred[i].pred_instances_3d.bboxes_3d.tensor,
                all_scores_t=teacher_pred[i].pred_instances_3d.scores_3d,
                conf_threshold=conf_thr, tau=tau,
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
