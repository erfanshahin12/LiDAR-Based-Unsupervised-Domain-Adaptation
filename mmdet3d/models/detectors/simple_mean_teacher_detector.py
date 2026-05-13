import copy

import torch
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.base import Base3DDetector


@MODELS.register_module()
class SimpleMeanTeacher3DDetector(Base3DDetector):
    """Lean Mean-Teacher wrapper for 3D detectors.

    Training uses two loss terms:
      1. Supervised loss on labeled source data (student only).
      2. Pseudo-label loss on unlabeled target data: teacher predicts,
         confidence-filtered boxes become GT for the student.

    The teacher is never directly trained; it accumulates student knowledge
    via EMA after every iteration (driven by MeanTeacherHook).

    Args:
        detector (dict): Config for the base detector shared by student and
            teacher (e.g. VoxelNetBEVRoI).
        mean_teacher_cfg (dict): Mean Teacher hyper-parameters.
            - ``ema_momentum`` (float, 0.999): EMA decay α.
            - ``update_teacher_buffers`` (bool, False): also EMA BN buffers.
            - ``conf_threshold`` (float, 0.5): teacher confidence cutoff.
            - ``source_loss_weight`` (float, 1.0): supervised loss weight.
            - ``target_loss_weight`` (float, 0.5): pseudo-label loss weight.
            - ``burn_in_iters`` (int, 0): skip target loss for first N iters.
            - ``min_pseudo_per_sample`` (int, 0): min boxes per sample to
              keep (samples below this get empty GT → zero gradient).
            - ``eval_use_teacher`` (bool, True): inference uses teacher.
            - ``verbose`` (bool, False): per-iter debug logging.
        pretrained_ckpt (str, optional): Path to pretrained checkpoint; loaded
            into student then copied to teacher.
        train_cfg (dict, optional): Passed through (unused at wrapper level).
        test_cfg (dict, optional): Passed through (unused at wrapper level).
        init_cfg: Passed to Base3DDetector.
    """

    def __init__(self,
                 detector,
                 mean_teacher_cfg=None,
                 pretrained_ckpt=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if mean_teacher_cfg is None:
            mean_teacher_cfg = {}
        self.mean_teacher_cfg = mean_teacher_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.student = MODELS.build(copy.deepcopy(detector))
        self.teacher = MODELS.build(copy.deepcopy(detector))

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student.train()
        self.teacher.train()

        if pretrained_ckpt is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(pretrained_ckpt, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            missing, unexpected = self.student.load_state_dict(state_dict, strict=False)
            logger = MMLogger.get_current_instance()
            logger.info(
                f'[SimpleMT] loaded pretrained weights from {pretrained_ckpt}: '
                f'{len(missing)} missing, {len(unexpected)} unexpected keys')
            critical = ('bbox_head.', 'voxel_encoder.', 'middle_encoder.',
                        'backbone.', 'neck.')
            bad_missing = [k for k in missing if k.startswith(critical)]
            bad_extra   = [k for k in unexpected if k.startswith(critical)]
            if bad_missing or bad_extra:
                raise RuntimeError(
                    f'Pretrained checkpoint architectural mismatch.\n'
                    f'  missing:    {bad_missing}\n'
                    f'  unexpected: {bad_extra}')

        # Copy student → teacher (parameters then BN buffers)
        for (t_name, t_p), (s_name, s_p) in zip(
                self.teacher.named_parameters(),
                self.student.named_parameters()):
            assert t_name == s_name, \
                f'Parameter name mismatch: {t_name} vs {s_name}'
            t_p.data.copy_(s_p.data)

        for (t_name, t_b), (s_name, s_b) in zip(
                self.teacher.named_buffers(),
                self.student.named_buffers()):
            assert t_name == s_name, \
                f'Buffer name mismatch: {t_name} vs {s_name}'
            t_b.copy_(s_b)

        self._train_iter = 0

        MMLogger.get_current_instance().info(
            f'[SimpleMT] init done — pretrained_ckpt={pretrained_ckpt}, '
            f'ema_momentum={mean_teacher_cfg.get("ema_momentum", 0.999)}, '
            f'burn_in_iters={mean_teacher_cfg.get("burn_in_iters", 0)}')

    # ------------------------------------------------------------------
    # EMA update (called by MeanTeacherHook after each iteration)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ema_update(self):
        alpha = self.mean_teacher_cfg.get('ema_momentum', 0.999)
        update_bufs = self.mean_teacher_cfg.get('update_teacher_buffers', False)

        for (_, t_p), (_, s_p) in zip(
                self.teacher.named_parameters(),
                self.student.named_parameters()):
            t_p.data.mul_(alpha).add_(s_p.data, alpha=1 - alpha)

        if update_bufs:
            for (_, t_b), (_, s_b) in zip(
                    self.teacher.named_buffers(),
                    self.student.named_buffers()):
                if t_b.dtype.is_floating_point:
                    t_b.data.mul_(alpha).add_(s_b.data, alpha=1 - alpha)

    # ------------------------------------------------------------------
    # Confidence filtering
    # ------------------------------------------------------------------

    def filter_teacher_predictions(self, pred):
        """Keep only teacher detections above conf_threshold."""
        thresh = self.mean_teacher_cfg.get('conf_threshold', 0.5)
        scores = pred.pred_instances_3d.scores_3d
        if scores is None or len(scores) == 0:
            return pred

        mask = scores >= thresh
        filtered = copy.copy(pred)
        filtered.pred_instances_3d = InstanceData(
            bboxes_3d=pred.pred_instances_3d.bboxes_3d[mask],
            scores_3d=scores[mask],
            labels_3d=pred.pred_instances_3d.labels_3d[mask])
        return filtered

    # ------------------------------------------------------------------
    # Pseudo-label construction
    # ------------------------------------------------------------------

    def _create_pseudo_labels(self, teacher_preds, target_samples):
        """Inject filtered teacher boxes into target samples as GT.

        Samples whose box count falls below min_pseudo_per_sample get
        empty GT so they contribute zero gradient.
        """
        min_pseudo = self.mean_teacher_cfg.get('min_pseudo_per_sample', 0)
        verbose    = self.mean_teacher_cfg.get('verbose', False)

        pseudo_samples = copy.deepcopy(target_samples)
        total = 0
        for pred, sample in zip(teacher_preds, pseudo_samples):
            boxes  = pred.pred_instances_3d.bboxes_3d
            labels = pred.pred_instances_3d.labels_3d
            if len(boxes) < min_pseudo:
                boxes  = boxes[:0]
                labels = labels[:0]
            total += len(boxes)
            sample.gt_instances_3d = InstanceData(
                bboxes_3d=boxes, labels_3d=labels)

        if verbose:
            per = [len(p.pred_instances_3d.bboxes_3d) for p in teacher_preds]
            MMLogger.get_current_instance().info(
                f'[PseudoLabels] boxes per sample — '
                f'mean={sum(per)/max(len(per),1):.1f} '
                f'min={min(per) if per else 0} '
                f'max={max(per) if per else 0} total={total}')
        if total == 0:
            MMLogger.get_current_instance().warning(
                '[PseudoLabels] No pseudo-labels kept this batch')

        return pseudo_samples

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def loss(self, batch_inputs_dict, batch_data_samples):
        """Two-term Mean Teacher loss: supervised source + pseudo-label target."""
        w_src    = self.mean_teacher_cfg.get('source_loss_weight', 1.0)
        w_tgt    = self.mean_teacher_cfg.get('target_loss_weight', 0.5)
        burn_in  = self.mean_teacher_cfg.get('burn_in_iters', 0)
        verbose  = self.mean_teacher_cfg.get('verbose', False)

        self._train_iter += 1
        past_burnin = self._train_iter > burn_in

        losses = {}

        def preprocess(inputs, data_samples):
            result = self.student.data_preprocessor(
                {'inputs': inputs, 'data_samples': data_samples})
            if isinstance(result, dict):
                return result.get('inputs', result), result.get('data_samples', data_samples)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            return result, data_samples

        def scale(v, w):
            if isinstance(v, (list, tuple)):
                return [vi * w if isinstance(vi, torch.Tensor) else vi for vi in v]
            return v * w

        # ── Term 1: Supervised source loss ──────────────────────────────
        src_in, src_samp = preprocess(
            batch_inputs_dict['labeled'],
            batch_data_samples['labeled'])
        if w_src > 0:
            for k, v in self.student.loss(src_in, src_samp).items():
                losses[f'{k}_source'] = scale(v, w_src)

        # ── Teacher inference on target (no gradient) ───────────────────
        tgt_in_raw   = batch_inputs_dict['unlabeled']['weak']
        tgt_samp_raw = batch_data_samples['unlabeled']['weak']
        tgt_in, tgt_samp = preprocess(tgt_in_raw, tgt_samp_raw)

        self.teacher.eval()
        with torch.no_grad():
            teacher_preds = self.teacher.predict(tgt_in, tgt_samp)
        self.teacher.train()

        filtered      = [self.filter_teacher_predictions(p) for p in teacher_preds]
        pseudo_samples = self._create_pseudo_labels(filtered, tgt_samp)

        # ── Term 2: Pseudo-label loss on target ──────────────────────────
        if w_tgt > 0 and past_burnin:
            # Re-voxelize from raw inputs to get fresh student-side tensors
            stu_tgt_in, _ = preprocess(tgt_in_raw, tgt_samp_raw)
            for k, v in self.student.loss(stu_tgt_in, pseudo_samples).items():
                losses[f'{k}_target'] = scale(v, w_tgt)
        elif verbose and not past_burnin:
            MMLogger.get_current_instance().info(
                f'[SimpleMT iter {self._train_iter}] burn-in active '
                f'({burn_in - self._train_iter} iters remaining)')

        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, batch_inputs, batch_data_samples, use_teacher=None, **kwargs):
        """Run inference with teacher (default) or student.

        The wrapper has no data_preprocessor, so MMEngine's val_step only
        device-transfers the points; voxelization is done here via the
        chosen subnet's preprocessor.
        """
        if use_teacher is None:
            use_teacher = self.mean_teacher_cfg.get('eval_use_teacher', True)
        subnet = self.teacher if use_teacher else self.student
        was_training = subnet.training
        subnet.eval()
        try:
            data = subnet.data_preprocessor(
                {'inputs': batch_inputs, 'data_samples': batch_data_samples},
                training=False)
            return subnet.predict(data['inputs'], data['data_samples'], **kwargs)
        finally:
            subnet.train(was_training)

    def _forward(self, batch_inputs, batch_data_samples=None):
        return self.student._forward(batch_inputs, batch_data_samples)

    def extract_feat(self, batch_inputs):
        raise NotImplementedError(
            'Call extract_feat on self.student or self.teacher directly.')
