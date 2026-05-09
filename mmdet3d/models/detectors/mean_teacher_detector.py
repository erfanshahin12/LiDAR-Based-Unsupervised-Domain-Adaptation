import copy
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.structures import LiDARInstance3DBoxes


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
            - ``conf_threshold`` (float, 0.6): Teacher confidence threshold for
              pseudo-label filtering.
            - ``use_class_specific_thresh`` (bool, False): Per-class thresholds.
            - ``class_thresholds`` (dict, None): {class_id: threshold}.
            - ``source_loss_weight`` (float, 1.0): Weight on the supervised source loss.
            - ``target_loss_weight`` (float, 0.5): Weight on the pseudo-label loss.
            - ``contrastive_weight`` (float, 0.1): Weight on the contrastive loss.
            - ``use_bev_consistency`` (bool, True): Enable the contrastive loss term.
            - ``tau`` (float, 0.07): InfoNCE temperature.
            - ``symmetric_contrastive`` (bool, True): Average s→t and t→s directions.
            - ``burn_in_iters`` (int, 0): Skip target + contrastive losses for the
              first N iterations (teacher is too close to student to be useful).
            - ``min_pseudo_per_sample`` (int, 0): Discard samples with fewer
              pseudo-boxes than this (avoids noisy gradients on empty scenes).
            - ``verbose`` (bool, False): Enable per-iter debug logging.

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
                     symmetric_contrastive=True,
                     conf_threshold=0.6,
                     use_class_specific_thresh=False,
                     class_thresholds=None,
                     source_loss_weight=1.0,
                     target_loss_weight=0.5,
                     contrastive_weight=0.1,
                     burn_in_iters=0,
                     min_pseudo_per_sample=0,
                     verbose=False,
                 ),
                 pretrained_ckpt=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        self.student = MODELS.build(copy.deepcopy(detector))
        self.teacher = MODELS.build(copy.deepcopy(detector))

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.student.train()
        self.teacher.train()

        self.mean_teacher_cfg = mean_teacher_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Load pretrained weights into student
        if pretrained_ckpt is not None:

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(pretrained_ckpt, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Load to student
            missing, unexpected = self.student.load_state_dict(state_dict, strict=False)
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Pretrained weights loaded from {pretrained_ckpt}: '
                f'{len(missing)} missing keys, {len(unexpected)} unexpected keys')
            if missing:
                logger.warning(f'Missing keys: {missing[:5]}')
                logger.warning(f'Unexpected keys: {unexpected[:5]}')


        # Initialise teacher from student (parameters + BN buffers).
        for (t_name, t_param), (s_name, s_param) in zip(
                self.teacher.named_parameters(),
                self.student.named_parameters()):
            assert t_name == s_name, \
                f'Teacher/student parameter name mismatch: {t_name} vs {s_name}'
            t_param.data.copy_(s_param.data)

        for (t_name, t_buf), (s_name, s_buf) in zip(
                self.teacher.named_buffers(),
                self.student.named_buffers()):
            assert t_name == s_name, \
                f'Teacher/student buffer name mismatch: {t_name} vs {s_name}'
            t_buf.copy_(s_buf)

        # Counters (initialised lazily to survive checkpoint resume)
        self._ema_update_count = 0
        self._last_param_norm = None
        self._train_iter = 0

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

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
            MMLogger.get_current_instance().debug(
                f'[EMA #{self._ema_update_count}] avg param norm: {avg_norm:.4f}{delta}')
            self._last_param_norm = avg_norm

    # ------------------------------------------------------------------
    # Teacher prediction filtering
    # ------------------------------------------------------------------

    def filter_teacher_predictions(self, teacher_pred):
        """Keep only high-confidence teacher detections for pseudo-labelling.

        Args:
            teacher_pred: Single-sample prediction object with ``pred_instances_3d``.

        Returns:
            A new prediction object with low-confidence boxes removed.
            ``bev_features`` is preserved unchanged.
        """
        verbose = self.mean_teacher_cfg.get('verbose', False)
        conf_threshold = self.mean_teacher_cfg.get('conf_threshold', 0.6)
        use_class_specific = self.mean_teacher_cfg.get('use_class_specific_thresh', False)

        scores = teacher_pred.pred_instances_3d.scores_3d
        labels = teacher_pred.pred_instances_3d.labels_3d
        bboxes = teacher_pred.pred_instances_3d.bboxes_3d

        if scores is None:
            return teacher_pred

        if use_class_specific:
            class_thresholds = self.mean_teacher_cfg.get('class_thresholds') or {}
            mask = torch.zeros_like(scores, dtype=torch.bool)
            for class_id, thresh in class_thresholds.items():
                mask |= (labels == class_id) & (scores >= thresh)
            # Default threshold for classes not listed.
            unspecified = torch.ones_like(scores, dtype=torch.bool)
            for class_id in class_thresholds:
                unspecified &= (labels != class_id)
            mask |= unspecified & (scores >= conf_threshold)
        else:
            mask = scores >= conf_threshold

        kept = mask.sum().item()
        if verbose:
            logger = MMLogger.get_current_instance()
            total = len(scores)
            logger.info(
                f'[Filter] kept {kept}/{total} '
                f'({(total - kept) / max(total, 1) * 100:.1f}% removed, '
                f'thresh={conf_threshold:.2f})')

        filtered_pred = copy.copy(teacher_pred)
        filtered_pred.pred_instances_3d = InstanceData(
            bboxes_3d=bboxes[mask],
            scores_3d=scores[mask],
            labels_3d=labels[mask],
        )
        if hasattr(teacher_pred, 'bev_features'):
            filtered_pred.bev_features = teacher_pred.bev_features

        return filtered_pred

    # ------------------------------------------------------------------
    # InfoNCE contrastive loss
    # ------------------------------------------------------------------

    def contrastive_loss(self, bev_s, bev_t, boxes_t_s, boxes_t,
                         tau=0.07, symmetric=True):
        """InfoNCE object-level BEV-feature consistency between student and teacher.

        Features are bilinearly sampled at the box centres.  Only boxes whose
        centres fall inside the valid BEV area in **both** feature maps are used,
        and they are aligned by box index before computing the loss.

        Args:
            bev_s: Student BEV feature map (strong aug) [C, H, W].
            bev_t: Teacher BEV feature map (weak aug) [C, H, W].
            boxes_t_s: Teacher boxes in strong-aug space [N, 7].
            boxes_t: Teacher boxes in weak-aug space [N, 7].
            tau: InfoNCE temperature.
            symmetric: Average s→t and t→s directions.

        Returns:
            Scalar loss.  Weighting by ``contrastive_weight`` is done by the caller.
        """
        device = bev_s.device
        if boxes_t is None or boxes_t.shape[0] == 0:
            return torch.tensor(0., device=device)

        pc_range = self.mean_teacher_cfg['point_cloud_range']
        min_x, min_y = pc_range[0], pc_range[1]
        max_x, max_y = pc_range[3], pc_range[4]

        def sample_bev(bev, boxes):
            """Bilinear-sample BEV [C,H,W] at box xy-centres.

            Returns (feats [N_valid, C], valid_idx [N_valid]) or (None, None).
            """
            norm_x = 2.0 * (boxes[:, 0] - min_x) / (max_x - min_x) - 1.0
            norm_y = 2.0 * (boxes[:, 1] - min_y) / (max_y - min_y) - 1.0
            valid_mask = (norm_x > -1.0) & (norm_x < 1.0) & \
                         (norm_y > -1.0) & (norm_y < 1.0)
            valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                return None, None
            # F.grid_sample: input [1,C,H,W], grid [1,1,N,2] → [1,C,1,N].
            grid = torch.stack(
                [norm_x[valid_idx], norm_y[valid_idx]], dim=-1
            ).unsqueeze(0).unsqueeze(0)
            feats = F.grid_sample(
                bev.unsqueeze(0), grid,
                mode='bilinear', align_corners=True, padding_mode='border'
            ).squeeze(0).squeeze(1).T  # [N_valid, C]
            return feats, valid_idx

        F_t, idx_t = sample_bev(bev_t, boxes_t)
        F_s, idx_s = sample_bev(bev_s, boxes_t_s)
        if F_t is None or F_s is None:
            return torch.tensor(0., device=device)

        # Align: build {original_box_index → row} maps then re-index by the
        # intersection so row i of F_s and F_t correspond to the same box.
        map_t = {v: i for i, v in enumerate(idx_t.tolist())}
        map_s = {v: i for i, v in enumerate(idx_s.tolist())}
        common = sorted(set(map_t) & set(map_s))
        if not common:
            return torch.tensor(0., device=device)

        rows_t = torch.tensor([map_t[k] for k in common], device=device)
        rows_s = torch.tensor([map_s[k] for k in common], device=device)
        F_t = F.normalize(F_t[rows_t], dim=1)
        F_s = F.normalize(F_s[rows_s], dim=1)

        N = F_s.shape[0]
        if N == 0:
            return torch.tensor(0., device=device)

        pos_mask = torch.eye(N, device=device)

        sim_st = torch.mm(F_s, F_t.T) / tau
        loss_st = -(pos_mask * F.log_softmax(sim_st, dim=1)).sum(dim=1).mean()

        if symmetric:
            sim_ts = torch.mm(F_t, F_s.T) / tau
            loss_ts = -(pos_mask * F.log_softmax(sim_ts, dim=1)).sum(dim=1).mean()
            return (loss_st + loss_ts) * 0.5

        return loss_st

    # ------------------------------------------------------------------
    # Box transform: weak-aug space → strong-aug space
    # ------------------------------------------------------------------

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

        origin = metainfo_strong.get('box_origin', (0.5, 0.5, 0.5))
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

        return LiDARInstance3DBoxes(
            boxes_tensor, box_dim=boxes_tensor.shape[-1], origin=origin)

    # ------------------------------------------------------------------
    # Pseudo-label construction
    # ------------------------------------------------------------------

    def _create_pseudo_labels(self, teacher_predictions, target_samples_strong):
        """Replace ``gt_instances_3d`` in each strong-aug sample with teacher predictions.

        Samples whose pseudo-box count is below ``min_pseudo_per_sample`` receive
        an empty GT so they contribute zero gradient rather than noisy gradient.

        Args:
            teacher_predictions: Filtered + box-transformed teacher preds.
            target_samples_strong: Strongly augmented samples (already preprocessed).

        Returns:
            Deep-copied samples with updated ``gt_instances_3d``.
        """
        verbose = self.mean_teacher_cfg.get('verbose', False)
        min_pseudo = self.mean_teacher_cfg.get('min_pseudo_per_sample', 0)
        logger = MMLogger.get_current_instance()

        pseudo_labeled_samples = copy.deepcopy(target_samples_strong)
        total_boxes = 0

        for pred, sample_strong in zip(teacher_predictions, pseudo_labeled_samples):
            instance = pred.pred_instances_3d
            boxes  = instance.bboxes_3d
            labels = instance.labels_3d
            scores = instance.scores_3d if hasattr(instance, 'scores_3d') else None

            if len(boxes) < min_pseudo:
                boxes  = boxes[:0]
                labels = labels[:0]
                scores = scores[:0] if scores is not None else None

            total_boxes += len(boxes)
            gt = InstanceData(bboxes_3d=boxes, labels_3d=labels)
            if scores is not None:
                gt.scores_3d = scores
            sample_strong.gt_instances_3d = gt

        if total_boxes == 0:
            logger.warning(
                '[PseudoLabels] No pseudo-labels kept '
                '(all filtered out or below min_pseudo_per_sample)')
        elif verbose:
            logger.info(f'[PseudoLabels] Total boxes assigned: {total_boxes}')

        return pseudo_labeled_samples

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def loss(self, batch_inputs_dict, batch_data_samples):
        """Compute the three-term Mean Teacher loss.

        Terms:
          1. Supervised source loss (student on labeled data).
          2. Pseudo-label loss (student on unlabeled target with teacher GT).
          3. InfoNCE BEV contrastive loss (student vs teacher features).

        Terms 2 and 3 are skipped for the first ``burn_in_iters`` iterations.
        """
        w_source = self.mean_teacher_cfg.get('source_loss_weight', 1.0)
        w_target = self.mean_teacher_cfg.get('target_loss_weight', 0.5)
        w_cont   = self.mean_teacher_cfg.get('contrastive_weight', 0.1)
        burn_in  = self.mean_teacher_cfg.get('burn_in_iters', 0)
        verbose  = self.mean_teacher_cfg.get('verbose', False)

        self._train_iter += 1
        past_burnin = self._train_iter > burn_in

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

        # ── TERM 1: Source supervised loss ───────────────────────────
        source_in, source_samp = preprocess(
            batch_inputs_dict['labeled'],
            batch_data_samples['labeled'])

        if w_source > 0:
            loss_source = self.student.loss(source_in, source_samp)
            for key, value in loss_source.items():
                if isinstance(value, (list, tuple)):
                    if value and isinstance(value[0], torch.Tensor):
                        value = sum(value)
                    else:
                        continue
                losses[f'{key}_source'] = value * w_source

        # ── Teacher forward on weak-augmented target ─────────────────
        target_weak  = batch_inputs_dict['unlabeled']['weak']
        target_strong = batch_inputs_dict['unlabeled']['strong']
        target_samp_weak   = batch_data_samples['unlabeled']['weak']
        target_samp_strong = batch_data_samples['unlabeled']['strong']

        target_weak_in, target_samp_weak_proc = preprocess(target_weak, target_samp_weak)

        self.teacher.train()
        with torch.no_grad():
            teacher_pred = self.teacher.predict(
                target_weak_in, target_samp_weak_proc,
                return_bev_features=True)

        filtered_preds = [self.filter_teacher_predictions(p) for p in teacher_pred]

        # Transform teacher boxes from weak to strong aug space.
        transformed_preds = copy.deepcopy(filtered_preds)
        for pred, samp in zip(transformed_preds, target_samp_strong):
            if len(pred.pred_instances_3d.bboxes_3d) > 0:
                pred.pred_instances_3d.bboxes_3d = self._transform_boxes(
                    pred.pred_instances_3d.bboxes_3d, samp.metainfo)

        # ── TERM 2: Pseudo-label loss ─────────────────────────────────
        target_strong_in, target_samp_strong_proc = preprocess(
            target_strong, target_samp_strong)

        pseudo_samples = self._create_pseudo_labels(
            transformed_preds, target_samp_strong_proc)

        if w_target > 0 and past_burnin:
            loss_target = self.student.loss(target_strong_in, pseudo_samples)
            for key, value in loss_target.items():
                if isinstance(value, (list, tuple)):
                    losses[f'{key}_target'] = [v * w_target for v in value]
                else:
                    losses[f'{key}_target'] = value * w_target

        # ── TERM 3: BEV contrastive loss ──────────────────────────────
        use_bev = self.mean_teacher_cfg.get('use_bev_consistency', False)

        if use_bev and w_cont > 0 and past_burnin:
            student_pred = self.student.predict(
                target_strong_in, target_samp_strong_proc,
                return_bev_features=True)

            loss_cont_total = torch.tensor(0., device=device)
            n_valid = 0

            for i in range(len(filtered_preds)):
                bev_s = getattr(student_pred[i], 'bev_features', None)
                bev_t = getattr(filtered_preds[i], 'bev_features', None)
                if bev_s is None or bev_t is None:
                    continue

                boxes_t   = filtered_preds[i].pred_instances_3d.bboxes_3d.tensor
                boxes_t_s = transformed_preds[i].pred_instances_3d.bboxes_3d.tensor
                if boxes_t.shape[0] == 0:
                    continue

                loss_i = self.contrastive_loss(
                    bev_s, bev_t, boxes_t_s, boxes_t,
                    tau=self.mean_teacher_cfg.get('tau', 0.07),
                    symmetric=self.mean_teacher_cfg.get('symmetric_contrastive', True),
                )
                loss_cont_total += loss_i
                n_valid += 1

            losses['loss_contrastive'] = (
                loss_cont_total / n_valid * w_cont if n_valid > 0
                else torch.tensor(0., device=device))
        else:
            losses['loss_contrastive'] = torch.tensor(0., device=device)

        # ── Debug summary ──────────────────────────────────────────────
        if verbose:
            source_loss_total = sum(v for k, v in losses.items()
                                    if '_source' in k and isinstance(v, torch.Tensor))
            target_loss_parts = [v for k, v in losses.items()
                                  if '_target' in k and isinstance(v, torch.Tensor)]
            target_loss_total = (sum(target_loss_parts) if target_loss_parts
                                 else torch.tensor(0., device=device))
            contrastive_loss_total = losses.get(
                'loss_contrastive', torch.tensor(0., device=device))
            logger.info(
                f'[iter {self._train_iter}] '
                f'source_loss={source_loss_total.item():.4f}  '
                f'target_pseudo_loss={target_loss_total.item():.4f}  '
                f'contrastive_loss={contrastive_loss_total.item():.4f}')

        return {k: v for k, v in losses.items() if not k.startswith('_')}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, batch_inputs, batch_data_samples,
                use_teacher=False, **kwargs):
        """Run inference with student (default) or teacher.

        The chosen subnet is temporarily set to eval mode so BN running stats
        are not updated from validation data.
        """
        subnet = self.teacher if use_teacher else self.student
        was_training = subnet.training
        subnet.eval()
        try:
            result = subnet.predict(batch_inputs, batch_data_samples, **kwargs)
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
