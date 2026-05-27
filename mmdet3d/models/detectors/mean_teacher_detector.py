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

        # Periodic pseudo-label store: {lidar_path -> {'gt_boxes': (N,7), 'gt_labels': (N,), 'scores': (N,)}}
        # Populated by PseudoLabelRefreshHook; empty dict means fall back to per-iteration teacher predict.
        self.pseudo_label_store: dict = {}

        logger = MMLogger.get_current_instance()
        logger.info(
            f'ema_momentum={mean_teacher_cfg.get("ema_momentum", 0.999)}, '
            f'update_teacher_buffers={mean_teacher_cfg.get("update_teacher_buffers", False)}, '
            f'eval_use_teacher={mean_teacher_cfg.get("eval_use_teacher", True)}, '
            f'burn_in_iters={mean_teacher_cfg.get("burn_in_iters", 0)} ')

    # ------------------------------------------------------------------
    # Load pretrained weights to both student and teacher at initialization
    # ------------------------------------------------------------------

    def init_weights(self):
        super().init_weights()

        if self.pretrained_ckpt is not None:

            # Keys allowed to be missing from a pretrain checkpoint.  Hybrid thresholding's
            # IoU head (``bbox_head.conv_iou.*``) is added at adaptation time
            # and therefore expected to be absent from pretrain w/o IoU head.
            hybrid_iou_prefix = 'bbox_head.conv_iou.'

            def _check_loaded_keys(missing, unexpected):
                # Guard against silent partial loads in core layers (e.g. num_classes mismatch).
                critical_prefixes = (
                    'bbox_head.', 'voxel_encoder.', 'middle_encoder.', 'backbone.', 'neck.')
                critical_missing    = [
                    k for k in missing
                    if k.startswith(critical_prefixes)
                    and not k.startswith(hybrid_iou_prefix)
                ]
                critical_unexpected = [k for k in unexpected if k.startswith(critical_prefixes)]
                if critical_missing or critical_unexpected:
                    raise RuntimeError(
                        f'Pretrained checkpoint architectural mismatch — refusing to train '
                        f'with randomly-initialised core layers.\n'
                        f'  missing    (model has, ckpt lacks): {critical_missing}\n'
                        f'  unexpected (ckpt has, model lacks): {critical_unexpected}\n'
                        f'Check that num_classes, anchor sizes, and head architecture match '
                        f'between this config and {self.pretrained_ckpt}.')
                # Surface (but don't fail on) IoU-head keys so it's
                # visible in the log that random init was used.
                hybrid_keys = [k for k in missing if k.startswith(hybrid_iou_prefix)]
                if hybrid_keys:
                    MMLogger.get_current_instance().info(
                        f'  [Hybrid IoU] {len(hybrid_keys)} IoU-head keys absent from '
                        f'checkpoint — randomly initialised at adaptation time: '
                        f'{hybrid_keys}')

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(self.pretrained_ckpt, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            logger = MMLogger.get_current_instance()

            # Load to student
            missing, unexpected = self.student.load_state_dict(state_dict, strict=False)
            logger.info(
                f'Student: pretrained weights loaded from {self.pretrained_ckpt}: '
                f'{len(missing)} missing keys, {len(unexpected)} unexpected keys')
            _check_loaded_keys(missing, unexpected)

            # Load to teacher
            missing, unexpected = self.teacher.load_state_dict(state_dict, strict=False)
            logger.info(
                f'Teacher: pretrained weights loaded from {self.pretrained_ckpt}: '
                f'{len(missing)} missing keys, {len(unexpected)} unexpected keys')
            _check_loaded_keys(missing, unexpected)

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
            MMLogger.get_current_instance().info(
                f'[EMA #{self._ema_update_count}] avg param norm: {avg_norm:.4f}{delta}')
            self._last_param_norm = avg_norm

    # ------------------------------------------------------------------
    # Periodic pseudo-label store management
    # ------------------------------------------------------------------

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
        """Build pseudo-labeled strong samples directly from the cached store.

        Equivalent to ``_create_pseudo_labels`` but reads boxes from
        ``pseudo_label_store`` instead of from teacher prediction objects,
        skipping the intermediate wrapper step.  Canonical-frame boxes are
        transformed into the strong-augmentation frame via ``_transform_boxes``.
        Samples missing from the store contribute zero gradient (empty GT).
        """
        verbose = self.mean_teacher_cfg.get('verbose', False)
        logger = MMLogger.get_current_instance()

        pseudo_labeled_samples = copy.deepcopy(target_samp_strong)
        total_boxes = 0

        for samp_strong, pseudo_samp in zip(target_samp_strong, pseudo_labeled_samples):
            key = samp_strong.metainfo.get('lidar_path')
            entry = self.pseudo_label_store.get(key)

            if entry is not None and len(entry['gt_boxes']) > 0:
                boxes_t = torch.from_numpy(
                    entry['gt_boxes'].astype(np.float32)).to(device)
                labels_t = torch.from_numpy(
                    entry['gt_labels'].astype(np.int64)).to(device)
                scores_t = torch.from_numpy(
                    entry['scores'].astype(np.float32)).to(device)

                boxes_3d = LiDARInstance3DBoxes(boxes_t)
                boxes_3d = self._transform_boxes(boxes_3d, samp_strong.metainfo)
            else:
                boxes_3d = LiDARInstance3DBoxes(
                    torch.zeros(0, 7, device=device))
                labels_t = torch.zeros(0, dtype=torch.long, device=device)
                scores_t = torch.zeros(0, device=device)

            total_boxes += len(boxes_3d)
            gt = InstanceData(bboxes_3d=boxes_3d, labels_3d=labels_t)
            gt.scores_3d = scores_t
            pseudo_samp.gt_instances_3d = gt

        if verbose:
            logger.info(
                f'[PseudoStats/store] total_boxes={total_boxes} '
                f'across {len(pseudo_labeled_samples)} samples')
        if total_boxes == 0:
            logger.warning(
                '[PseudoLabels] No pseudo-labels from store '
                '(all frames missing or empty)')

        return pseudo_labeled_samples

    # ------------------------------------------------------------------
    # Teacher prediction filtering
    # ------------------------------------------------------------------

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
        filtered_pred.pred_instances_3d = InstanceData(
            bboxes_3d=bboxes[mask],
            scores_3d=hybrid[mask],
            labels_3d=labels[mask],
        )
        if hasattr(teacher_pred, 'bev_features'):
            filtered_pred.bev_features = teacher_pred.bev_features

        return filtered_pred

    # ------------------------------------------------------------------
    # Iter-0 sanity diagnostic
    # ------------------------------------------------------------------

    def _log_output_sanity(self, target_weak_in, target_samp_weak):
        """One-shot diagnostic at iter 1: trace the full prediction chain for teacher and student.

        Both models are evaluated on the same weak-aug target input. Because
        PillarFeatureNet (legacy=True) modifies the voxel tensor in-place when
        computing cluster/voxel-centre offsets, every forward call receives a
        fresh deep copy of the input so later calls are not corrupted.

        Sections:
          1. Input stats (voxel count, xyz ranges, intensity range).
          2. Raw cls-score histogram before NMS (eval mode, isolated copy).
          3. Post-NMS predictions in eval and train mode (reveals BN instability).
          Teacher and student should produce identical numbers at iter 1 —
          any divergence here is a bug in initialization.
        """
        logger = MMLogger.get_current_instance()
        thresholds = [0.1, 0.3, 0.5]

        def _fresh_copy(inp):
            """Return a deep copy of the preprocessed input dict.

            PillarFeatureNet with legacy=True subtracts voxel-centre offsets
            directly into the voxel tensor (in-place view assignment).  Every
            forward call must receive its own copy so that successive calls in
            this diagnostic do not see accumulated coordinate shifts.
            """
            return copy.deepcopy(inp)

        # ── 1. Input statistics — read before any forward modifies the tensor ─
        try:
            voxels = target_weak_in['voxels']['voxels']
            coors  = target_weak_in['voxels']['coors']
            n_pts  = target_weak_in['voxels']['num_points']
            total_voxels  = voxels.shape[0]
            total_points  = int(n_pts.sum().item())
            pts_flat = []
            for vi in range(min(total_voxels, 500)):
                nv = int(n_pts[vi].item())
                if nv > 0:
                    pts_flat.append(voxels[vi, :nv, :])
            if pts_flat:
                pts = torch.cat(pts_flat, dim=0).cpu()
                intensity_str = (
                    f'  intensity=[{pts[:, 3].min():.1f}, {pts[:, 3].max():.1f}]'
                    if pts.shape[1] >= 4 else '')
                logger.info(
                    f'[Sanity iter-1] voxelized input: '
                    f'total_voxels={total_voxels}  total_points={total_points}  '
                    f'batch_indices={coors[:, 0].unique().cpu().tolist()}  '
                    f'x=[{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}]  '
                    f'y=[{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]  '
                    f'z=[{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}]'
                    + intensity_str)
            else:
                logger.warning('[Sanity iter-1] NO voxels (empty input!)')
        except Exception as exc:
            logger.warning(f'[Sanity iter-1] could not inspect voxels: {exc}')

        # ── 2. Raw cls-score histogram (eval mode, isolated copy per call) ────
        def log_cls_scores(model, label):
            original_mode = model.training
            try:
                model.eval()
                with torch.no_grad():
                    x = model.extract_feat(_fresh_copy(target_weak_in), return_bev=False)
                    cls_scores_raw = []
                    if hasattr(model, 'bbox_head'):
                        head = model.bbox_head
                        if hasattr(head, 'forward_single'):
                            for feat in x:
                                cls, *_ = head.forward_single(feat)
                                cls_scores_raw.append(cls.sigmoid().cpu())
                        else:
                            outs = head(x)
                            if isinstance(outs, (list, tuple)) and len(outs) > 0:
                                if isinstance(outs[0], (list, tuple)):
                                    for t in outs[0]:
                                        if isinstance(t, torch.Tensor):
                                            cls_scores_raw.append(t.sigmoid().cpu())
                    if cls_scores_raw:
                        all_cls = torch.cat([c.flatten() for c in cls_scores_raw])
                        pct_str = '  '.join(
                            f'p{p}={torch.quantile(all_cls, p/100).item():.4f}'
                            for p in [50, 75, 90, 95, 99])
                        above_thresh = {t: int((all_cls >= t).sum()) for t in thresholds}
                        logger.info(
                            f'[Sanity {label}] raw cls scores (pre-NMS, {len(all_cls)} anchors): '
                            f'{pct_str}  '
                            + '  '.join(f'>={t:.1f}:{n}' for t, n in above_thresh.items()))
                    else:
                        logger.warning(f'[Sanity {label}] could not extract raw cls scores')
            except Exception as exc:
                logger.warning(f'[Sanity {label}] raw cls score extraction failed: {exc}')
            finally:
                model.train(original_mode)

        # ── 3. Post-NMS predictions (isolated copy per mode per model) ────────
        def log_predictions(model, label):
            original_mode = model.training
            for mode_label, use_eval in (('eval', True), ('train', False)):
                model.eval() if use_eval else model.train()
                with torch.no_grad():
                    try:
                        preds = model.predict(
                            _fresh_copy(target_weak_in),
                            copy.deepcopy(target_samp_weak))
                    except Exception as exc:
                        logger.warning(f'[Sanity {label}:{mode_label}] predict raised: {exc}')
                        continue
                score_lists = [
                    getattr(p.pred_instances_3d, 'scores_3d', None).detach().cpu()
                    for p in preds
                    if getattr(p, 'pred_instances_3d', None) is not None
                    and getattr(p.pred_instances_3d, 'scores_3d', None) is not None
                ]
                if score_lists:
                    scores = torch.cat(score_lists)
                    count_str = '  '.join(
                        f'>={t:.1f}:{int((scores >= t).sum())}' for t in thresholds)
                    logger.info(
                        f'[Sanity {label}:{mode_label}] '
                        f'total_returned={len(scores)}  {count_str}')
                else:
                    logger.info(f'[Sanity {label}:{mode_label}] zero predictions returned')
            model.train(original_mode)

        logger.info('[Sanity iter-1] --- TEACHER ---')
        log_cls_scores(self.teacher, 'teacher')
        log_predictions(self.teacher, 'teacher')

        logger.info('[Sanity iter-1] --- STUDENT ---')
        log_cls_scores(self.student, 'student')
        log_predictions(self.student, 'student')

        logger.info(
            '[Sanity iter-1] INTERPRETATION: '
            'total_voxels~0 → points outside range; check KittiToNuscenes + PointsRangeFilter. '
            'intensity not in [0,255] → KittiToNuscenes ×255 scaling not applied. '
            'cls scores all low → pretrained weights not loading or wrong coordinate frame. '
            'cls scores OK but total_returned=0 → NMS/score_thr too aggressive. '
            'eval >> train → BN train-mode instability (expected). '
            'teacher != student at iter-1 → initialization bug.')

    # ------------------------------------------------------------------
    # InfoNCE contrastive loss
    # ------------------------------------------------------------------

    def contrastive_loss(self, bev_s, bev_t, boxes_t_s, boxes_t,
                         tau=0.07):
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

        # Symmetric InfoNCE: average s→t and t→s directions
        sim_st = torch.mm(F_s, F_t.T) / tau
        loss_st = -(pos_mask * F.log_softmax(sim_st, dim=1)).sum(dim=1).mean()

        sim_ts = torch.mm(F_t, F_s.T) / tau
        loss_ts = -(pos_mask * F.log_softmax(sim_ts, dim=1)).sum(dim=1).mean()
        
        return (loss_st + loss_ts) * 0.5

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

    # ------------------------------------------------------------------
    # Pseudo-label construction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

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

        # Iter-1 sanity check: verify teacher and student produce identical outputs
        # on the same weak-augmented input.
        if self._train_iter == 1:
            self._log_output_sanity(target_weak_in, target_samp_weak)

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
            conf_threshold = self.mean_teacher_cfg.get('conf_threshold', 0.6)
            total_before = sum(len(p.pred_instances_3d.scores_3d) for p in teacher_pred)
            total_after  = sum(len(p.pred_instances_3d.scores_3d) for p in filtered_preds)
            per_sample   = [len(p.pred_instances_3d.scores_3d) for p in filtered_preds]
            logger.info(
                f'[Filter] kept {total_after}/{total_before} boxes across {len(filtered_preds)} samples '
                f'(thresh={conf_threshold:.2f})  per-sample: {per_sample}')

        # Transform teacher boxes from weak to strong aug space.
        # transformed_preds is used for the BEV contrastive loss (boxes_t_s).
        transformed_preds = copy.deepcopy(filtered_preds)

        for pred, samp in zip(transformed_preds, target_samp_strong):
            if len(pred.pred_instances_3d.bboxes_3d) > 0:
                pred.pred_instances_3d.bboxes_3d = self._transform_boxes(
                    pred.pred_instances_3d.bboxes_3d, samp.metainfo)

        # ── TERMS 2 + 3: Pseudo-label loss and BEV contrastive loss ──────────
        # Pseudo-label source: use the periodically-refreshed store when
        # populated (PseudoLabelRefreshHook), otherwise fall back to the
        # current-iteration teacher predictions (online Mean Teacher).
        # When BEV contrastive is active we need the student's BEV feature map
        # from the target-strong pass.  Running student.predict() after
        # student.loss() would be a *third* forward with a live computation
        # graph, blowing GPU memory.  Instead, extract features once and feed
        # them to both bbox_head.loss (pseudo-label) and contrastive_loss.

        if self.pseudo_label_store:         # create pseudo-labels from the cached pseudo-label store
            pseudo_samples = self._create_pseudo_labels_from_store(
                target_samp_strong, device)

        else:                               # create pseudo-labels from the current teacher predictions
            pseudo_samples = self._create_pseudo_labels(
                transformed_preds, target_samp_strong)

        use_bev = self.mean_teacher_cfg.get('use_bev_consistency', False)

        _ds_switch(self.student, 'target')
        if use_bev and w_cont > 0:
            # Single forward: get neck features + BEV map in one pass.
            x_strong, bev_features_student = self.student.extract_feat(
                target_strong_in, return_bev=True)
            if w_target > 0:
                loss_target = self.student.bbox_head.loss(x_strong, pseudo_samples)
                for key, value in loss_target.items():
                    if isinstance(value, (list, tuple)):
                        losses[f'{key}_target'] = [v * w_target for v in value]
                    else:
                        losses[f'{key}_target'] = value * w_target
        else:
            bev_features_student = None
            if w_target > 0:
                loss_target = self.student.loss(target_strong_in, pseudo_samples)
                for key, value in loss_target.items():
                    if isinstance(value, (list, tuple)):
                        losses[f'{key}_target'] = [v * w_target for v in value]
                    else:
                        losses[f'{key}_target'] = value * w_target

        # ── TERM 3: BEV contrastive loss ──────────────────────────────
        if use_bev and w_cont > 0:
            loss_cont_total = torch.tensor(0., device=device)
            n_valid = 0

            for i in range(len(filtered_preds)):
                bev_s = bev_features_student[i] if bev_features_student is not None else None
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
                )
                loss_cont_total += loss_i
                n_valid += 1

            losses['loss_contrastive'] = (
                loss_cont_total / n_valid * w_cont if n_valid > 0
                else torch.tensor(0., device=device))
        else:
            losses['loss_contrastive'] = torch.tensor(0., device=device)

        # ── Debug summary ──────────────────────────────────────────────
        # if verbose:
        #     source_loss_total = sum(v for k, v in losses.items()
        #                             if '_source' in k and isinstance(v, torch.Tensor))
        #     target_loss_parts = [v for k, v in losses.items()
        #                           if '_target' in k and isinstance(v, torch.Tensor)]
        #     target_loss_total = (sum(target_loss_parts) if target_loss_parts
        #                          else torch.tensor(0., device=device))
        #     contrastive_loss_total = losses.get(
        #         'loss_contrastive', torch.tensor(0., device=device))
        #     logger.info(
        #         f'[iter {self._train_iter}] '
        #         f'source_loss={source_loss_total.item():.4f}  '
        #         f'target_pseudo_loss={target_loss_total.item():.4f}  '
        #         f'contrastive_loss={contrastive_loss_total.item():.4f}')

        return {k: v for k, v in losses.items() if not k.startswith('_')}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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
            # The MT wrapper has no data_preprocessor of its own, so mmengine's
            # val_step only device-transfers the points.  Voxelize here via the
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
