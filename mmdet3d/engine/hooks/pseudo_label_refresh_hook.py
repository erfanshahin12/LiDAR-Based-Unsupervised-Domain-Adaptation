import glob
import os
import pickle
import re
from typing import Optional, Sequence

import numpy as np
import torch
from mmengine.dist import collect_results_cpu, get_dist_info
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from torch.utils.data import DataLoader

from mmdet3d.registry import HOOKS


def _ps_collate_fn(batch):
    """Collate Det3DDataset samples into a single batch dict."""
    inputs = {'points': [item['inputs']['points'] for item in batch]}
    data_samples = [item['data_samples'] for item in batch]
    return {'inputs': inputs, 'data_samples': data_samples}


@HOOKS.register_module()
class PseudoLabelRefreshHook(Hook):
    """Periodically refreshes the pseudo-label store on MeanTeacher3DDetector.

    Every ``interval`` epochs (and at the epochs listed in ``update_at_epochs``)
    the teacher model is run in eval mode over the full unlabeled target split.
    Predictions are filtered using a per-scene keep-fraction scheme and saved to
    disk as ``<work_dir>/ps_labels/ps_label_e{epoch}.pkl``.  The detector's
    in-memory ``pseudo_label_store`` is updated so the student can use the
    refreshed labels for the next training interval.

    On training start the hook checks for existing pkl files from prior runs and
    loads the most recent one whose epoch ≤ ``runner.epoch``.

    Keep-fraction filtering
    ~~~~~~~~~~~~~~~~~~~~~~~
    Per scene, rank all candidate boxes by
    ``rank = iou_weight * IoU + (1 - iou_weight) * CLS``
    (reduces to CLS when ``iou_weight=0``) and keep the top
    ``ceil(keep_frac * N)`` boxes.  ``ceil`` guarantees ≥1 box per non-empty
    scene; the fraction-based selection is epoch-invariant because it depends
    on rank, not absolute score magnitude.

    Optional hard guards (active only when > 0):
    ``rank >= hybrid_thr``, ``IoU >= iou_thr``, ``CLS >= cls_thr``.

    During the refresh inference pass, ``mean_teacher_cfg['conf_threshold']`` is
    temporarily lowered to ``ps_min_score`` so the detector's
    ``filter_teacher_predictions`` acts as a broad pre-floor, passing through the
    full candidate set (all NMS survivors scoring ≥ ps_min_score).  The
    keep-fraction filter is then applied here in the hook on the raw scores.
    After inference the original ``conf_threshold`` is restored so the online
    store-empty training path continues to use the configured cutoff.

    Coverage gate
    ~~~~~~~~~~~~~
    After each refresh, if the new scene coverage drops more than
    ``coverage_gate_drop`` (fraction) relative to the previously accepted
    coverage, the new labels are **discarded** and the existing store is kept.
    This prevents a degenerate teacher (real confidence collapse) from poisoning
    training.  The first refresh (epoch 0) is always accepted.

    Args:
        interval (int): Refresh every this many epochs. Default: 1.
        update_at_epochs (Sequence[int]): Additionally refresh at these specific
            epochs.  Include 0 to refresh before the first training epoch.
            Default: (0,).
        ps_label_subdir (str): Subdirectory under ``work_dir`` for pkl files.
            Default: 'ps_labels'.
        ps_batch_size (int): Batch size for the teacher inference pass.
            Default: 8.
        ps_num_workers (int): Dataloader workers for the inference pass.
            Default: 6.
        keep_frac (float): Fraction of candidate boxes to keep per scene
            (``ceil(keep_frac * N)`` top-ranked boxes).  1.0 = keep all.
            Default: 0.25.
        coverage_gate_drop (float): Maximum fractional drop in scene coverage
            before the refresh is rejected.  0.30 = reject if coverage falls
            >30% relative to the previous accepted value.  Default: 0.30.
        hybrid_thr (float): Optional minimum ranking-score guard applied after
            the keep-fraction cut.  0.0 = disabled.  Default: 0.0.
        iou_weight (float): IoU weight in the ranking score [0, 1].
            0 = CLS-only ranking.  Default: 0.0.
        iou_thr (float): Optional minimum raw RoI-IoU score guard.
            0.0 = disabled.  Default: 0.0.
        cls_thr (float): Optional minimum raw CLS score guard.
            0.0 = disabled.  Default: 0.0.
        ps_min_score (float): Broad threshold used during teacher inference to
            collect the candidate pool (temporarily overrides
            ``mean_teacher_cfg['conf_threshold']``).  Should be ≤
            ``test_cfg.score_thr`` (0.1 by default), acting as the absolute
            lower bound on candidates.  Default: 0.05.
        use_top1_fallback (bool): If True, scenes with zero boxes after the
            keep-fraction filter keep their top-1 candidate as a pseudo-label.
            Rarely needed since ``ceil`` already prevents empty scenes when N>0.
            Default: True.
        min_pts (int): Minimum number of LiDAR points that must fall inside a
            pseudo-label box for it to be kept.  Applied as a hard guard after
            the keep-fraction cut (same as ``cls_thr``/``iou_thr``).  Boxes
            with fewer than ``min_pts`` interior points are discarded regardless
            of their ranking score.  0 disables the filter.  Default: 5.
    """

    def __init__(
        self,
        interval: int = 1,
        update_at_epochs: Sequence[int] = (0,),
        ps_label_subdir: str = 'ps_labels',
        ps_batch_size: int = 8,
        ps_num_workers: int = 6,
        keep_frac: float = 0.25,
        coverage_gate_drop: float = 0.30,
        hybrid_thr: float = 0.0,
        iou_weight: float = 0.0,
        iou_thr: float = 0.0,
        cls_thr: float = 0.0,
        ps_min_score: float = 0.05,
        use_top1_fallback: bool = True,
        min_pts: int = 5,
    ) -> None:
        self.interval = interval
        self.update_at_epochs = list(update_at_epochs)
        self.ps_label_subdir = ps_label_subdir
        self.ps_batch_size = ps_batch_size
        self.ps_num_workers = ps_num_workers
        self.keep_frac = keep_frac
        self.coverage_gate_drop = coverage_gate_drop
        self.hybrid_thr = hybrid_thr
        self.iou_weight = iou_weight
        self.iou_thr = iou_thr
        self.cls_thr = cls_thr
        self.ps_min_score = ps_min_score
        self.use_top1_fallback = use_top1_fallback
        self.min_pts = min_pts

        self._ps_loader: Optional[DataLoader] = None
        self._prev_coverage: Optional[float] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unwrap_model(self, runner: Runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        return model

    def _should_refresh(self, epoch: int) -> bool:
        if epoch in self.update_at_epochs:
            return True
        if epoch != 0 and epoch % self.interval == 0:
            return True
        return False

    def _build_ps_loader(self, runner: Runner) -> DataLoader:
        """Build a DataLoader from the unlabeled_weak sub-dataset."""
        train_ds = runner.train_dataloader.dataset
        # MTCombinedDataset exposes the weak target dataset directly.
        weak_ds = getattr(train_ds, 'unlabeled_weak_dataset', None)
        if weak_ds is None:
            raise AttributeError(
                'PseudoLabelRefreshHook expects the training dataset to be '
                'an MTCombinedDataset with an `unlabeled_weak_dataset` attribute.')
        return DataLoader(
            weak_ds,
            batch_size=self.ps_batch_size,
            num_workers=self.ps_num_workers,
            collate_fn=_ps_collate_fn,
            drop_last=False,
            shuffle=False,
        )

    def _ps_dir(self, runner: Runner) -> str:
        """Return the ps_labels directory for the current run (under log_dir)."""
        return os.path.join(runner.log_dir, self.ps_label_subdir)

    def _load_existing_pkl(self, runner: Runner, model) -> None:
        """Load the most recent ps_label pkl whose epoch <= runner.epoch.

        On a fresh start (runner.epoch == 0) nothing is loaded — the hook will
        generate labels at epoch 0 as usual.  On resume (runner.epoch > 0) all
        timestamp subdirectories under work_dir are searched so that the pkl
        written by the original run is found even though the resumed run has a
        new log_dir (new timestamp).
        """
        start_epoch = runner.epoch
        if start_epoch == 0:
            return  # fresh start — let the epoch-0 refresh generate labels

        # Search all <work_dir>/<timestamp>/ps_labels/ dirs so resume finds the
        # pkl from the previous timestamp directory.
        search_pattern = os.path.join(
            runner.work_dir, '*', self.ps_label_subdir, 'ps_label_e*.pkl')
        pkls = glob.glob(search_pattern)
        if not pkls:
            return

        best_epoch, best_path = -1, None
        for path in pkls:
            m = re.search(r'ps_label_e(\d+)\.pkl', path)
            if m:
                e = int(m.group(1))
                if e <= start_epoch and e > best_epoch:
                    best_epoch, best_path = e, path
        if best_path is not None:
            logger = MMLogger.get_current_instance()
            logger.info(
                f'[PseudoLabelRefreshHook] Loading pseudo labels from {best_path}')
            model.load_pseudo_labels_from_pkl(best_path)

    @staticmethod
    def _scene_coverage(labels: dict):
        """Compute scene coverage statistics.

        Returns:
            (n_covered, n_scenes, fraction)
        """
        n_scenes = len(labels)
        n_covered = sum(1 for v in labels.values() if len(v['gt_labels']) > 0)
        frac = n_covered / max(n_scenes, 1)
        return n_covered, n_scenes, frac

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def before_train(self, runner: Runner) -> None:
        model = self._unwrap_model(runner)
        assert hasattr(model, 'pseudo_label_store'), (
            'PseudoLabelRefreshHook requires model.pseudo_label_store '
            '(MeanTeacher3DDetector).')
        assert hasattr(model, 'set_pseudo_labels'), (
            'PseudoLabelRefreshHook requires model.set_pseudo_labels().')
        assert hasattr(model, 'teacher'), (
            'PseudoLabelRefreshHook requires model.teacher.')

        self._ps_loader = self._build_ps_loader(runner)

        # Resume: load labels from the most recent pkl for this work_dir.
        self._load_existing_pkl(runner, model)

    def before_train_epoch(self, runner: Runner) -> None:
        epoch = runner.epoch
        if not self._should_refresh(epoch):
            return

        model = self._unwrap_model(runner)
        logger = MMLogger.get_current_instance()
        logger.info(
            f'[PseudoLabelRefreshHook] Refreshing pseudo-labels at epoch {epoch}')

        # Temporarily lower conf_threshold so filter_teacher_predictions acts as
        # a broad pre-floor (ps_min_score), passing raw candidates through.  The
        # keep-fraction filter applied below does the real quality cut.
        # try/finally guarantees the original threshold is restored even if
        # inference raises, so the online store-empty training path is never
        # left using ps_min_score as its quality cutoff.
        orig_conf_thr = model.mean_teacher_cfg.get('conf_threshold', 0.6)
        model.mean_teacher_cfg['conf_threshold'] = self.ps_min_score
        try:
            new_labels = self._run_teacher_inference(model, logger)
        finally:
            model.mean_teacher_cfg['conf_threshold'] = orig_conf_thr

        # DDP: gather all-rank results on rank 0, then broadcast.
        rank, world_size = get_dist_info()
        if world_size > 1:
            all_labels = collect_results_cpu(
                list(new_labels.items()), len(self._ps_loader.dataset))
            if rank == 0:
                new_labels = dict(all_labels)
            else:
                new_labels = {}

        # Apply keep-fraction filter, optional top-1 fallback, and coverage
        # gate on rank 0.
        accepted = True
        cov_new = 0.0
        if rank == 0:
            raw_labels = new_labels
            new_labels = self._apply_three_constraint_filter(new_labels, logger)
            if self.use_top1_fallback:
                new_labels = self._top1_fallback(new_labels, raw_labels, logger)

            _, _, cov_new = self._scene_coverage(new_labels)
            prev_str = (f'{self._prev_coverage:.3f}'
                        if self._prev_coverage is not None else 'N/A')
            logger.info(
                f'[PseudoLabelRefreshHook] Coverage: prev={prev_str} '
                f'new={cov_new:.3f}')
            if (self._prev_coverage is not None
                    and cov_new < (1.0 - self.coverage_gate_drop)
                    * self._prev_coverage):
                logger.warning(
                    f'[PseudoLabelRefreshHook] Coverage gate tripped: '
                    f'new coverage {cov_new:.3f} < '
                    f'{1.0 - self.coverage_gate_drop:.2f} × '
                    f'prev {self._prev_coverage:.3f}. '
                    f'Skipping store update — keeping existing pseudo-labels.')
                accepted = False

        # Rank 0 writes the pkl when accepted.
        ps_dir = self._ps_dir(runner)
        if rank == 0 and accepted:
            os.makedirs(ps_dir, exist_ok=True)
            pkl_path = os.path.join(ps_dir, f'ps_label_e{epoch}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(new_labels, f)
            logger.info(
                f'[PseudoLabelRefreshHook] Wrote {len(new_labels)} entries '
                f'to {pkl_path}')

        if world_size > 1:
            # Broadcast the filtered dict and the accepted flag to non-zero ranks.
            # If the gate tripped, payload is None so all ranks skip the update.
            import torch.distributed as dist
            if rank == 0:
                payload = new_labels if accepted else None
                dist.broadcast_object_list([payload, accepted], src=0)
            else:
                container = [None, None]
                dist.broadcast_object_list(container, src=0)
                new_labels = container[0]
                accepted = container[1]

        if accepted:
            model.set_pseudo_labels(new_labels)
            if rank == 0:
                self._prev_coverage = cov_new
            self._log_ps_stats(new_labels, logger)

    # ------------------------------------------------------------------
    # Teacher inference pass
    # ------------------------------------------------------------------

    def _run_teacher_inference(self, model, logger) -> dict:
        """Run teacher over the entire weak target split and build the store.

        ``mean_teacher_cfg['conf_threshold']`` is expected to already be lowered
        to ``ps_min_score`` by the caller so that ``filter_teacher_predictions``
        acts only as a broad pre-floor.  Raw ``iou_scores`` and ``cls_scores``
        are preserved in the output dict for the keep-fraction filter.
        """
        if getattr(model, 'mean_teacher_cfg', {}).get('use_dsnorm', False):
            from mmdet3d.models.layers.dsnorm import set_ds_target
            model.student.apply(set_ds_target)
        model.student.eval()
        new_labels: dict = {}
        total_pos = 0

        for batch in self._ps_loader:
            batch_inputs = batch['inputs']
            batch_data_samples = batch['data_samples']

            # Save raw xyz before voxelization for interior-point counting.
            # Each element is (M_i, C) on CPU; we only need the first 3 dims.
            if self.min_pts > 0:
                pts_raw = [
                    (p.numpy() if isinstance(p, torch.Tensor) else np.asarray(p))[:, :3]
                    for p in batch_inputs['points']
                ]
            else:
                pts_raw = [None] * len(batch_data_samples)

            # Voxelize via the student's own data_preprocessor.
            with torch.no_grad():
                data = model.student.data_preprocessor(
                    {'inputs': batch_inputs,
                     'data_samples': batch_data_samples},
                    training=False)
                pred_list = model.student.predict(
                    data['inputs'], data['data_samples'])

            for data_sample, pred, pts_xyz in zip(
                    batch_data_samples, pred_list, pts_raw):
                key = data_sample.metainfo.get('lidar_path')
                if key is None:
                    continue

                # Broad pre-floor via the detector's own filter
                # (conf_threshold == ps_min_score at this point).
                pred = model.filter_teacher_predictions(pred)
                inst = pred.pred_instances_3d

                boxes_t  = inst.bboxes_3d.tensor.cpu().numpy()[:, :7]
                labels_t = inst.labels_3d.cpu().numpy()
                scores_t = inst.scores_3d.cpu().numpy()
                iou_t    = getattr(inst, 'iou_scores_3d', None)
                cls_t    = getattr(inst, 'cls_scores_3d', None)

                # Interior point count per box (used by min_pts guard).
                if pts_xyz is not None and len(boxes_t) > 0 and len(pts_xyz) > 0:
                    from mmdet3d.structures.ops.box_np_ops import points_in_rbbox
                    in_box = points_in_rbbox(pts_xyz, boxes_t, z_axis=2,
                                             origin=(0.5, 0.5, 0))
                    pt_counts = in_box.sum(axis=0).astype(np.int32)
                else:
                    pt_counts = np.zeros(len(boxes_t), dtype=np.int32)

                new_labels[key] = {
                    'gt_boxes':  boxes_t.astype(np.float32),
                    'gt_labels': labels_t.astype(np.int64),
                    'scores':    scores_t.astype(np.float32),
                    'iou_scores': iou_t.cpu().numpy().astype(np.float32)
                                  if iou_t is not None else scores_t.astype(np.float32),
                    'cls_scores': cls_t.cpu().numpy().astype(np.float32)
                                  if cls_t is not None else None,
                    'pt_counts': pt_counts,
                }
                total_pos += len(labels_t)

        model.student.train()
        logger.info(
            f'[PseudoLabelRefreshHook] Collected {total_pos} candidate '
            f'pseudo-boxes across {len(new_labels)} frames '
            f'(pre-floor={self.ps_min_score})')
        return new_labels

    # ------------------------------------------------------------------
    # Keep-fraction filter
    # ------------------------------------------------------------------

    def _apply_three_constraint_filter(self, new_labels: dict, logger) -> dict:
        """Filter pseudo-boxes with per-scene keep-fraction selection.

        Primary selection: keep the top ``ceil(keep_frac * N)`` boxes per scene,
        ranked by
        ``rank = iou_weight * IoU + (1 - iou_weight) * CLS``
        (reduces to CLS when ``iou_weight=0``).  ``ceil`` guarantees ≥1 box
        whenever the scene has any candidate.

        Optional hard guards applied after the fraction cut (active when > 0):
          ``rank >= hybrid_thr``, ``IoU >= iou_thr``, ``CLS >= cls_thr``.

        ``CLS`` is ``cls_scores`` (raw ``scores_3d`` ranking score preserved by
        ``filter_teacher_predictions``).  ``IoU`` is ``iou_scores`` (post-NMS
        RoI IoU head score).  The stored ``scores`` field is set to the ranking
        score so ``_create_pseudo_labels_from_store`` and downstream paths see
        the pseudo-label quality score consistently.

        Args:
            new_labels: Candidate dict from ``_run_teacher_inference``.
            logger: MMLogger instance.

        Returns:
            Filtered dict with the same structure.
        """
        filtered: dict = {}
        total_before = 0
        total_after  = 0

        for key, entry in new_labels.items():
            cls_sc   = entry.get('cls_scores')
            iou_sc   = entry['iou_scores']
            pt_cnts  = entry.get('pt_counts')   # may be None if min_pts=0

            # Fallback: if raw cls_scores were not stored, use scores.
            if cls_sc is None:
                cls_sc = entry['scores']

            # Ranking score — cls-only when iou_weight=0 (default).
            if self.iou_weight > 0:
                rank = self.iou_weight * iou_sc + (1.0 - self.iou_weight) * cls_sc
            else:
                rank = cls_sc

            n = len(rank)
            total_before += n

            if n == 0:
                # Empty scene: pass through unchanged.
                filtered[key] = {
                    'gt_boxes':   entry['gt_boxes'],
                    'gt_labels':  entry['gt_labels'],
                    'scores':     rank,
                    'iou_scores': iou_sc,
                    'cls_scores': cls_sc,
                    'pt_counts':  pt_cnts,
                }
                continue

            # Per-scene keep-fraction: top ceil(keep_frac * N) by ranking score.
            if self.keep_frac is not None and self.keep_frac < 1.0:
                k = int(np.ceil(self.keep_frac * n))
                order = np.argsort(rank)[::-1]
                top_k_idx = order[:k]
                mask = np.zeros(n, dtype=bool)
                mask[top_k_idx] = True
            else:
                mask = np.ones(n, dtype=bool)

            # Optional hard guards (backward-compat; disabled when set to 0.0).
            if self.hybrid_thr > 0:
                mask = mask & (rank >= self.hybrid_thr)
            if self.iou_thr > 0:
                mask = mask & (iou_sc >= self.iou_thr)
            if self.cls_thr > 0:
                mask = mask & (cls_sc >= self.cls_thr)
            if self.min_pts > 0 and pt_cnts is not None:
                mask = mask & (pt_cnts >= self.min_pts)

            total_after += int(mask.sum())

            filtered[key] = {
                'gt_boxes':   entry['gt_boxes'][mask],
                'gt_labels':  entry['gt_labels'][mask],
                'scores':     rank[mask],    # ranking score for store
                'iou_scores': iou_sc[mask],
                'cls_scores': cls_sc[mask],
                'pt_counts':  pt_cnts[mask] if pt_cnts is not None else None,
            }

        logger.info(
            f'[PseudoLabelRefreshHook] Keep-fraction filter: '
            f'kept {total_after}/{total_before} boxes '
            f'(keep_frac={self.keep_frac}, iou_weight={self.iou_weight}, '
            f'hybrid_thr={self.hybrid_thr}, iou_thr={self.iou_thr}, '
            f'cls_thr={self.cls_thr}, min_pts={self.min_pts})')
        return filtered

    @staticmethod
    def _top1_fallback(filtered: dict, candidates: dict, logger) -> dict:
        """For scenes with 0 boxes after the three-constraint filter, keep the
        top-1 candidate ranked by hybrid score.

        Prevents empty-GT scenes from training the student toward background
        suppression. Every scene where the teacher found anything above
        ps_min_score receives at least one pseudo-label.
        """
        n_fallback = 0
        for key in filtered:
            if len(filtered[key]['scores']) == 0:
                src = candidates.get(key, {})
                src_scores = src.get('scores', np.zeros(0, dtype=np.float32))
                if len(src_scores) > 0:
                    top1 = int(np.argmax(src_scores))
                    src_cls = src.get('cls_scores')
                    src_iou = src.get('iou_scores', src_scores)
                    filtered[key] = {
                        'gt_boxes':   src['gt_boxes'][top1:top1 + 1],
                        'gt_labels':  src['gt_labels'][top1:top1 + 1],
                        'scores':     src_scores[top1:top1 + 1],
                        'iou_scores': src_iou[top1:top1 + 1],
                        'cls_scores': src_cls[top1:top1 + 1] if src_cls is not None else None,
                    }
                    n_fallback += 1
        if n_fallback:
            logger.info(
                f'[PseudoLabelRefreshHook] Top-1 fallback: {n_fallback} scenes '
                f'with 0 boxes above threshold received their top-1 candidate')
        return filtered

    def _log_ps_stats(self, new_labels: dict, logger) -> None:
        non_empty = [v['gt_labels'] for v in new_labels.values() if len(v['gt_labels']) > 0]
        if not non_empty:
            logger.warning('[PseudoLabelRefreshHook] No pseudo-boxes kept after filtering.')
            return
        n_scenes = len(new_labels)
        n_covered = len(non_empty)
        all_labels = np.concatenate(non_empty)
        n_total_surviving = len(all_labels)
        logger.info(
            f'[PseudoLabelRefreshHook] Pseudo-label store summary: '
            f'{n_total_surviving} boxes | scene coverage '
            f'{n_covered}/{n_scenes} ({100 * n_covered / max(n_scenes, 1):.1f}%)')
        for cls_id in np.unique(all_labels):
            n = int((all_labels == cls_id).sum())
            logger.info(
                f'[PseudoLabelRefreshHook] class {int(cls_id)}: {n} pseudo-boxes')
