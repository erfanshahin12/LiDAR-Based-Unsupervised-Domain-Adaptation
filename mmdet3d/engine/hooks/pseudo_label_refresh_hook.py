import glob
import os
import pickle
import re
from typing import List, Optional, Sequence

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
    Predictions are filtered using the detector's own ``filter_teacher_predictions``
    (which reads ``conf_threshold`` from ``mean_teacher_cfg``) and saved to disk as
    ``<work_dir>/ps_labels/ps_label_e{epoch}.pkl``.  The detector's in-memory
    ``pseudo_label_store`` is updated so the student can use the refreshed labels
    for the next training interval.

    On training start the hook checks for existing pkl files from prior runs and
    loads the most recent one whose epoch ≤ ``runner.epoch``.

    Adaptive confidence threshold
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``use_knee_threshold=True`` (default), each epoch the hook:

    1. Runs teacher inference with a broad threshold (``ps_min_score``, default
       0.05) to collect all candidate boxes.
    2. Finds the *knee* of the descending score curve (Kneedle method) — the
       score where the high-confidence tail separates from the dense low-score
       cluster.
    3. If the knee threshold would keep fewer than ``min_boxes_kept`` boxes,
       lowers the threshold to the score at which exactly ``min_boxes_kept``
       boxes are retained (box-count floor).  This prevents training starvation
       when teacher confidence collapses globally.
    4. Updates ``model.mean_teacher_cfg['conf_threshold']`` so the online
       training path (when the store is empty) uses the same threshold.

    The count floor is the critical design choice: a *score* floor (e.g. 0.30)
    acts as an aggressive count filter once the score distribution collapses,
    while a *count* floor guarantees sufficient gradient signal every epoch.

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
        use_knee_threshold (bool): If True, use the Kneedle-based adaptive
            threshold with ``min_boxes_kept`` count floor.  If False, use the
            fixed ``conf_threshold`` from ``mean_teacher_cfg``.  Default: True.
        min_boxes_kept (int): Minimum number of pseudo-boxes to retain per
            epoch.  If the knee threshold would keep fewer boxes, it is lowered
            until this count is satisfied.  Ignored when
            ``use_knee_threshold=False``.  Default: 2000.
        ps_min_score (float): Broad threshold used during the teacher inference
            pass when ``use_knee_threshold=True``.  Should be low enough to
            capture the full score distribution (≤ model's test_cfg.score_thr).
            Acts as the absolute lower bound on the adaptive threshold.
            Default: 0.05.
        use_top1_fallback (bool): If True, scenes with zero boxes above the
            adaptive threshold keep their top-1 candidate as a pseudo-label.
            Default: True.
        apply_dim_scaling (bool): If True, rescale pseudo-box dimensions
            (l, w, h) by ``dim_scale_factors`` after teacher inference.
            Use this to recalibrate source-domain anchor bias when the
            target domain has systematically different car sizes (e.g.
            nuScenes → KITTI).  z_center is adjusted to keep the box bottom
            at the same position after height scaling.  Default: False.
        dim_scale_factors (List[float]): Per-axis scale factors ``[s_l, s_w,
            s_h]`` applied to box dimensions when ``apply_dim_scaling=True``.
            Values < 1 shrink boxes; values > 1 enlarge them.  Ignored when
            ``apply_dim_scaling=False``.  Default: None.
    """

    def __init__(
        self,
        interval: int = 1,
        update_at_epochs: Sequence[int] = (0,),
        ps_label_subdir: str = 'ps_labels',
        ps_batch_size: int = 8,
        ps_num_workers: int = 6,
        use_knee_threshold: bool = True,
        min_boxes_kept: int = 2000,
        ps_min_score: float = 0.05,
        use_top1_fallback: bool = True,
        apply_dim_scaling: bool = False,
        dim_scale_factors: Optional[List[float]] = None,
    ) -> None:
        self.interval = interval
        self.update_at_epochs = list(update_at_epochs)
        self.ps_label_subdir = ps_label_subdir
        self.ps_batch_size = ps_batch_size
        self.ps_num_workers = ps_num_workers
        self.use_knee_threshold = use_knee_threshold
        self.min_boxes_kept = min_boxes_kept
        self.ps_min_score = ps_min_score
        self.use_top1_fallback = use_top1_fallback
        self.apply_dim_scaling = apply_dim_scaling
        if apply_dim_scaling:
            if dim_scale_factors is None or len(dim_scale_factors) != 3:
                raise ValueError(
                    'dim_scale_factors must be a list of 3 floats [s_l, s_w, s_h] '
                    'when apply_dim_scaling=True')
            if any(s <= 0 for s in dim_scale_factors):
                raise ValueError('All dim_scale_factors must be positive')
        self.dim_scale_factors = list(dim_scale_factors) if dim_scale_factors else None

        self._ps_loader: Optional[DataLoader] = None

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

        # For adaptive mode, temporarily lower conf_threshold so the inference
        # pass collects a broad candidate set; we apply the final threshold below.
        orig_conf_thr = model.mean_teacher_cfg.get('conf_threshold', 0.6)
        if self.use_knee_threshold:
            model.mean_teacher_cfg['conf_threshold'] = self.ps_min_score

        new_labels = self._run_teacher_inference(model, logger)

        if self.apply_dim_scaling:
            new_labels = self._apply_dim_scaling(new_labels, logger)

        # Restore original threshold — may be overwritten by adaptive logic below.
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

        # Compute and apply the adaptive threshold on rank 0 (has full dataset).
        # Single-rank training: rank == 0, so the condition is always satisfied.
        adaptive_thr = orig_conf_thr
        if self.use_knee_threshold and rank == 0:
            adaptive_thr = self._compute_adaptive_threshold(new_labels, logger)
            raw_labels = new_labels
            new_labels = self._filter_by_threshold(new_labels, adaptive_thr)
            if self.use_top1_fallback:
                new_labels = self._top1_fallback(
                    new_labels, raw_labels, logger)

        # Rank 0 writes the pkl; all ranks update the model store.
        ps_dir = self._ps_dir(runner)
        if rank == 0:
            os.makedirs(ps_dir, exist_ok=True)
            pkl_path = os.path.join(ps_dir, f'ps_label_e{epoch}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(new_labels, f)
            logger.info(
                f'[PseudoLabelRefreshHook] Wrote {len(new_labels)} entries '
                f'to {pkl_path}')

        if world_size > 1:
            # Broadcast the merged (and adaptive-filtered) dict to non-zero ranks.
            import torch.distributed as dist
            if rank == 0:
                dist.broadcast_object_list([new_labels, adaptive_thr], src=0)
            else:
                container = [None, None]
                dist.broadcast_object_list(container, src=0)
                new_labels, adaptive_thr = container[0], container[1]

        # Propagate the adaptive threshold so the online path uses the same cutoff.
        if self.use_knee_threshold:
            model.mean_teacher_cfg['conf_threshold'] = adaptive_thr

        model.set_pseudo_labels(new_labels)
        self._log_ps_stats(new_labels, logger)

    # ------------------------------------------------------------------
    # Teacher inference pass
    # ------------------------------------------------------------------

    def _run_teacher_inference(self, model, logger) -> dict:
        """Run teacher over the entire weak target split and build the store.

        Confidence filtering is delegated to ``model.filter_teacher_predictions``
        so the threshold is read from a single place (``mean_teacher_cfg['conf_threshold']``).
        """
        if getattr(model, 'mean_teacher_cfg', {}).get('use_dsnorm', False):
            from mmdet3d.models.layers.dsnorm import set_ds_target
            model.teacher.apply(set_ds_target)
        model.teacher.eval()
        new_labels: dict = {}
        total_pos = 0

        for batch in self._ps_loader:
            batch_inputs = batch['inputs']
            batch_data_samples = batch['data_samples']

            # Voxelize via the teacher's own data_preprocessor.
            with torch.no_grad():
                data = model.teacher.data_preprocessor(
                    {'inputs': batch_inputs,
                     'data_samples': batch_data_samples},
                    training=False)
                pred_list = model.teacher.predict(
                    data['inputs'], data['data_samples'])

            for data_sample, pred in zip(batch_data_samples, pred_list):
                key = data_sample.metainfo.get('lidar_path')
                if key is None:
                    continue

                # Re-use the same filtering logic as the online training path.
                pred = model.filter_teacher_predictions(pred)
                inst = pred.pred_instances_3d

                boxes_t  = inst.bboxes_3d.tensor.cpu().numpy()[:, :7]
                labels_t = inst.labels_3d.cpu().numpy()
                scores_t = inst.scores_3d.cpu().numpy()

                new_labels[key] = {
                    'gt_boxes':  boxes_t.astype(np.float32),
                    'gt_labels': labels_t.astype(np.int64),
                    'scores':    scores_t.astype(np.float32),
                }
                total_pos += len(labels_t)

        model.teacher.train()
        conf_threshold = model.mean_teacher_cfg.get('conf_threshold', 0.6)
        logger.info(
            f'[PseudoLabelRefreshHook] Generated {total_pos} positive '
            f'pseudo-boxes across {len(new_labels)} frames '
            f'(conf_threshold={conf_threshold})')
        return new_labels

    @staticmethod
    def _knee_threshold(scores: np.ndarray) -> float:
        """Kneedle: score at the elbow of the descending score curve.

        Finds the point of maximum perpendicular distance from the line
        connecting the first and last points of the normalised descending
        score curve.  For distributions with a dense low-score cluster and a
        sparse high-quality tail, this falls at the natural break between them.
        """
        sorted_scores = np.sort(scores)[::-1]
        n = len(sorted_scores)
        if n < 3:
            return float(sorted_scores[0]) if n else 0.0
        x = np.linspace(0, 1, n)
        s_min, s_max = sorted_scores[-1], sorted_scores[0]
        y = (sorted_scores - s_min) / (s_max - s_min + 1e-8)
        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])
        line_len = np.linalg.norm(p2 - p1)
        distances = np.abs(
            (p2[0] - p1[0]) * (p1[1] - y) - (p1[0] - x) * (p2[1] - p1[1])
        ) / (line_len + 1e-8)
        return float(sorted_scores[np.argmax(distances)])

    def _compute_adaptive_threshold(self, new_labels: dict, logger) -> float:
        """Knee threshold with a box-count floor.

        1. Compute the knee of the descending score curve.
        2. If fewer than ``min_boxes_kept`` candidates meet the knee threshold,
           lower it to the score of the ``min_boxes_kept``-th highest-scoring box
           (count floor).  This prevents training starvation when confidence
           collapses globally.
        3. Clamp below by ``ps_min_score`` as an absolute lower bound.
        """
        all_scores = []
        for v in new_labels.values():
            if len(v['scores']) > 0:
                all_scores.extend(v['scores'].tolist())
        if not all_scores:
            logger.warning(
                '[PseudoLabelRefreshHook] No candidates for adaptive threshold; '
                f'using ps_min_score={self.ps_min_score}')
            return self.ps_min_score
        all_scores_arr = np.array(all_scores, dtype=np.float32)

        # Step 1: quality threshold from the knee
        knee_thr = self._knee_threshold(all_scores_arr)
        n_above_knee = int((all_scores_arr >= knee_thr).sum())

        # Step 2: count floor — if knee keeps too few boxes, lower the threshold
        floor_triggered = False
        if n_above_knee < self.min_boxes_kept:
            floor_triggered = True
            sorted_desc = np.sort(all_scores_arr)[::-1]
            if len(sorted_desc) >= self.min_boxes_kept:
                knee_thr = float(sorted_desc[self.min_boxes_kept - 1])
            else:
                knee_thr = float(sorted_desc[-1])  # keep all

        # Step 3: absolute lower bound
        thr = max(knee_thr, self.ps_min_score)
        n_kept = int((all_scores_arr >= thr).sum())

        logger.info(
            f'[PseudoLabelRefreshHook] Adaptive threshold: '
            f'knee={self._knee_threshold(all_scores_arr):.4f}  →  '
            f'applied={thr:.4f}  '
            f'({"count floor" if floor_triggered else "knee"}, '
            f'kept={n_kept}/{len(all_scores_arr)},'
            f' min_boxes={self.min_boxes_kept})')
        return thr

    def _filter_by_threshold(self, new_labels: dict, thr: float) -> dict:
        """Return a copy of new_labels with boxes scoring below thr removed."""
        filtered: dict = {}
        for key, entry in new_labels.items():
            scores = entry['scores']
            keep = scores >= thr
            filtered[key] = {
                'gt_boxes':  entry['gt_boxes'][keep],
                'gt_labels': entry['gt_labels'][keep],
                'scores':    scores[keep],
            }
        return filtered

    @staticmethod
    def _top1_fallback(filtered: dict, candidates: dict, logger) -> dict:
        """For scenes with 0 boxes after global threshold, keep the top-1 candidate.

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
                    filtered[key] = {
                        'gt_boxes':  src['gt_boxes'][top1:top1 + 1],
                        'gt_labels': src['gt_labels'][top1:top1 + 1],
                        'scores':    src_scores[top1:top1 + 1],
                    }
                    n_fallback += 1
        if n_fallback:
            logger.info(
                f'[PseudoLabelRefreshHook] Top-1 fallback: {n_fallback} scenes '
                f'with 0 boxes above threshold received their top-1 candidate')
        return filtered

    def _apply_dim_scaling(self, new_labels: dict, logger) -> dict:
        """Rescale pseudo-box l/w/h by dim_scale_factors.

        z_center is shifted by (h_new - h_old) / 2 so the box bottom stays
        at the same position after height scaling (gravity-center convention).
        """
        sl, sw, sh = self.dim_scale_factors
        scaled = {}
        for key, entry in new_labels.items():
            boxes = entry['gt_boxes'].copy()  # (N, 7): x, y, z, l, w, h, yaw
            if len(boxes):
                h_old = boxes[:, 5].copy()
                boxes[:, 3] *= sl
                boxes[:, 4] *= sw
                boxes[:, 5] *= sh
                boxes[:, 2] += (boxes[:, 5] - h_old) / 2  # keep bottom fixed
            scaled[key] = {
                'gt_boxes':  boxes,
                'gt_labels': entry['gt_labels'],
                'scores':    entry['scores'],
            }
        logger.info(
            f'[PseudoLabelRefreshHook] Applied dim scaling '
            f'[s_l={sl:.3f}, s_w={sw:.3f}, s_h={sh:.3f}] to pseudo-boxes')
        return scaled

    def _log_ps_stats(self, new_labels: dict, logger) -> None:
        non_empty = [v['gt_labels'] for v in new_labels.values() if len(v['gt_labels']) > 0]
        if not non_empty:
            logger.warning('[PseudoLabelRefreshHook] No pseudo-boxes kept after filtering.')
            return
        n_scenes = len(new_labels)
        n_covered = len(non_empty)
        logger.info(
            f'[PseudoLabelRefreshHook] Scene coverage: '
            f'{n_covered}/{n_scenes} ({100 * n_covered / max(n_scenes, 1):.1f}%) '
            f'scenes have ≥1 pseudo-box')
        all_labels = np.concatenate(non_empty)
        for cls_id in np.unique(all_labels):
            n = int((all_labels == cls_id).sum())
            logger.info(
                f'[PseudoLabelRefreshHook] class {int(cls_id)}: {n} pseudo-boxes')
