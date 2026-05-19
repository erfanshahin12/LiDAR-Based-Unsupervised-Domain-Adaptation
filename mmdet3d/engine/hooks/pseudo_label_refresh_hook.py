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
    Predictions are filtered using the detector's own ``filter_teacher_predictions``
    (which reads ``conf_threshold`` from ``mean_teacher_cfg``) and saved to disk as
    ``<work_dir>/ps_labels/ps_label_e{epoch}.pkl``.  The detector's in-memory
    ``pseudo_label_store`` is updated so the student can use the refreshed labels
    for the next training interval.

    On training start the hook checks for existing pkl files from prior runs and
    loads the most recent one whose epoch ≤ ``runner.epoch``.

    Args:
        interval (int): Refresh every this many epochs. Default: 4.
        update_at_epochs (Sequence[int]): Additionally refresh at these specific
            epochs.  Include 0 to refresh before the first training epoch.
            Default: (0,).
        ps_label_subdir (str): Subdirectory under ``work_dir`` for pkl files.
            Default: 'ps_labels'.
        ps_batch_size (int): Batch size for the teacher inference pass.
            Default: 4.
        ps_num_workers (int): Dataloader workers for the inference pass.
            Default: 4.
    """

    def __init__(
        self,
        interval: int = 4,
        update_at_epochs: Sequence[int] = (0,),
        ps_label_subdir: str = 'ps_labels',
        ps_batch_size: int = 4,
        ps_num_workers: int = 4,
    ) -> None:
        self.interval = interval
        self.update_at_epochs = list(update_at_epochs)
        self.ps_label_subdir = ps_label_subdir
        self.ps_batch_size = ps_batch_size
        self.ps_num_workers = ps_num_workers

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

    def _load_existing_pkl(self, runner: Runner, model) -> None:
        """Load the most recent ps_label pkl whose epoch <= runner.epoch."""
        ps_dir = os.path.join(runner.work_dir, self.ps_label_subdir)
        if not os.path.isdir(ps_dir):
            return
        pkls = glob.glob(os.path.join(ps_dir, 'ps_label_e*.pkl'))
        if not pkls:
            return
        # Find the latest pkl with epoch <= start_epoch.
        start_epoch = runner.epoch
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

        new_labels = self._run_teacher_inference(model, logger)

        # DDP: gather all-rank results on rank 0, then broadcast.
        rank, world_size = get_dist_info()
        if world_size > 1:
            all_labels = collect_results_cpu(
                list(new_labels.items()), len(self._ps_loader.dataset))
            if rank == 0:
                new_labels = dict(all_labels)
            else:
                new_labels = {}

        # Rank 0 writes the pkl; all ranks update the model store.
        ps_dir = os.path.join(runner.work_dir, self.ps_label_subdir)
        if rank == 0:
            os.makedirs(ps_dir, exist_ok=True)
            pkl_path = os.path.join(ps_dir, f'ps_label_e{epoch}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(new_labels, f)
            logger.info(
                f'[PseudoLabelRefreshHook] Wrote {len(new_labels)} entries '
                f'to {pkl_path}')

        if world_size > 1:
            # Broadcast the merged dict to non-zero ranks via temp file.
            import torch.distributed as dist
            if rank == 0:
                dist.broadcast_object_list([new_labels], src=0)
            else:
                container = [None]
                dist.broadcast_object_list(container, src=0)
                new_labels = container[0]

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

    def _log_ps_stats(self, new_labels: dict, logger) -> None:
        counts: list = [v['gt_labels'] for v in new_labels.values() if len(v['gt_labels']) > 0]
        if not counts:
            logger.warning('[PseudoLabelRefreshHook] No pseudo-boxes kept after filtering.')
            return
        all_labels = np.concatenate(counts)
        for cls_id in np.unique(all_labels):
            n = int((all_labels == cls_id).sum())
            logger.info(
                f'[PseudoLabelRefreshHook] class {int(cls_id)}: {n} pseudo-boxes')
