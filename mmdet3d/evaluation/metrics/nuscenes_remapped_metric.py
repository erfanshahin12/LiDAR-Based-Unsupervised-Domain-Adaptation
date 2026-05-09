from os import path as osp
from typing import Dict, List, Optional, Union
import tempfile

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics.nuscenes_metric import (
    NuScenesMetric, output_to_nusc_box, lidar_nusc_box_to_global)


@METRICS.register_module()
class NuScenesRemappedMetric(NuScenesMetric):
    """NuScenes mAP for models trained with KITTI-style class remapping.

    Three incompatibilities between the standard :class:`NuScenesMetric` and a
    KITTI-aligned nuScenes pretraining setup are resolved here:

    1. **Label space mismatch**: ``dataset_meta['classes']`` contains the full
       10 nuScenes class list (from ``metainfo_source``), but after
       ``ClassRemapWithLabel`` the model outputs 3-class labels
       (0 = Car, 1 = Pedestrian, 2 = Cyclist).  Using the 10-class list would
       map label 0 → 'car', label 1 → 'truck', etc. — completely wrong.
       This class stores its own ``model_classes`` list and uses it throughout.

    2. **Missing velocity**: PointPillars outputs 7-DoF boxes; the nuScenes
       evaluator requires 9-DoF (vx, vy).  ``process()`` zero-pads on the fly.

    3. **Class name format**: Model uses KITTI-style capitalised names
       (``'Car'``, ``'Pedestrian'``, ``'Cyclist'``).  These are translated to
       nuScenes lowercase names (``'car'``, ``'pedestrian'``, ``'bicycle'``)
       for the submission JSON and for the ``DetectionConfig`` range filter.

    Note on the Cyclist→bicycle mapping
    ------------------------------------
    The training pipeline merges nuScenes ``bicycle`` *and* ``motorcycle`` GT
    into a single ``Cyclist`` label.  For evaluation we map ``Cyclist``
    predictions to ``bicycle`` (or whichever nuScenes name the caller chooses).
    Motorcycle GT is excluded from evaluation because the ``DetectionConfig`` is
    restricted to the three target classes.  This is a known limitation —
    motorcycle detection contributes to training loss but not to this mAP.

    Args:
        data_root (str): NuScenes dataset root path.
        ann_file (str): Path to the annotation pkl file.
        model_classes (List[str]): Model output class names **in label order**.
            E.g. ``['Car', 'Pedestrian', 'Cyclist']`` means label 0 = Car, etc.
        class_mapping (Dict[str, str]): Maps each model class name to a nuScenes
            category name.  Example::

                {'Car': 'car', 'Pedestrian': 'pedestrian', 'Cyclist': 'bicycle'}

        metric (str | List[str]): Metrics to evaluate. Defaults to ``'bbox'``.
        modality (dict): Sensor modality. Defaults to lidar-only.
        prefix (str, optional): Metric name prefix.
        format_only (bool): Write JSON without evaluating.
        jsonfile_prefix (str, optional): Path prefix for result JSON.
        eval_version (str): nuScenes eval config version.
            Defaults to ``'detection_cvpr_2019'``.
        collect_device (str): Distributed collection device.
        backend_args (dict, optional): mmengine backend arguments.
    """

    # Detection ranges from the official detection_cvpr_2019 config.
    _NUS_RANGES: Dict[str, int] = {
        'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50,
        'construction_vehicle': 50, 'pedestrian': 40, 'motorcycle': 40,
        'bicycle': 40, 'traffic_cone': 30, 'barrier': 30,
    }

    def __init__(
            self,
            data_root: str,
            ann_file: str,
            model_classes: List[str],
            class_mapping: Dict[str, str],
            metric: Union[str, List[str]] = 'bbox',
            modality: dict = dict(use_camera=False, use_lidar=True),
            prefix: Optional[str] = None,
            format_only: bool = False,
            jsonfile_prefix: Optional[str] = None,
            eval_version: str = 'detection_cvpr_2019',
            collect_device: str = 'cpu',
            backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'NuScenesRemapped metric'
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metric=metric,
            modality=modality,
            prefix=prefix,
            format_only=format_only,
            jsonfile_prefix=jsonfile_prefix,
            eval_version=eval_version,
            collect_device=collect_device,
            backend_args=backend_args)

        self.model_classes = model_classes
        self.class_mapping = class_mapping
        # NuScenes submission class names in the same order as model labels.
        self.nus_classes = [class_mapping[c] for c in model_classes]

        # Build a DetectionConfig restricted to our 3 target nuScenes classes,
        # replacing the parent's 10-class config.
        self.eval_detection_configs = self._make_eval_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_eval_config(self):
        """Return the standard full 10-class :class:`DetectionConfig`.

        ``DetectionConfig.__init__`` asserts that ``class_range`` contains
        exactly the 10 ``DETECTION_NAMES`` — a subset is rejected.  We keep
        the full config and compute our own subset mAP in
        :meth:`_evaluate_single` from ``label_aps``.
        """
        from nuscenes.eval.detection.config import config_factory
        return config_factory(self.eval_version)

    # ------------------------------------------------------------------
    # Overridden: process
    # ------------------------------------------------------------------

    def process(self, data_batch: dict, data_samples: list) -> None:
        """Collect predictions; zero-pad velocity when boxes are 7-DoF.

        Replaces the parent ``process()`` entirely so that:
        * We do not access ``pred_instances`` (2-D boxes), which lidar-only
          models (VoxelNet, PointPillars) do not populate.
        * Velocity padding happens before results are stored.
        """
        for data_sample in data_samples:
            result: dict = {}

            pred_3d = data_sample['pred_instances_3d']
            for attr in pred_3d:
                val = pred_3d[attr].to('cpu')
                # AMP produces FP16 tensors; upcast to FP32 so the nuScenes
                # evaluator (pyquaternion, numpy) gets clean float32 values.
                pred_3d[attr] = val.float() if isinstance(val, torch.Tensor) and val.is_floating_point() else val
            # bboxes_3d is a LiDARInstance3DBoxes, not a plain tensor — cast its
            # internal tensor separately.
            if pred_3d['bboxes_3d'].tensor.dtype != torch.float32:
                pred_3d['bboxes_3d'].tensor = pred_3d['bboxes_3d'].tensor.float()

            # Pad 7-DoF → 9-DoF with zero velocity so output_to_nusc_box works.
            bboxes = pred_3d['bboxes_3d']
            if bboxes.tensor.shape[-1] == 7:
                zeros = bboxes.tensor.new_zeros(bboxes.tensor.shape[0], 2)
                pred_3d['bboxes_3d'] = type(bboxes)(
                    torch.cat([bboxes.tensor, zeros], dim=-1),
                    box_dim=9, origin=(0.5, 0.5, 0.5))

            result['pred_instances_3d'] = pred_3d
            # Provide an empty dict so format_results' key iteration works
            # without triggering the '3d' filter on 'pred_instances'.
            result['pred_instances'] = {}
            result['sample_idx'] = data_sample['sample_idx']
            self.results.append(result)

    # ------------------------------------------------------------------
    # Overridden: compute_metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Use ``model_classes`` for box formatting, ``nus_classes`` for eval."""
        logger: MMLogger = MMLogger.get_current_instance()

        # 'version' may or may not be in dataset_meta depending on the pkl.
        self.version = self.dataset_meta.get('version', 'v1.0-trainval')
        self.data_infos = load(
            self.ann_file, backend_args=self.backend_args)['data_list']

        # Pass model_classes so _format_lidar_bbox receives the right label→name
        # mapping (KITTI-style names that it will translate internally).
        result_dict, tmp_dir = self.format_results(
            results, self.model_classes, self.jsonfile_prefix)

        metric_dict: Dict[str, float] = {}
        if self.format_only:
            logger.info(
                f'Results saved to {osp.basename(self.jsonfile_prefix)}')
        else:
            for metric in self.metrics:
                # Pass nus_classes so _evaluate_single reads the right keys
                # from metrics_summary.json (which uses nuScenes class names).
                ap_dict = self.nus_evaluate(
                    result_dict,
                    classes=self.nus_classes,
                    metric=metric,
                    logger=logger)
                metric_dict.update(ap_dict)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    # ------------------------------------------------------------------
    # Overridden: box formatting
    # ------------------------------------------------------------------

    def _format_lidar_bbox(
            self,
            results: List[dict],
            sample_idx_list: List[int],
            classes: Optional[List[str]] = None,
            jsonfile_prefix: Optional[str] = None) -> str:
        """Write predictions to a nuScenes submission JSON.

        ``classes`` is ``self.model_classes`` (KITTI-style names coming from
        ``compute_metrics``).  This method translates them to nuScenes names
        before writing and before the range-distance filter in
        ``lidar_nusc_box_to_global``.

        Zero velocity is already baked into the boxes by ``process()``.
        Attribute is set to the class default (no velocity to infer from).
        """
        # Translate KITTI names → nuScenes names, preserving label order.
        nus_classes = [self.class_mapping[c] for c in classes]

        nusc_annos: dict = {}
        print('Start to convert detection format...')

        for i, det in enumerate(mmengine.track_iter_progress(results)):
            boxes, _ = output_to_nusc_box(det)
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos[sample_idx]['token']

            # Rotate + translate to global frame; filter by per-class distance.
            # Uses nus_classes so the range-filter dict lookup succeeds.
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_idx], boxes,
                nus_classes, self.eval_detection_configs)

            annos = []
            for box in boxes:
                nus_name = nus_classes[box.label]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),   # [0.0, 0.0]
                    detection_name=nus_name,
                    detection_score=box.score,
                    attribute_name=self.DefaultAttribute[nus_name])
                annos.append(nusc_anno)

            # Always write an entry, even if empty, so every sample token
            # appears in the JSON (required by NuScenesEval token check).
            nusc_annos[sample_token] = annos

        nusc_submissions = {'meta': self.modality, 'results': nusc_annos}
        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print(f'Results written to {res_path}')
        mmengine.dump(nusc_submissions, res_path)
        return res_path

    # ------------------------------------------------------------------
    # Overridden: evaluation
    # ------------------------------------------------------------------

    def _evaluate_single(
            self,
            result_path: str,
            classes: Optional[List[str]] = None,
            result_name: str = 'pred_instances_3d') -> Dict[str, float]:
        """Run ``NuScenesEval`` and report metrics for our 3 target classes.

        The official evaluator uses the full 10-class config internally; its
        ``mean_ap`` therefore averages over all 10 classes and is misleading
        for our 3-class setup.  We override it by computing the mean ourselves
        from ``label_aps`` for only our target nuScenes classes.  TP errors are
        likewise averaged over the target classes only.

        ``classes`` is ``self.nus_classes`` (nuScenes lowercase names).
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }

        # --- diagnostic: surface token mismatches before NuScenesEval raises ---
        import json
        from nuscenes.eval.detection.evaluate import load_prediction, load_gt
        from nuscenes.eval.detection.data_classes import DetectionBox
        with open(result_path) as _f:
            _pred_data = json.load(_f)
        _pred_tokens = set(_pred_data['results'].keys())
        _gt_boxes = load_gt(nusc, eval_set_map[self.version], DetectionBox, verbose=False)
        _gt_tokens = set(_gt_boxes.sample_tokens)
        _logger = MMLogger.get_current_instance()
        _logger.info(f'[NuScenesRemappedMetric] pred tokens: {len(_pred_tokens)}, '
                     f'gt tokens: {len(_gt_tokens)}')
        if _pred_tokens != _gt_tokens:
            _logger.warning(f'  in pred but not gt ({len(_pred_tokens - _gt_tokens)}): '
                            f'{list(_pred_tokens - _gt_tokens)[:5]}')
            _logger.warning(f'  in gt but not pred ({len(_gt_tokens - _pred_tokens)}): '
                            f'{list(_gt_tokens - _pred_tokens)[:5]}')
        # -----------------------------------------------------------------------

        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        metrics = mmengine.load(osp.join(output_dir, 'metrics_summary.json'))
        detail: Dict[str, float] = {}
        prefix = f'{result_name}_NuScenes'

        # Unique nuScenes target names (e.g. ['car', 'pedestrian', 'bicycle']).
        target = list(dict.fromkeys(classes or self.nus_classes))

        # Per-class AP at each distance threshold + TP errors.
        per_class_mean_ap: List[float] = []
        per_err: Dict[str, List[float]] = {}

        for name in target:
            if name not in metrics['label_aps']:
                continue

            # AP at each distance threshold
            aps = []
            for dist_th, ap in metrics['label_aps'][name].items():
                detail[f'{prefix}/{name}_AP_dist_{dist_th}'] = float(f'{ap:.4f}')
                aps.append(ap)
            per_class_mean_ap.append(float(np.mean(aps)))

            # TP errors per class
            for err_name, val in metrics['label_tp_errors'][name].items():
                detail[f'{prefix}/{name}_{err_name}'] = float(f'{val:.4f}')
                per_err.setdefault(err_name, []).append(val)

        # Mean AP over our 3 target classes (replaces the 10-class official mAP).
        our_map = float(np.mean(per_class_mean_ap)) if per_class_mean_ap else 0.0
        detail[f'{prefix}/mAP'] = float(f'{our_map:.4f}')

        # Mean TP errors over our target classes (replaces the 10-class means).
        for err_name, vals in per_err.items():
            mapped = self.ErrNameMapping.get(err_name, err_name)
            detail[f'{prefix}/{mapped}'] = float(f'{np.mean(vals):.4f}')

        # NDS is still the official 10-class score; flag it so it's not
        # mistaken for a 3-class figure.
        detail[f'{prefix}/NDS_10cls'] = float(f'{metrics["nd_score"]:.4f}')
        return detail
