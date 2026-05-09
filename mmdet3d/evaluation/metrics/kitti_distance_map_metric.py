from typing import Dict, List, Optional

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS


@METRICS.register_module()
class KittiDistanceMAPMetric(BaseMetric):
    """NuScenes-style distance-based mAP evaluated on KITTI data.

    Instead of the IoU-based 3D/BEV AP used by the KITTI metric, this class
    matches predictions to GT by BEV centre distance — the same criterion used
    by the official nuScenes detection metric.  This lets you compare a
    nuScenes-pretrained model's Car detection quality with nuScenes numbers
    without needing the nuScenes dataset format (tokens, global frame, etc.).

    Algorithm (per class, per distance threshold ``d``):
    1. Sort all predictions in the dataset by score (descending).
    2. For each prediction: match to the nearest unmatched GT whose BEV centre
       distance is < ``d``.  Mark as TP on match, FP otherwise.
    3. Compute the precision-recall curve; AP = 101-point interpolation.
    4. mAP = mean AP over all thresholds.

    Args:
        classes (list[str]): Class names to evaluate (label 0, 1, … in order).
        dist_thresholds (list[float]): BEV centre-distance match thresholds in
            metres.  Default mirrors nuScenes: [0.5, 1.0, 2.0, 4.0].
        collect_device (str): Device used for distributed result collection.
    """

    def __init__(
            self,
            classes: List[str],
            dist_thresholds: List[float] = (0.5, 1.0, 2.0, 4.0),
            collect_device: str = 'cpu') -> None:
        super().__init__(collect_device=collect_device)
        self.classes = classes
        self.dist_thresholds = list(dist_thresholds)

    # ------------------------------------------------------------------
    # Collect per-sample predictions and GT
    # ------------------------------------------------------------------

    def process(self, data_batch: dict, data_samples: list) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_instances_3d']
            gt   = data_sample['gt_instances_3d']

            self.results.append(dict(
                pred_bev_xy=pred['bboxes_3d'].tensor.cpu().numpy()[:, :2],
                pred_scores=pred['scores_3d'].cpu().numpy(),
                pred_labels=pred['labels_3d'].cpu().numpy(),
                gt_bev_xy  =gt['bboxes_3d'].tensor.cpu().numpy()[:, :2],
                gt_labels  =gt['labels_3d'].cpu().numpy(),
            ))

    # ------------------------------------------------------------------
    # Aggregate and report
    # ------------------------------------------------------------------

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()
        metric_dict: Dict[str, float] = {}

        for cls_idx, cls_name in enumerate(self.classes):
            aps = []
            for threshold in self.dist_thresholds:
                ap = self._compute_ap(results, cls_idx, threshold)
                aps.append(ap)
                metric_dict[f'{cls_name}/AP@{threshold}m'] = round(ap, 4)

            mean_ap = float(np.mean(aps))
            metric_dict[f'{cls_name}/mAP'] = round(mean_ap, 4)
            logger.info(
                f'{cls_name} mAP: {mean_ap:.4f}  '
                f'(per-threshold: '
                + ', '.join(
                    f'{t}m={a:.4f}'
                    for t, a in zip(self.dist_thresholds, aps))
                + ')')

        overall_map = float(np.mean([
            v for k, v in metric_dict.items() if k.endswith('/mAP')]))
        metric_dict['mAP'] = round(overall_map, 4)
        logger.info(f'Overall mAP: {overall_map:.4f}')

        return metric_dict

    # ------------------------------------------------------------------
    # Per-class, per-threshold AP
    # ------------------------------------------------------------------

    def _compute_ap(
            self,
            results: List[dict],
            cls_idx: int,
            threshold: float) -> float:
        """Standard 101-point interpolated AP with distance-based matching."""
        all_preds: List[tuple] = []   # (score, is_tp)
        total_gt = 0

        for sample in results:
            gt_mask   = sample['gt_labels'] == cls_idx
            pred_mask = sample['pred_labels'] == cls_idx

            gt_bev    = sample['gt_bev_xy'][gt_mask]
            pred_bev  = sample['pred_bev_xy'][pred_mask]
            scores    = sample['pred_scores'][pred_mask]

            total_gt += int(gt_mask.sum())

            if len(pred_bev) == 0:
                continue

            # Sort predictions by score descending (greedy matching).
            order    = np.argsort(-scores)
            pred_bev = pred_bev[order]
            scores   = scores[order]

            matched_gt = np.zeros(len(gt_bev), dtype=bool)

            for i in range(len(pred_bev)):
                if len(gt_bev) == 0:
                    all_preds.append((scores[i], False))
                    continue

                dists = np.linalg.norm(pred_bev[i] - gt_bev, axis=1)
                dists[matched_gt] = np.inf   # exclude already-matched GT

                best = int(np.argmin(dists))
                if dists[best] < threshold:
                    matched_gt[best] = True
                    all_preds.append((scores[i], True))
                else:
                    all_preds.append((scores[i], False))

        if total_gt == 0 or len(all_preds) == 0:
            return 0.0

        # Sort globally by score descending.
        all_preds.sort(key=lambda x: -x[0])
        is_tp = np.array([tp for _, tp in all_preds], dtype=np.float32)

        tp_cum = np.cumsum(is_tp)
        fp_cum = np.cumsum(1 - is_tp)

        recalls    = tp_cum / total_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-10)

        # 101-point interpolated AP over [0, 1] recall.
        ap = 0.0
        for r in np.linspace(0.0, 1.0, 101):
            mask = recalls >= r
            ap += (precisions[mask].max() if mask.any() else 0.0) / 101

        return float(ap)
