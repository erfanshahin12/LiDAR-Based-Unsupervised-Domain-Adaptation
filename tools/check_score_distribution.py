"""
Check the raw prediction score distribution of a checkpoint on KITTI.

Usage (from ~/mmdetection3d/):
    python tools/check_score_distribution.py \
        configs/mean_teacher/test_kitti_nuspretrained_kitti_metric.py \
        work_dirs/baseline_pointpillars_5may/epoch_24.pth \
        [--num-samples 200]
"""
import argparse
import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Check prediction score distribution on KITTI')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='number of dataset samples to run (default: 200)')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = '/tmp/check_score_dist'

    # Restrict dataset size for speed
    cfg.test_dataloader.dataset.indices = list(range(args.num_samples))
    cfg.test_dataloader.batch_size = 4
    cfg.test_dataloader.num_workers = 2

    # Set score_thr=0 so ALL predictions survive NMS — we want the full distribution
    cfg.model.test_cfg.score_thr = 0.0
    cfg.model.test_cfg.nms_pre = 1000
    cfg.model.test_cfg.max_num = 1000

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)

    model = runner.model.cuda().eval()
    dataloader = runner.test_dataloader

    all_scores = []
    n_empty = 0

    with torch.no_grad():
        for data in dataloader:
            results = model.test_step(data)
            for r in results:
                scores = r.pred_instances_3d.scores_3d.cpu().numpy()
                if len(scores) == 0:
                    n_empty += 1
                all_scores.extend(scores.tolist())

    all_scores = np.array(all_scores)

    print(f'\n{"="*50}')
    print(f'Samples evaluated : {args.num_samples}')
    print(f'Samples with zero predictions: {n_empty}')
    print(f'Total raw predictions (score_thr=0): {len(all_scores)}')

    if len(all_scores) == 0:
        print('No predictions at all — model may not be detecting anything.')
        return

    print(f'\nScore statistics:')
    print(f'  max    : {all_scores.max():.4f}')
    print(f'  p90    : {np.percentile(all_scores, 90):.4f}')
    print(f'  p75    : {np.percentile(all_scores, 75):.4f}')
    print(f'  median : {np.median(all_scores):.4f}')
    print(f'  mean   : {all_scores.mean():.4f}')
    print(f'  min    : {all_scores.min():.4f}')

    print(f'\nPredictions surviving each threshold:')
    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    for t in thresholds:
        count = (all_scores >= t).sum()
        pct = 100.0 * count / len(all_scores)
        per_sample = count / args.num_samples
        print(f'  score >= {t:.2f}: {count:5d}  ({pct:5.1f}% of preds,  {per_sample:.1f} per sample)')

    print(f'\n  --> Recommended conf_threshold: just below the score where')
    print(f'      per-sample count drops to near 0.')
    print('='*50)


if __name__ == '__main__':
    main()
