# DEPRECATED — this config implemented a separate finetune phase that added
# the post-NMS BEVRoIIoUHead on top of a frozen pretrained detector.
# That two-stage quality-head design has been superseded by the new
# per-anchor quality head trained from scratch in the pretrain phase (see
# mt_pretrain_iouhead_pointpillars_config.py).  This file is kept as a
# historical reference only; it should NOT be used for new experiments.
_base_ = ['./mt_pretrain_pointpillars_config.py']

# Checkpoint produced by the base pretrain config (24-epoch run).
load_from = './work_dirs/pretrain_pp_ros_26may/epoch_24.pth'

# The new pretrain config enables predict_iou from the start; no separate
# finetune phase is needed.  To enable the per-anchor IoU head in a
# continued training run, add the keys below to mt_pretrain_pointpillars_config.py
# or use mt_pretrain_iouhead_pointpillars_config.py instead.
model = dict(
    bbox_head=dict(
        predict_iou=True,
        loss_iou_weight=1.0),
    test_cfg=dict(
        nms_thr=0.01,
        score_thr=0.1,
        nms_pre=1000,
        max_num=500,
        score_type='cls',
        score_weights=dict(iou=0.5, cls=0.5)))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=6)

optim_wrapper = dict(type='AmpOptimWrapper',
                     loss_scale='dynamic',
                     optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
                     accumulative_counts=1,
                     clip_grad=dict(max_norm=35, norm_type=2))
