_base_ = ['./mt_pretrain_pointpillars_config.py']

# Checkpoint produced by the base pretrain config (24-epoch run).
load_from = './work_dirs/pretrain_pp_ros_26may/epoch_24.pth'

# Enable the IoU regression head on Anchor3DHead. mmengine deep-merges
# this into the parent model.bbox_head dict.
model = dict(
    bbox_head=dict(
        predict_iou=True,
        loss_iou_weight=1.0),
    # Keep inference cls-only; IoU is learned here but not used at test
    # time in this fine-tune run.
    test_cfg=dict(
        score_type='cls',
        score_weights=dict(iou=0.0, cls=1.0)))

# Freeze every parameter whose name does not contain 'conv_iou'.
custom_hooks = [
    dict(type='FreezeExceptHook', name_substrings=['conv_iou']),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=6)

optim_wrapper = dict(type='AmpOptimWrapper',
                     loss_scale='dynamic',
                     optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
                     accumulative_counts=1,
                     clip_grad=dict(max_norm=35, norm_type=2))
