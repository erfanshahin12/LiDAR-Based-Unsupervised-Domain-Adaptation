_base_ = ['./mt_pretrain_pointpillars_config.py']

# Checkpoint produced by the base pretrain config (24-epoch run).
load_from = './work_dirs/pretrain_pp_ros_26may/epoch_24.pth'

# Enable the IoU regression head on Anchor3DHead. mmengine deep-merges
# this into the parent model.bbox_head dict.
model = dict(
    # Add RoI feature extractor (matches middle_encoder output: 64ch, 504×504 BEV).
    roi_extractor_cfg=dict(
        in_channels=64,
        out_channels=256,
        roi_size=7,
        voxel_size=0.2,
        point_cloud_range=[-50.4, -50.4, -5.0, 50.4, 50.4, 3.0]),
    # Two-stage post-NMS IoU head trained jointly with conv_iou.
    bev_roi_iou_head_cfg=dict(hidden_dim=256),
    bbox_head=dict(
        predict_iou=True,
        loss_iou_weight=1.0),
    # Keep inference cls-only during finetune; IoU heads used only in MT training.
    test_cfg=dict(
        nms_thr=0.01,
        score_thr=0.1,
        nms_pre=1000,
        max_num=500,
        score_type='cls',
        score_weights=dict(iou=0.5, cls=0.5)))

# Train only the newly added components; freeze pretrained backbone/neck/anchor head.
custom_hooks = [
    dict(type='FreezeExceptHook',
         name_substrings=['conv_iou', 'roi_extractor', 'bev_roi_iou_head']),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=6)

optim_wrapper = dict(type='AmpOptimWrapper',
                     loss_scale='dynamic',
                     optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
                     accumulative_counts=1,
                     clip_grad=dict(max_norm=35, norm_type=2))
