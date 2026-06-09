# nuScenes→Car pretrain with per-anchor 3D-IoU quality head.
# Adds predict_iou=True + balanced bin sampling + rotation-aware RoI IoU MLP.
_base_ = ['./mt_pretrain_pointpillars_config.py']

model = dict(
    bbox_head=dict(
        predict_iou=True,
        loss_iou_weight=1.0,
        # 256 non-positive anchors/image split equally across bins; positives always added.
        iou_sample_cfg=dict(
            num_per_img=256,
            bins=[0.0, 0.1, 0.3, 0.5, 1.0],
        ),
        # RoI IoU MLP: rotation-aware 7×7 affine crop on the BEV scatter map.
        # BEV map: 64-channel PointPillarsScatter output at 504×504 for the
        # nuScenes range [-50.40, -50.40, -5, 50.40, 50.40, 3].
        roi_extractor_cfg=dict(
            in_channels=64,
            out_channels=128,
            roi_size=7,
            voxel_size=0.2,
            point_cloud_range=[-50.40, -50.40, -5, 50.40, 50.40, 3],
        ),
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500,
        # score_type is cls by default; Should change code to use hybrid or iou scoring if preferred.
    ),
)
