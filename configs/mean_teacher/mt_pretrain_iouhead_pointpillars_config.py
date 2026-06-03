# nuScenes→Car pretrain with per-anchor 3D-IoU quality head.
# Adds predict_iou=True + balanced bin sampling to mt_pretrain_pointpillars_config.py.
# roi_extractor_cfg is omitted intentionally: the RoI extractor is trained from
# scratch during the teacher-student phase, not on source data.
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
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500,
        score_type='hybrid',          # 'cls' | 'iou' | 'hybrid'
        score_weights=dict(cls=0.5, iou=0.5),
    ),
)
