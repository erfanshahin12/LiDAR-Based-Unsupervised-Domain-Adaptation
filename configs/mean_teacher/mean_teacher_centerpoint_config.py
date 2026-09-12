_base_ = ['../_base_/schedules/cyclic-20e.py',
          '../_base_/default_runtime.py']

source_dataset_type = 'NuScenesDataset'
source_data_root = 'data/nuscenes/'
ann_file_source = 'nuscenes_infos_train.pkl'
data_prefix_source = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
classes_nuscenes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
box_origin_source = (0.5, 0.5, 0.5)
metainfo_source = dict(classes=classes_nuscenes, origin=box_origin_source)

target_dataset_type = 'KittiDataset'
target_data_root = 'data/kitti/'
ann_file_target = 'kitti_infos_train.pkl'
ann_file_target_val = 'kitti_infos_val.pkl'
data_prefix_target = dict(pts='training/velodyne_reduced')
classes_kitti = ['Car']
box_origin_target = (0.5, 0.5, 0)
metainfo_target = dict(classes=classes_kitti, origin=box_origin_target)

pretrained_ckpt = './work_dirs/baseline_centerpoint_ros_10jun/epoch_20.pth'

# nuScenes-centered range — KITTI data enters via KittiToNuscenes transform, so it
# lives in nuScenes frame and must use this range (matches the pretrain checkpoint).
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=['Car'], origin=(0.5, 0.5, 0.5))
backend_args = None

# ── Pipelines ─────────────────────────────────────────────────────────────────

source_pipeline = [     # nuScenes — supervised source
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
         sweeps_num=5,
         use_dim=[0, 1, 2, 3],          # drop ring index → 4-ch (x,y,z,intensity)
         backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ClassRemapWithLabel',
         mapping={'car': 'Car'},
         class_names=classes_kitti,
         keep_unmapped=False),
    # Source object scale augmentation (stable-baseline ROS).
    dict(type='RandomObjectScaling', scale_range=[0.75, 1.0], class_names=['Car']),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925],
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=classes_kitti),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

target_weak_pipeline = [    # KITTI — teacher inference (no augmentation)
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=4, use_dim=4),
    dict(type='KittiToNuscenes'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points']),
]

target_strong_pipeline = [  # KITTI — student training (strong augmentation)
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=4, use_dim=4),
    dict(type='KittiToNuscenes'),

    # ── Hard-instance sampling (online, from nuScenes dbinfos) ──────────
    # Mirrors the PointPillars config.  Set sample_groups=None to disable.
    # dict(
    #     type='HardInstanceSampling',
    #     source_db_path=source_data_root + 'nuscenes_dbinfos_train.pkl',
    #     source_class_mapping=dict(Car='car'),
    #     db_path_prefix=source_data_root,
    #     sample_groups=dict(Car=5),
    #     use_pred_boxes_for_collision=False,  # no preds available in strong pipeline
    #     iou_thresh=0.3,  # applied between injected instances (inter-instance collision)
    #     carve=True,
    #     carve_extra_width=(1.0, 0.5, 0.5),
    #     size_normalize=dict(size_res=[-0.71, -0.35, -0.16]),
    #     class_names=classes_kitti,
    #     points_loader=dict(
    #         type='LoadPointsFromFile',
    #         coord_type='LIDAR',
    #         load_dim=5,
    #         use_dim=4)),

    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925],
         scale_ratio_range=[0.95, 1.05]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # PointsRangeFilter intentionally omitted: after rotating KITTI data into the
    # nuScenes frame and applying ±22.5° augmentation, a tight range filter can
    # eliminate all points and crash the CUDA voxelizer with gridDim=0.
    # The voxelizer's internal clip (same range) is the effective bound.
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

val_pipeline = [    # KITTI val — NusOnKittiMetric inverts predictions back to KITTI frame
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         backend_args=backend_args),
    dict(type='KittiToNuscenes'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

# ── Datasets ──────────────────────────────────────────────────────────────────

labeled_dataset = dict(
    type=source_dataset_type,
    data_root=source_data_root,
    ann_file=ann_file_source,
    data_prefix=data_prefix_source,
    pipeline=source_pipeline,
    metainfo=metainfo_source,
    modality=input_modality,
    box_type_3d='LiDAR',
    test_mode=False,
    with_velocity=False,
    backend_args=backend_args)

unlabeled_weak_dataset = dict(
    type=target_dataset_type,
    data_root=target_data_root,
    ann_file=ann_file_target,
    data_prefix=data_prefix_target,
    pipeline=target_weak_pipeline,
    metainfo=metainfo_target,
    modality=input_modality,
    box_type_3d='LiDAR',
    test_mode=False,
    load_eval_anns=False,
    filter_empty_gt=False,
    backend_args=backend_args)

unlabeled_strong_dataset = dict(
    type=target_dataset_type,
    data_root=target_data_root,
    ann_file=ann_file_target,
    data_prefix=data_prefix_target,
    pipeline=target_strong_pipeline,
    metainfo=metainfo_target,
    modality=input_modality,
    box_type_3d='LiDAR',
    test_mode=False,
    load_eval_anns=False,
    filter_empty_gt=False,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=8,
    num_workers=6,
    prefetch_factor=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='mean_teacher_collate_fn'),
    dataset=dict(
        type='MTCombinedDataset',
        labeled_dataset=labeled_dataset,
        unlabeled_weak_dataset=unlabeled_weak_dataset,
        unlabeled_strong_dataset=unlabeled_strong_dataset,
        metainfo=metainfo))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=target_dataset_type,
        data_root=target_data_root,
        ann_file=ann_file_target_val,
        data_prefix=data_prefix_target,
        pipeline=val_pipeline,
        metainfo=metainfo_target,
        modality=input_modality,
        box_type_3d='LiDAR',
        test_mode=False,
        filter_empty_gt=False,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='NusOnKittiMetric',
    ann_file=target_data_root + ann_file_target_val,
    metric='bbox',
    pcd_limit_range=point_cloud_range,
    label_mapping=None,
    default_cam_key='CAM2',
    ground_snap=dict(enabled=True, pctl=2.0, min_pts=25, margin=0.3, max_disp=0.0),
    backend_args=backend_args)

test_evaluator = val_evaluator

# ── Model ─────────────────────────────────────────────────────────────────────
# Architecture copied verbatim from mt_pretrain_centerpoint_config.py so the
# Car-only checkpoint (epoch_20.pth) loads with zero core-layer mismatches.
# Deviations from that pretrain config:
#   - type changed to CenterPointBEVRoI (adds bbox_head alias + return_bev support)
#   - roi_extractor_cfg added (in_channels=256 = SparseEncoder dense BEV channels)
#   - Wrapped in MeanTeacher3DDetector with DSNorm + cls-only pseudo-label settings

voxel_size = [0.1, 0.1, 0.2]

model = dict(
    type='MeanTeacher3DDetector',
    mean_teacher_cfg=dict(
        point_cloud_range=point_cloud_range,
        ema_momentum=0.99995,
        # EMA the teacher's DSNorm/BN buffers from the student so the teacher's
        # target-domain running stats track KITTI (standard Mean-Teacher). With
        # False the teacher normalised KITTI using frozen nuScenes stats while
        # its affine params were EMA'd toward the student's KITTI-batch stats —
        # a growing miscalibration. The EMA loop only touches float buffers.
        update_teacher_buffers=True,
        use_bev_consistency=True,
        tau=0.07,
        # contrastive thresholds (independent of pseudo-label conf_threshold).
        # fg_threshold: score above which a teacher pred is a contrastive fg anchor.
        # neg_threshold: score at-or-below which a pred is a background negative.
        fg_threshold=0.5,
        neg_threshold=0.25,
        # Warmup before contrastive loss fires: roi_extractor is randomly
        # initialised (not in the pretrained checkpoint) and would inject
        # noise into the backbone gradient until it learns meaningful BEV
        # features.  1 epoch = 3517 iters at this dataset / batch size.
        contrastive_warmup_iters=3517,  # 1 epoch
        source_loss_weight=1.0,
        target_loss_weight=0.5,
        contrastive_weight=0.0,
        verbose=True,
        eval_use_teacher=True,
        use_dsnorm=True,
        # cls-only filtering — no IoU head in the pretrained checkpoint
        conf_threshold=0.2,                 # placeholder for refresh hook to overwrite; Will be used by detector if ps-label store is empty
        hybrid_w_iou=0,
        iou_warmup_iters=0,
        iou_distill_weight=0,
        # Soft-quality weighting of pseudo-label losses (Stage 2). The quality
        # weight is the teacher CLS score only (no cls+iou hybrid). enable=False
        # ⇒ both heads fall back to hard targets (single master switch).
        pseudo_loss_cfg=dict(enable_soft_quality=True),
    ),
    pretrained_ckpt=pretrained_ckpt,

    detector=dict(
        type='CenterPointBEVRoI',
        data_preprocessor=dict(
            type='Det3DDataPreprocessor',
            voxel=True,
            voxel_layer=dict(
                max_num_points=10,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                max_voxels=(90000, 120000))),

        pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),

        pts_middle_encoder=dict(
            type='SparseEncoder',
            in_channels=4,
            sparse_shape=[41, 1024, 1024],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
            encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
            block_type='basicblock'),

        pts_backbone=dict(
            type='SECOND',
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False)),

        pts_neck=dict(
            type='SECONDFPN',
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True),

        # RoI extractor for BEV contrastive consistency loss.
        # in_channels=512 matches the SECONDFPN neck output (2 x 256) — the same
        # detection representation the CenterHead regresses from. The contrastive
        # loss now operates on neck features (not the pre-backbone SparseEncoder
        # map) so it actually shapes the features used for box regression.
        roi_extractor_cfg=dict(
            in_channels=512,
            out_channels=256,
            roi_size=7,
            voxel_size=voxel_size[0],
            point_cloud_range=point_cloud_range),

        pts_bbox_head=dict(
            type='CenterHead',
            in_channels=512,
            tasks=[dict(num_class=1, class_names=['Car'])],
            common_heads=dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
            share_conv_channel=64,
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                pc_range=point_cloud_range[:2],
                code_size=7),
            separate_head=dict(
                type='DCNSeparateHead',
                dcn_config=dict(
                    type='DCN',
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    padding=1,
                    groups=4),
                init_bias=-2.19,
                final_kernel=3),
            loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
            norm_bbox=True),

        train_cfg=dict(
            pts=dict(
                grid_size=[1024, 1024, 40],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                point_cloud_range=point_cloud_range,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),

        test_cfg=dict(
            pts=dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4],
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                pc_range=point_cloud_range[:2],
                nms_type='rotate',
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.01))))

# ── Hooks / schedule ──────────────────────────────────────────────────────────

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best=None),
    visualization=dict(type='Det3DVisualizationHook', draw=False))

custom_hooks = [
    dict(type='MeanTeacherHook', interval=1),
    dict(
        type='PseudoLabelRefreshHook',
        interval=3,
        update_at_epochs=(0,),
        ps_batch_size=8,
        ps_num_workers=6,
        # Per-scene keep-fraction: keep top 70% by CLS score per scene, then a
        # FIXED absolute CLS floor of 0.40 as a guard (cls-only ranking;
        # CenterPoint has no RoI-IoU head).
        # cls_thr=0.40 replaces the old adaptive cls_percentile (now 0/off)
        # min_pts=10 drops sparse phantom FPs.
        # Ordering: fraction-first, floor-as-guard (the hook's only mode).
        # See floor_before note if changing.
        # coverage_gate_drop=0.10 → skip store update if coverage drops >10%.
        keep_frac=0.7,
        iou_weight=0,
        iou_thr=0.0,
        cls_thr=0.40,
        cls_percentile=0,
        hybrid_thr=0.0,
        coverage_gate_drop=0.10,
        ps_min_score=0.05,
        use_top1_fallback=False,
        min_pts=10,
        hard_instance_quantile=25,
        # Augmentation-consistency geometry QC: keep only boxes that
        # reproduce under a BEV flip (teacher second view), 3D-IoU >= thresh.
        # No target-domain priors. enabled=False ⇒ skip the second pass.
        consistency_filter=dict(enabled=False, iou_thresh=0.5,
                                direction='horizontal'),
        # Stage 2 — Memory Ensemble & Voting (ST3D port): match each refresh
        # against the persistent memory bank; matched boxes keep the higher
        # score + reset counter, disappeared boxes age out (ignore@2, remove@3),
        # new boxes are added. enabled=False ⇒ write the filtered store as-is.
        memory_ensemble=dict(enabled=False, iou_thresh=0.1, ignore_thresh=2,
                             rm_thresh=3, weighted=False),
        # ── Ground-snap ───────────────────────────────────────────────────────────────
        # Snap each box bottom to the local ground estimated from its own footprint
        # points (association-free, label-free). Shared by the refresh hook (training
        # pseudo-labels) and NusOnKittiMetric (eval predictions) so train/test match.
        ground_snap=dict(enabled=True, pctl=2.0, min_pts=25, margin=0.3, max_disp=0.0)),
]

train_cfg = dict(max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    accumulative_counts=4,
    clip_grad=dict(max_norm=35, norm_type=2))

# ── LR schedule ───────────────────────────────────────────────────────────────
# Override the inherited cyclic-20e schedule, which COSINE-RAMPS lr UP from 1e-4
# to 1e-3 over epochs 0-8 (eta_min=lr*10). On this 10-epoch adaptation that ramp
# coincided exactly with the training collapse (lr reached ~8.7e-4 by epoch 6).
# Adapting from a pretrained checkpoint wants a brief warmup then a DECAY.
# Pattern matches configs/_base_/schedules/cosine.py.
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=10, end=10,
         by_epoch=True, eta_min=1e-6),
]
