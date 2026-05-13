_base_ = ['../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py']

# ── Dataset identities ───────────────────────────────────────────────────────
source_dataset_type = 'NuScenesDataset'
source_data_root    = 'data/nuscenes/'
ann_file_source     = 'nuscenes_infos_train.pkl'
data_prefix_source  = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
classes_nuscenes    = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                       'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
metainfo_source     = dict(classes=classes_nuscenes, origin=(0.5, 0.5, 0.5))

target_dataset_type = 'KittiDataset'
target_data_root    = 'data/kitti/'
ann_file_target     = 'kitti_infos_train.pkl'
ann_file_target_val = 'kitti_infos_val.pkl'
data_prefix_target  = dict(pts='training/velodyne_reduced')
classes_kitti       = ['Car']
metainfo_target     = dict(classes=classes_kitti, origin=(0.5, 0.5, 0))

pretrained_ckpt = './work_dirs/baseline_pointpillars_5may/epoch_24.pth'

# ── Geometry ─────────────────────────────────────────────────────────────────
point_cloud_range = [-50.40, -50.40, -5, 50.40, 50.40, 3]
voxel_size        = [0.2, 0.2, 8]
input_modality    = dict(use_lidar=True, use_camera=False)
backend_args      = None

# Shared metainfo at dataset-wrapper level (single class: Car)
metainfo = dict(classes=['Car'], origin=(0.5, 0.5, 0.5))

import numpy as np
x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
vx, vy, vz = voxel_size
output_shape = [
    int(np.round((y_max - y_min) / vy)),
    int(np.round((x_max - x_min) / vx))]

# ── Pipelines ─────────────────────────────────────────────────────────────────

# NuScenes source — supervised with real GT
source_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR', load_dim=5, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ClassRemapWithLabel',
         mapping={'car': 'Car'},
         class_names=classes_kitti,
         keep_unmapped=False),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925],
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=classes_kitti),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

# KITTI target — single pipeline shared by teacher and student.
# Augmentations are intentionally light so teacher pseudo-boxes remain
# accurate without requiring coordinate transforms between aug spaces.
# To add stronger student-side augmentation later: split into a
# separate strong pipeline and apply box transforms in _create_pseudo_labels.
target_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='KittiToNuscenes'),           # rotate to nuScenes frame (X-right, Y-fwd)
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.0873, 0.0873],       # ±5° — light enough that boxes stay coherent
         scale_ratio_range=[0.98, 1.02]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points']),
]

# KITTI val — KittiToNuscenes applied so predictions come out in nuScenes
# frame; NusOnKittiMetric inverts them back before KITTI eval.
val_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR', load_dim=4, use_dim=4,
         backend_args=backend_args),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True, with_label_3d=True,
         backend_args=backend_args),
    dict(type='KittiToNuscenes'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

# ── Sub-datasets ──────────────────────────────────────────────────────────────

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

# Both weak and strong unlabeled datasets share the same target_pipeline.
# The detector consumes only the 'weak' side; the 'strong' side is collated
# but unused (minor CPU overhead, kept to avoid changes to MTCombinedDataset).
unlabeled_weak_dataset = dict(
    type=target_dataset_type,
    data_root=target_data_root,
    ann_file=ann_file_target,
    data_prefix=data_prefix_target,
    pipeline=target_pipeline,
    metainfo=metainfo_target,
    modality=input_modality,
    box_type_3d='LiDAR',
    test_mode=False,
    load_eval_anns=False,
    filter_empty_gt=False,
    backend_args=backend_args)

unlabeled_strong_dataset = unlabeled_weak_dataset   # same pipeline, detector ignores it

# ── Dataloaders ───────────────────────────────────────────────────────────────

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

# ── Evaluator ─────────────────────────────────────────────────────────────────

val_evaluator = dict(
    type='NusOnKittiMetric',
    ann_file=target_data_root + ann_file_target_val,
    metric='bbox',
    pcd_limit_range=[0, -40, -3, 70.4, 40, 0],
    label_mapping=None,
    default_cam_key='CAM2',
    backend_args=backend_args)

test_evaluator = val_evaluator

# ── Model ─────────────────────────────────────────────────────────────────────

model = dict(
    type='SimpleMeanTeacher3DDetector',
    mean_teacher_cfg=dict(
        ema_momentum=0.999,
        update_teacher_buffers=False,
        conf_threshold=0.3,
        source_loss_weight=1.0,
        target_loss_weight=0.25,
        burn_in_iters=500,
        min_pseudo_per_sample=1,
        eval_use_teacher=True,
        verbose=True),
    pretrained_ckpt=pretrained_ckpt,

    detector=dict(
        type='VoxelNetBEVRoI',
        data_preprocessor=dict(
            type='Det3DDataPreprocessor',
            voxel=True,
            voxel_layer=dict(
                max_num_points=64,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=(30000, 40000))),

        voxel_encoder=dict(
            type='PillarFeatureNet',
            in_channels=4,
            feat_channels=[64],
            with_distance=False,
            voxel_size=voxel_size,
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
            point_cloud_range=point_cloud_range),

        middle_encoder=dict(
            type='PointPillarsScatter',
            in_channels=64,
            output_shape=output_shape),

        backbone=dict(
            type='SECOND',
            in_channels=64,
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2],
            out_channels=[64, 128, 256]),

        neck=dict(
            type='SECONDFPN',
            in_channels=[64, 128, 256],
            upsample_strides=[1, 2, 4],
            out_channels=[128, 128, 128]),

        bbox_head=dict(
            type='Anchor3DHead',
            num_classes=1,
            in_channels=384,
            feat_channels=384,
            use_direction_classifier=True,
            assign_per_class=True,
            anchor_generator=dict(
                type='AlignedAnchor3DRangeGenerator',
                ranges=[
                    [-50.40, -50.40, -1.80, 50.40, 50.40, -1.80],  # Car
                ],
                sizes=[
                    [4.60, 1.95, 1.72],  # Car
                ],
                rotations=[0, 1.57],
                reshape_out=False),
            diff_rad_by_sin=True,
            dir_offset=-0.7854,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                loss_weight=1.5),
            loss_dir=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.2)),

        train_cfg=dict(
            assigner=[
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False),

        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.05,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50)))

# ── Runtime ───────────────────────────────────────────────────────────────────

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best=None),
    visualization=dict(type='Det3DVisualizationHook', draw=False))

custom_hooks = [dict(type='MeanTeacherHook', interval=1)]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    accumulative_counts=4,
    clip_grad=dict(max_norm=35, norm_type=2))
