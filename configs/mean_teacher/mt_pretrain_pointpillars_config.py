_base_ = [  '../_base_/schedules/schedule-2x.py',
            '../_base_/default_runtime.py']

dataset_type = 'NuScenesDataset'
# data_root = '/DATA/nuScenes/'
data_root = '/home/erfans00/nuscenes/'
ann_file_train = 'nuscenes_infos_train.pkl'
ann_file_val = 'nuscenes_infos_val.pkl'
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
classes_nuscenes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                  'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
box_origin = (0.5, 0.5, 0.5)        # nuScenes box origin
metainfo_source = dict(classes=classes_nuscenes, origin=box_origin,
                       version='v1.0-mini')

# point_cloud_range = [-50.40, -50.40, -5, 50.40, 50.40, 3]   # nuScenes point cloud range
point_cloud_range = [0, -50.40, -5, 68.80, 50.40, 3]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(
        classes=['Car', 'Pedestrian', 'Cyclist'],
        box_origin=(0.5, 0.5, 0.5))
backend_args = None

# Dataset
train_pipeline = [       # nuScenes
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
            type='ClassRemapWithLabel',
            mapping={
                'car': 'Car',
                'bicycle': 'Cyclist',
                'motorcycle': 'Cyclist',
                'pedestrian': 'Pedestrian',
            },
            class_names=metainfo['classes'],
            keep_unmapped=False),  # Drop unmapped classes like 'trailer', 'barrier'
   dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],        # +/- 22.5 degrees
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=metainfo['classes']),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]

val_pipeline = [        # nuScenes
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        test_mode=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
            type='ClassRemapWithLabel',
            mapping={
                'car': 'Car',
                'bicycle': 'Cyclist',
                'motorcycle': 'Cyclist',
                'pedestrian': 'Pedestrian',
            },
            class_names=metainfo['classes'],
            keep_unmapped=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]

# test_pipeline = [        # nuScenes
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(
#         type='LoadPointsFromMultiSweeps',
#         sweeps_num=10,
#         test_mode=True,
#         backend_args=backend_args),
#     dict(
#         type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#     dict(
#             type='ClassRemapWithLabel',
#             mapping={
#                 'car': 'Car',
#                 'bicycle': 'Cyclist',
#                 'motorcycle': 'Cyclist',
#                 'pedestrian': 'Pedestrian',
#             },
#             class_names=metainfo['classes'],
#             keep_unmapped=False),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1333, 800),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             dict(
#                 type='GlobalRotScaleTrans',
#                 rot_range=[0, 0],
#                 scale_ratio_range=[1., 1.],
#                 translation_std=[0, 0, 0]),
#             dict(type='RandomFlip3D'),
#             dict(
#                 type='PointsRangeFilter', point_cloud_range=point_cloud_range)
#         ]),
#     dict(type='Pack3DDetInputs', keys=['points'])
#     ]


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(          # nuScenes
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        metainfo=metainfo_source,
        modality=input_modality,
        box_type_3d='LiDAR',
        test_mode=False,
        with_velocity=False,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(          # nuScenes
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        data_prefix=data_prefix,
        pipeline=val_pipeline,
        metainfo=metainfo_source,
        modality=input_modality,
        box_type_3d='LiDAR',
        test_mode=False,
        with_velocity=False,
        backend_args=backend_args))

# test_dataloader = dict(
    # batch_size=1,
    # num_workers=1,
    # persistent_workers=True,
    # drop_last=False,
    # sampler=dict(type='DefaultSampler', shuffle=False),
    # dataset=dict(          # nuScenes
    #     type=dataset_type,
    #     data_root=data_root,
    #     ann_file=ann_file_val,
    #     data_prefix=data_prefix,
    #     pipeline=test_pipeline,
    #     metainfo=metainfo_source,
    #     modality=input_modality,
    #     box_type_3d='LiDAR',
    #     test_mode=True,
    #     with_velocity=False,
    #     backend_args=backend_args))

val_evaluator = dict(
    type='NuScenesKittiMetric',
    ann_file=data_root + ann_file_val,
    metric='bbox',
    pcd_limit_range=point_cloud_range,
    default_cam_key='CAM_FRONT',
    label_mapping={                 # handles GT label mapping by the metric
        'car': 'Car',
        'bicycle': 'Cyclist',
        'motorcycle': 'Cyclist',
        'pedestrian': 'Pedestrian'}
)


# val_cfg = None
test_cfg = None

# Model
voxel_size = [0.2, 0.2, 8]      # nuscenes/kitti intermediate voxel size
x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
vx, vy, vz = voxel_size

import numpy as np
output_shape = [
    int(np.round((y_max - y_min) / vy)),
    int(np.round((x_max - x_min) / vx))]

model = dict(
    type='VoxelNetBEVRoI',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=64,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(30000, 40000))),
    
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=output_shape),       # output_shape = range / voxel_size (x and y)
    
    backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
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
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(                      
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -50.40, -1.62, 68.80, 50.40, -1.62],    # Pedestrian
                [0, -50.40, -1.67, 68.80, 50.40, -1.67],    # Cyclist
                [0, -50.40, -1.80, 68.80, 50.40, -1.80]     # Car
            ],
            # ranges=[
            #     [0, -50.40, -0.6, 68.80, 50.40, -0.6],    # Pedestrian
            #     [0, -50.40, -0.6, 68.80, 50.40, -0.6],    # Cyclist
            #     [0, -50.40, -1.78, 68.80, 50.40, -1.78]     # Car
            # ],
            sizes=[
                [0.8, 0.6, 1.73],           # Pedestrian    
                [1.72, 0.6, 1.73],          # Cyclist
                [4.25, 1.78, 1.65]            # Car
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),

    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),

    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

# less momory usage during training with amp
# LR adjusted to batch size (batch size=2 vs original total batch size=32)
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=6.25e-5, weight_decay=0.01))