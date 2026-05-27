_base_ = [  '../_base_/schedules/schedule-2x.py',
            '../_base_/default_runtime.py']

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
ann_file_train = 'nuscenes_infos_train.pkl'
ann_file_val = 'nuscenes_infos_val.pkl'
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')

box_origin = (0.5, 0.5, 0.5)        # nuScenes box origin

classes_nuscenes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                  'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
metainfo_source = dict(classes=classes_nuscenes, origin=box_origin)
metainfo = dict(classes=['Car'], origin=box_origin)

# point_cloud_range = [-50.40, -50.40, -5, 50.40, 50.40, 3]   # nuScenes point cloud range
# point_cloud_range = [0, -50.40, -5, 68.80, 50.40, 3]        # front face point cloud range (x_min=0 to avoid training on rear points that won't be seen during inference)
point_cloud_range = [-50.40, -50.40, -5, 50.40, 50.40, 3]
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

db_sampler = dict(
    type='DataBaseSampler',
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(car=5)),
    classes=['car'],
    sample_groups=dict(car=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3],   # match 4-ch scene points produced by LoadPointsFromMultiSweeps
        backend_args=backend_args))

# Dataset
train_pipeline = [       # nuScenes
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,                          # keep all 5 cols so MultiSweeps can set col-4 time
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,
        use_dim=[0, 1, 2, 3],              # drop ring index; final output: x,y,z,intensity (4-ch, KITTI-compatible)
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    dict(
            type='ClassRemapWithLabel',
            mapping={
                'car': 'Car',
                # 'bicycle': 'Cyclist',
                # 'motorcycle': 'Cyclist',
                # 'pedestrian': 'Pedestrian',
            },
            class_names=metainfo['classes'],
            keep_unmapped=False),  # Keep or Drop unmapped classes like 'trailer', 'barrier'
    dict(
        type='RandomObjectScaling',
        scale_range=[0.75, 1.0],   # shrink nuScenes Cars toward KITTI size
        num_try=50,
        class_names=['Car']),
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
        sweeps_num=5,
        use_dim=[0, 1, 2, 3],
        test_mode=True,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
            type='ClassRemapWithLabel',
            mapping={
                'car': 'Car',
                # 'bicycle': 'Cyclist',
                # 'motorcycle': 'Cyclist',
                # 'pedestrian': 'Pedestrian',
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
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
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

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,
        use_dim=[0, 1, 2, 3],
        test_mode=True,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=8,
    num_workers=6,
    prefetch_factor=4,      # each worker pre-fetches 4 batches
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
    num_workers=2,
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
        filter_empty_gt=False,
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

# val_evaluator = [
#     dict(
#         type='NuScenesKittiMetric',
#         ann_file=data_root + ann_file_val,
#         metric='bbox',
#         pcd_limit_range=point_cloud_range,
#         default_cam_key='CAM_FRONT',
#         label_mapping={                 # handles GT label mapping by the metric
#             'car': 'Car',
#             'bicycle': 'Cyclist',
#             'motorcycle': 'Cyclist',
#             'pedestrian': 'Pedestrian'}
#         ),
#     # dict(
#     #     type='DumpResults', 
#     #     out_file_path='work_dirs/pretrain_8apr/results.pkl'
#     #     )
# ]


val_evaluator = [
    dict(
        type='NuScenesRemappedMetric',
        data_root=data_root,
        ann_file=data_root + ann_file_val,
        metric='bbox',
        model_classes=metainfo['classes'],    # ['Car', 'Pedestrian', 'Cyclist']
        class_mapping={
            'Car':        'car',
            # 'Pedestrian': 'pedestrian',
            # 'Cyclist':    'bicycle',          # bicycle+motorcycle → Cyclist during train
        },
    )
]

test_dataloader = val_dataloader
test_evaluator = val_evaluator

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
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        point_cloud_range=point_cloud_range),

    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=output_shape),       # output_shape = range / voxel_size (x and y)

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
        # num_classes=3,
        num_classes=1,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(                      
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-50.40, -50.40, -1.80, 50.40, 50.40, -1.80],    # Car
                # [-50.40, -50.40, -1.62, 50.40, 50.40, -1.62],    # Pedestrian
                # [-50.40, -50.40, -1.67, 50.40, 50.40, -1.67],    # Cyclist
                ],
            # ranges=[                                          # front face range
            #     [0, -50.40, -1.80, 68.80, 50.40, -1.80],    # Car
            #     [0, -50.40, -1.62, 68.80, 50.40, -1.62],    # Pedestrian
            #     [0, -50.40, -1.67, 68.80, 50.40, -1.67]     # Cyclist
            # ],
            # ranges=[
            #     [0, -50.40, -0.6, 68.80, 50.40, -0.6],    # Pedestrian
            #     [0, -50.40, -0.6, 68.80, 50.40, -0.6],    # Cyclist
            #     [0, -50.40, -1.78, 68.80, 50.40, -1.78]     # Car
            # ],
            sizes=[
                [4.2, 2.0, 1.6],         # Car
                # [0.72, 0.66, 1.76],           # Pedestrian    
                # [1.68, 0.6, 1.27]           # Cyclist
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.5),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),

    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            # dict(  # for Pedestrian
            #     type='Max3DIoUAssigner',
            #     iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
            #     pos_iou_thr=0.45,
            #     neg_iou_thr=0.3,
            #     min_pos_iou=0.3,
            #     ignore_iof_thr=-1),
            # dict(  # for Cyclist
            #     type='Max3DIoUAssigner',
            #     iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
            #     pos_iou_thr=0.45,
            #     neg_iou_thr=0.3,
            #     min_pos_iou=0.3,
            #     ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),

    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=3, save_best=None),
    visualization=dict(type='Det3DVisualizationHook', draw=False)
)


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=12)

# Gradient accumulation with 8 steps to achieve effective batch size of 32 (8 x 4)
optim_wrapper = dict(type='AmpOptimWrapper',
                     loss_scale='dynamic',
                     optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
                     accumulative_counts=4,
                     clip_grad=dict(max_norm=35, norm_type=2))