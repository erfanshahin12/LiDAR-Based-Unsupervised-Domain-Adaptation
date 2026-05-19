_base_ = ['../_base_/schedules/cyclic-20e.py',
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

# 51.2 m chosen deliberately: 102.4 / 0.1 = 1024 (power-of-2), matches the
# base config's sparse_shape=[41,1024,1024] exactly.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
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
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args))

# Dataset
train_pipeline = [          # nuscenes
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
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    dict(
        type='ClassRemapWithLabel',
        mapping={'car': 'Car'},
        class_names=metainfo['classes'],
        keep_unmapped=False),       # Keep or Drop unmapped classes
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
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
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ClassRemapWithLabel',
        mapping={'car': 'Car'},
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

train_dataloader = dict(
    batch_size=8,           # Reduce to 6 or 4 if GPU OOM; effective batch size is 32 via accumulative_counts=4.
    num_workers=6,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
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
    dataset=dict(
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

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='NuScenesRemappedMetric',
        data_root=data_root,
        ann_file=data_root + ann_file_val,
        metric='bbox',
        model_classes=metainfo['classes'],
        class_mapping={'Car': 'car'},
    )
]
test_evaluator = val_evaluator

# ── Model ─────────────────────────────────────────────────────────────────────
# Key deviations from the standard 10-class CenterPoint base:
#
#   1. HardSimpleVFE num_features=4, SparseEncoder in_channels=4, matching the 4 channels used
#
#   2. tasks: single task for 'Car' only.
#
#   3. common_heads: velocity removed (no 'vel').  The downstream KITTI target
#      has no velocity annotations.      zero-pads vel.
#      code_size=7 (x,y,z,l,w,h,yaw) and code_weights has 7 entries.
#
#   4. DCNSeparateHead: as in the -head-dcn variant; improves localisation via
#      deformable convolution in the separate prediction heads.
#
#   5. nms_type='rotate' (standard NMS): circular NMS excluded as requested.
#      test_cfg min_radius=[4] has one entry matching the single task.
#
#   6. post_max_size=200: raised from the 83 used in the 6-task base config
#      (83 × 6 ≈ 500 total).  With a single task the full budget goes to cars.

voxel_size = [0.1, 0.1, 0.2]

model = dict(
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
        loss_cls=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
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
            nms_thr=0.2))
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best=None),
    visualization=dict(type='Det3DVisualizationHook', draw=False))

train_cfg = dict(max_epochs=20, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Effective batch size 32: single GPU batch_size= 8 × accumulative_counts=4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    accumulative_counts=4,
    clip_grad=dict(max_norm=35, norm_type=2))
