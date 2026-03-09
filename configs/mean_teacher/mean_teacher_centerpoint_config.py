_base_ = ['../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py']

source_dataset_type = 'NuScenesDataset'
# source_data_root = '/DATA/nuScenes/'
source_data_root = '/home/erfans00/nuscenes/'
ann_file_source = 'nuscenes_infos_train.pkl'
data_prefix_source = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
classes_nuscenes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                  'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
box_origin_source = (0.5, 0.5, 0.5)        # nuScenes box origin
metainfo_source = dict(classes=classes_nuscenes, origin=box_origin_source)

target_dataset_type = 'KittiDataset'
target_data_root = '/DATA/kitti_mmdet3d/'
ann_file_target = 'kitti_infos_train.pkl'
data_prefix_target = dict(pts='training/velodyne_reduced')
classes_kitti = ['Car', 'Pedestrian', 'Cyclist']
box_origin_target = (0.5, 0.5, 0)        # KITTI box origin
metainfo_target = dict(classes=classes_kitti, box_origin=box_origin_target)

hard_instance_bank_path = './configs/mean_teacher/hard_instance_bank/hard_instance_bank_nuscenes_quantile_kitti_20.pkl'
pretrained_ckpt = './work_dirs/pretrain_16feb/epoch_24.pth'

# point_cloud_range = [-50.40, -50.40, -5, 50.40, 50.40, 3]   # nuScenes point cloud range
point_cloud_range = [0, -51.2, -5, 68.80, 51.2, 3]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(
        classes=['Car', 'Pedestrian', 'Cyclist'],
        box_origin=(0.5, 0.5, 0.5))
backend_args = None

# Dataset
# db_sampler_kitti= dict(
#     data_root=target_data_root,
#     info_path=target_data_root + 'kitti_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
#     classes=classes_kitti,
#     sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     backend_args=backend_args)

source_pipeline = [     # nuScenes         # supervised training on source data
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4),
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
            class_names=classes_kitti,
            keep_unmapped=False),  # Drop unmapped classes like 'trailer', 'barrier'
   dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],        # +/- 22.5 degrees
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=classes_kitti),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ]

target_weak_pipeline = [       # KITTI      # sent to teacher model for predicting pseudo instances
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(
        type='KittiToNuscenes'),                    # convert kitti coordinates to nuscenes format

        ### the augmentations below makes the pseudo-labels in a different coordinate system, either delete them or take into account the conversion back to previous coords
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.087, 0.087],      # ±5 degrees (WEAK)
    #     scale_ratio_range=[0.98, 1.02]),
    
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points'])            # pack only points for unlabeled data
]

target_strong_pipeline = [       # KITTI      # sent to student model for unsupervised training
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(
        type='KittiToNuscenes'),                    # transform coordinates to nuscenes style
    # dict(type='ObjectSample', db_sampler=db_sampler_kitti),

    dict(
        type='HardInstanceSampling',
        hard_instance_bank_path=hard_instance_bank_path,
        sample_groups=dict(
            Car=5, Pedestrian=3, Cyclist=3),        # number of hard instances to sample per class
        use_pred_boxes_for_collision=True,          # Use predictions for collision
        iou_thresh=0.3,                             # Collision detection threshold
        points_loader=dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,         # Source is nuScenes (5D)
            use_dim=4)),

    # dict(
    #     type='ObjectNoise',
    #     num_try=100,
    #     translation_std=[1.0, 1.0, 0.5],
    #     global_rot_range=[0.0, 0.0],
    #     rot_range=[-0.78539816, 0.78539816]
    #     ),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],                # +/- 45 degrees
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points'])            # pack only points for unlabeled data
]

labeled_dataset = dict(          # nuScenes
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
        backend_args=backend_args
        )

unlabeled_weak_dataset = dict(        # KITTI
    type=target_dataset_type,
    data_root=target_data_root,
    ann_file=ann_file_target,
    data_prefix=data_prefix_target,
    pipeline=target_weak_pipeline,
    metainfo=metainfo_target,
    modality=input_modality,
    box_type_3d='LiDAR',
    test_mode=False,
    load_eval_anns=False,           # not to load annotations, even though an ann_file is provided
    filter_empty_gt=False,          # skip GT check in prepare_data          
    backend_args=backend_args
    )

unlabeled_strong_dataset = dict(        # KITTI
    type=target_dataset_type,
    data_root=target_data_root,
    ann_file=ann_file_target,
    data_prefix=data_prefix_target,
    pipeline=target_strong_pipeline,
    metainfo=metainfo_target,
    modality=input_modality,
    box_type_3d='LiDAR',
    test_mode=False,
    load_eval_anns=False,           # not to load annotations, even though an ann_file is provided
    filter_empty_gt=False,          # skip GT check in prepare_data
    backend_args=backend_args
    )

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='mean_teacher_collate_fn'),
    dataset=dict(
            type='MTCombinedDataset',
            labeled_dataset=labeled_dataset,
            unlabeled_weak_dataset=unlabeled_weak_dataset,
            unlabeled_strong_dataset=unlabeled_strong_dataset,
            metainfo=metainfo)
    )

# val_dataloader = dict()
# test_dataloader = dict()

# Model
voxel_size = [0.1, 0.1, 0.2]
x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
vx, vy, vz = voxel_size

import numpy as np
grid_size = [
    int(np.round((x_max - x_min) / vx)),
    int(np.round((y_max - y_min) / vy)),
    int(np.round((z_max - z_min) / vz))]

sparse_shape = [grid_size[2]+1, grid_size[1], grid_size[0]]     # [41, 688, 1024] -> # BEV feature map: [86, 128]

model = dict(
    type='MeanTeacher3DDetector',
    mean_teacher_cfg=dict(
                     point_cloud_range=point_cloud_range,
                     ema_momentum=0.999,
                     use_bev_consistency=True,
                     tau=0.07,
                     lambda_weight=0.05,
                     voxel_size=voxel_size[0],
                     # Confidence thresholding params
                     conf_threshold=0.3,
                     use_class_specific_thresh=False,
                     class_thresholds=None,  # dict: {class_id: threshold}
                     # loss weights
                     source_loss_weight=1.0,
                     target_loss_weight=0.5,
                    contrastive_weight=1.0,
                 ),
    pretrained_ckpt=pretrained_ckpt,

    # The architecture for Student and Teacher
    detector = dict(
        type='CenterPoint',
        data_preprocessor=dict(
            type='Det3DDataPreprocessor',
            voxel=True,
            voxel_layer=dict(
                max_num_points=10,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=(90000, 120000))),

        pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),       # num_features = min(dims of source & target domains)

        pts_middle_encoder=dict(
            type='SparseEncoder',
            in_channels=4,                          # same as num_features in voxel encoder
            sparse_shape=sparse_shape,
            output_channels=128,                    # final output actually = 256
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32),
                              (32, 32, 64),
                              (64, 64, 128),
                              (128, 128)),
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
            in_channels=sum([256, 256]),
            tasks=[
                dict(num_class=1, class_names=['Car']),
                dict(num_class=1, class_names=['Pedestrian']),
                dict(num_class=1, class_names=['Cyclist']),
            ],
            common_heads=dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),      # velocity removed
            share_conv_channel=64,
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                post_center_range=point_cloud_range,
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=7),                   # velocity removed
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3),
            loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(
                type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
            norm_bbox=True),

        # model training and testing settings
        train_cfg=dict(
            pts=dict(
                grid_size=grid_size,
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),        # dropped last 2 for velocity

        test_cfg=dict(
            pts=dict(
                post_center_limit_range=point_cloud_range,
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4,0.85, 0.175],         # kept only three classes
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                nms_type='rotate',
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2)
                )
            )
)


# Runtime configs
# Hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5),
    logger=dict(type='LoggerHook', interval=1))

# custom_imports = dict(
#     imports=['mmdet3d.engine.hooks.mean_teacher_hook'],
#     allow_failed_imports=False
# )
custom_hooks = [dict(type='MeanTeacherHook', interval=1)]

# Scheduler and optimizer config
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
val_cfg = None
test_cfg = None