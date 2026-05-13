_base_ = ['../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py']

source_dataset_type = 'NuScenesDataset'
source_data_root = 'data/nuscenes/'
ann_file_source = 'nuscenes_infos_train.pkl'
data_prefix_source = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
classes_nuscenes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                  'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
box_origin_source = (0.5, 0.5, 0.5)        # nuScenes box origin
metainfo_source = dict(classes=classes_nuscenes, origin=box_origin_source)

target_dataset_type = 'KittiDataset'
target_data_root = 'data/kitti/'
ann_file_target = 'kitti_infos_train.pkl'
data_prefix_target = dict(pts='training/velodyne_reduced')
classes_kitti = ['Car']
box_origin_target = (0.5, 0.5, 0)        # KITTI box origin
metainfo_target = dict(classes=classes_kitti, origin=box_origin_target)

hard_instance_bank_path = './configs/mean_teacher/hard_instance_bank/hard_instance_bank_nuscenes_quantile_kitti_20.pkl'
pretrained_ckpt = './work_dirs/baseline_pointpillars_5may/epoch_24.pth'

# point_cloud_range = [-50.40, -50.40, -5, 50.40, 50.40, 3]   # nuScenes point cloud range
point_cloud_range = [-50.40, -50.40, -5, 50.40, 50.40, 3]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(
        classes=['Car'],
        origin=(0.5, 0.5, 0.5))
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
            },
            class_names=classes_kitti,
            keep_unmapped=False),  # Drop all non-Car classes
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

    # dict(
    #     type='HardInstanceSampling',
    #     hard_instance_bank_path=hard_instance_bank_path,
    #     sample_groups=dict(
    #         Car=5, Pedestrian=3, Cyclist=3),        # number of hard instances to sample per class
    #     use_pred_boxes_for_collision=True,          # Use predictions for collision
    #     iou_thresh=0.3,                             # Collision detection threshold
    #     points_loader=dict(
    #         type='LoadPointsFromFile',
    #         coord_type='LIDAR',
    #         load_dim=5,         # Source is nuScenes (5D)
    #         use_dim=4)),

    # dict(
    #     type='ObjectNoise',
    #     num_try=100,
    #     translation_std=[1.0, 1.0, 0.5],
    #     global_rot_range=[0.0, 0.0],
    #     rot_range=[-0.78539816, 0.78539816]
    #     ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],                        # +/- 22.5 degrees (matches source)
        scale_ratio_range=[0.95, 1.05]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # PointsRangeFilter is intentionally omitted here: after rotating KITTI data
    # (which in nuScenes frame clusters at x_nus≈0, y_nus>0) by ±22.5° the range
    # filter [0, -50.4,...] can eliminate ALL points, crashing the CUDA voxelizer
    # with gridDim=0.  The voxelizer's own internal clip (same range) is
    # equivalent and never receives an empty tensor.
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points'])            # pack only points for unlabeled data
]

val_pipeline = [        # KITTI val — evaluated in nuScenes frame, then inverted by NusOnKittiMetric
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True, with_label_3d=True, backend_args=backend_args),
    # Rotate KITTI (X-fwd, Y-left) → nuScenes (X-right, Y-fwd) so the teacher
    # sees data in its training frame.  NusOnKittiMetric inverts predictions back.
    dict(type='KittiToNuscenes'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
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
    batch_size=8,
    num_workers=6,
    prefetch_factor=4,      # each worker pre-fetches 4 batches
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='mean_teacher_collate_fn'),
    dataset=dict(
            type='MTCombinedDataset',
            labeled_dataset=labeled_dataset,
            unlabeled_weak_dataset=unlabeled_weak_dataset,
            unlabeled_strong_dataset=unlabeled_strong_dataset,
            metainfo=metainfo)
    )

ann_file_target_val = 'kitti_infos_val.pkl'

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
    pcd_limit_range=[0, -40, -3, 70.4, 40, 0],
    label_mapping=None,
    default_cam_key='CAM2',
    backend_args=backend_args)

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
    type='MeanTeacher3DDetector',
    mean_teacher_cfg=dict(
                     point_cloud_range=point_cloud_range,
                     ema_momentum=0.999,
                     update_teacher_buffers=False,      # BAN: teacher BN tracks target stats via .train()
                     use_bev_consistency=True,
                     tau=0.07,
                     symmetric_contrastive=True,
                     # Confidence thresholding params
                     conf_threshold=0.2,
                     use_class_specific_thresh=False,
                     class_thresholds=None,
                     # loss weights
                     source_loss_weight=1.0,
                     target_loss_weight=0.25,
                     contrastive_weight=0.05,
                     burn_in_iters=500,
                     min_pseudo_per_sample=3,
                     verbose=True,
                     eval_use_teacher=True,
                 ),
    pretrained_ckpt=pretrained_ckpt,

    # The architecture for Student and Teacher
    detector = dict(
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
                    [4.60, 1.95, 1.72],         # Car
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
            nms_thr=0.05,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=200,
            max_num=100)))

# Runtime configs
# Hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best=None),
    visualization=dict(type='Det3DVisualizationHook', draw=False)
)

# custom_imports = dict(
#     imports=['mmdet3d.engine.hooks.mean_teacher_hook'],
#     allow_failed_imports=False
# )
custom_hooks = [dict(type='MeanTeacherHook', interval=1)]

# Scheduler and optimizer config
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)

# Gradient accumulation with 8 steps to achieve effective batch size of 32 (8 x 4)
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
                     accumulative_counts=4,
                     clip_grad=dict(max_norm=35, norm_type=2))