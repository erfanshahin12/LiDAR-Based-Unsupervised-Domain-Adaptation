_base_ = ['./mean_teacher_centerpoint_config.py']\

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
ann_file_val = 'nuscenes_infos_val.pkl'
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
classes_nuscenes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
box_origin_source = (0.5, 0.5, 0.5)
metainfo = dict(classes=classes_nuscenes, origin=box_origin_source)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
backend_args = None


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
        class_names=['Car'],
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

val_dataloader = dict(
    batch_size=4,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        data_prefix=data_prefix,
        pipeline=val_pipeline,
        metainfo=metainfo,
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=False,
        filter_empty_gt=False,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='NuScenesRemappedMetric',
        data_root=data_root,
        ann_file=data_root + ann_file_val,
        metric='bbox',
        model_classes=['Car'],
        class_mapping={'Car': 'car'},
    )
]

test_evaluator = val_evaluator

voxel_size  = [0.1, 0.1, 0.2]

model = dict(
    detector=dict(
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