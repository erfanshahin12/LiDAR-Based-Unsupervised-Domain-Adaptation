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

pretrained_ckpt = './work_dirs/pretrain_pp_ros_26may/epoch_24.pth'

point_cloud_range = [-50.40, -50.40, -5, 50.40, 50.40, 3]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(
        classes=['Car'],
        origin=(0.5, 0.5, 0.5))
backend_args = None

# ── Pipelines ─────────────────────────────────────────────────────────────────

source_pipeline = [     # nuScenes         # supervised training on source data
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,
        use_dim=[0, 1, 2, 3],              # drop ring index; final output: x,y,z,intensity (4-ch, KITTI-compatible)
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
            type='ClassRemapWithLabel',
            mapping={
                'car': 'Car',
            },
            class_names=classes_kitti,
            keep_unmapped=False),  # Drop all non-Car classes
    # Source object scale augmentation (stable-baseline ROS).
    dict(type='RandomObjectScaling', scale_range=[0.75, 1.0], class_names=['Car']),
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

    # ── CMT hard-instance sampling (online, from nuScenes dbinfos) ──────────
    # Builds the bank in-process (no offline build_hard_bank.py step needed).
    # Selects nuScenes source instances whose point count is below the 25th
    # percentile of the KITTI target distribution, carves the insertion region,
    # and normalises box/point sizes toward KITTI.  Injected boxes are packed as
    # gt_bboxes_3d / gt_labels_3d and merged as reliable GT in the detector.
    # Set sample_groups=None (or remove) to disable without touching other config.
    dict(
        type='HardInstanceSampling',
        source_db_path=source_data_root + 'nuscenes_dbinfos_train.pkl',
        source_class_mapping=dict(Car='car'),       # nuScenes 'car' → KITTI 'Car'
        db_path_prefix=source_data_root,            # resolve relative DB point-file paths
        sample_groups=dict(Car=5),
        use_pred_boxes_for_collision=False,          # no preds available in strong pipeline
        iou_thresh=0.3,  # applied between injected instances (inter-instance collision)
        carve=True,                                 # CMT: remove returns in insertion zone
        carve_extra_width=(1.0, 0.5, 0.5),          # (dl, dw, dh) expand for carve
        size_normalize=dict(size_res=[-0.71, -0.35, -0.16]),  # match source pipeline shrink
        class_names=classes_kitti,
        points_loader=dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,                             # nuScenes points are 5D
            use_dim=4)),                            # keep x,y,z,intensity

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
        # gt_bboxes_3d / gt_labels_3d carry the injected hard-instance boxes;
        # real KITTI GT is never loaded (no LoadAnnotations3D in this pipeline).
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
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
    pcd_limit_range=point_cloud_range,
    label_mapping=None,
    default_cam_key='CAM2',
    ground_snap=dict(enabled=True, pctl=2.0, min_pts=25, margin=0.3, max_disp=0.0),
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
        ema_momentum=0.99995,
        # EMA the teacher's DSNorm/BN buffers from the student so the teacher's
        # target-domain running stats track KITTI (standard Mean-Teacher). With
        # False the teacher normalised KITTI using frozen nuScenes stats while
        # its affine params were EMA'd toward the student's KITTI-batch stats —
        # a growing miscalibration. The EMA loop only touches float buffers.
        update_teacher_buffers=True,
        # Contrastive BEV-consistency loss is DEACTIVATED (use_bev_consistency=
        # False, contrastive_weight=0). The detector-level roi_extractor_cfg is
        # also removed below — the RoI head's only consumer was this loss, so
        # there is nothing to build. (CenterPoint leaves this plumbing nominally
        # on at weight 0; here it is fully off.) The thresholds below are inert.
        use_bev_consistency=False,
        tau=0.07,
        fg_threshold=0.5,
        neg_threshold=0.25,
        contrastive_warmup_iters=3517,  # 1 epoch
        source_loss_weight=1.0,
        target_loss_weight=0.5,
        contrastive_weight=0.0,
        verbose=True,
        eval_use_teacher=True,
        use_dsnorm=True,
        # cls-only filtering — no IoU head in this architecture
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

        # NOTE: roi_extractor_cfg intentionally omitted. Its only consumer was
        # the MeanTeacher contrastive/BEV-consistency loss, which is deactivated
        # (see mean_teacher_cfg above). With it absent, VoxelNetBEVRoI sets
        # self.roi_extractor=None and contrastive_loss early-returns 0.

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
            # NMS ranks by classification score only (no IoU head).
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.05,
            min_bbox_size=0,
            nms_pre=1000,
            max_num=200)))

# Runtime configs
# Hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best=None),
    visualization=dict(type='Det3DVisualizationHook', draw=False)
)
custom_hooks = [
    dict(type='MeanTeacherHook', interval=1),
    dict(
        type='PseudoLabelRefreshHook',
        interval=3,             # re-run teacher every # epochs
        update_at_epochs=(0,),  # always refresh before epoch 0 starts
        ps_batch_size=8,
        ps_num_workers=6,
        # Per-scene keep-fraction: keep top 70% by CLS score per scene, then a
        # FIXED absolute CLS floor of 0.40 as a guard (cls-only ranking;
        # this architecture has no RoI-IoU head).
        # cls_thr=0.40 replaces the old adaptive cls_percentile (now 0/off):
        # min_pts=10 drops sparse phantom FPs.
        # Ordering: fraction-first, floor-as-guard (the hook's only mode).
        # coverage_gate_drop=0.10 → skip store update if coverage drops >10%.
        keep_frac=0.6,
        iou_weight=0,
        iou_thr=0.0,
        cls_thr=0.35,
        cls_percentile=0,
        hybrid_thr=0.0,
        coverage_gate_drop=0.10,
        ps_min_score=0.05,
        use_top1_fallback=False,
        min_pts=10,
        hard_instance_quantile=25,
        # Stage 2 — augmentation-consistency geometry QC: keep only boxes that
        # reproduce under a BEV flip (teacher second view), 3D-IoU >= thresh.
        # No target-domain priors. enabled=False ⇒ skip the second pass.
        consistency_filter=dict(enabled=True, iou_thresh=0.5,
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
        ground_snap=dict(enabled=True, pctl=2.0, min_pts=25, margin=0.3, max_disp=0.0),
    ),
]

# Scheduler and optimizer config
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)

# Gradient accumulation with 8 steps to achieve effective batch size of 32 (8 x 4)
optim_wrapper = dict(type='AmpOptimWrapper',
                     loss_scale='dynamic',
                     optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
                     accumulative_counts=4,
                     clip_grad=dict(max_norm=35, norm_type=2))

# Override schedule-2x's MultiStepLR (milestones at epochs 20 & 23 — never fires in
# a 10-epoch run, so LR would stay at 1e-4 flat).  Short warmup + cosine decay mirrors
# the CenterPoint Stage-1 fix that resolved the collapse (T_max=10 = run length).
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=10, end=10,
         by_epoch=True, eta_min=1e-6),
]