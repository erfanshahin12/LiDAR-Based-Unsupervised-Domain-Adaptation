_base_ = ['mt_pretrain_pointpillars_config.py']

# Replace the distance-based evaluator with the official KITTI IoU AP metric.
#
# dataset_meta['classes'] = ['Car'] (single class, from the base config).
# Model outputs label 0 = Car.  No label remapping needed (label_mapping=None).
# NusOnKittiMetric inverse-rotates predictions from nuScenes LiDAR frame back
# to KITTI LiDAR frame before passing them to KittiMetric.
#
# Reports: Car AP_R40 @ IoU 0.7 — bbox / BEV / 3D for Easy / Moderate / Hard.
# The Moderate BEV and 3D numbers are the values to report (ST3D convention).

data_root    = 'data/kitti/'
ann_file_val = 'kitti_infos_val.pkl'
point_cloud_range = [-50.40, -50.40, -5.0, 50.40, 50.40, 3.0]
backend_args = None

val_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, backend_args=backend_args),
    # Rotate points and GT boxes from KITTI LiDAR frame (X forward, Y left)
    # to nuScenes LiDAR frame (X right, Y forward) so the pretrained model
    # receives data in the coordinate system it was trained on.
    dict(type='KittiToNuscenes'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # Filter GT to the same spatial window used during training so the
    # metric is not penalised for missing objects outside the model's range.
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root=data_root,
        ann_file=ann_file_val,
        data_prefix=dict(pts='training/velodyne_reduced'),
        pipeline=val_pipeline,
        metainfo=dict(classes=['Car']),
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=False,
        filter_empty_gt=False,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='NusOnKittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    pcd_limit_range=[-50.40, -50.40, -5.0, 50.40, 50.40, 3.0],
    label_mapping=None,
    default_cam_key='CAM2',
    backend_args=None)

test_evaluator = val_evaluator

model = dict(
    roi_extractor_cfg=dict(
        in_channels=64,
        out_channels=256,
        roi_size=7,
        voxel_size=0.2,
        point_cloud_range=[-50.4, -50.4, -5.0, 50.4, 50.4, 3.0]),
    bbox_head=dict(predict_iou=True),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50,
        score_type='cls',
        score_weights=dict(iou=0.0, cls=1.0)))
