import sys
if "/home/erfans00/mmdetection3d" not in sys.path:
    sys.path.append("/home/erfans00/mmdetection3d")

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import os
import argparse
from copy import deepcopy
import json
import logging
from tqdm import tqdm
import pickle as pkl

from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.apis.inferencers import LidarDet3DInferencer
from mmdet3d.structures.bbox_3d import Box3DMode
from mmdet3d.apis.inference import init_model
from mmdet3d.structures import Box3DMode, get_box_type, LiDARInstance3DBoxes
from mmdet3d.evaluation import KittiMetric

# from src.expert_late_fusion.expert_late_fusion import expert_late_fusion
# from src.expert_late_fusion.utils import compute_fundamental_matrix, read_kitti_calibration_data, calibration_to_torch
# from src.expert_late_fusion.bbox_utils import project_bboxes, extract_corners

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


if __name__ == '__main__':
    class_dict = {
        0: 'Pedestrian',
        1: 'Cyclist',
        2: 'Car',
    }
    
    parser = argparse.ArgumentParser(
        description="Inference script for single modal detectors")
    parser.add_argument("-output_dir", default=None, type=str, required=True,
                        help="The directory where the predictions will be placed")
    parser.add_argument("-detector_config", default=None, type=str, required=True,
                        help="Pointpillars config path or name")
    parser.add_argument("-checkpoint_path", default=None, type=str, required=True,
                        help="Pointpillar   s checkpoint path")
    parser.add_argument("-validation_split_path", 
                        default='/DATA/kitti_mmdet3d/ImageSets/val.txt', 
                        type=str, required=False,
                        help="Pointpillars checkpoint path")
    parser.add_argument("-score_thr", default=0.05, type=float, required=False,
                        help="Confidence Score Threshold")
    parser.add_argument("-nms_pre", default=1000, type=int, required=False,
                        help=" Maximum number of detections to consider before Non-Maximum Suppression")
    # nms: to limit the maximum number of boxes per scene
    parser.add_argument("-max_num", default=500, type=int, required=False,
                        help="Maximum number of final detections to keep after NMS")
    # Caps the final number of detections kept after NMS.
    parser.add_argument("-nms_thr", default=0.2, type=float, required=False,
                        help="NMS IoU threshold")
    # Controls how much overlap (IoU) is allowed between two boxes before one is suppressed by Non-Maximum Suppression.
    parser.add_argument("-annotation_file_eval", 
                        default='/DATA/kitti_mmdet3d/kitti_infos_val.pkl', 
                        type=str, required=False,
                        help="Annotation file to evaluate with mmdet3d KittiMetric")
    parser.add_argument("-kitti_root_path", default='/DATA/kitti_mmdet3d/training', 
                        type=str, required=False,
                        help="Directory where the dataset is placed")

    args = parser.parse_args()
    kitti_root_path = Path(args.kitti_root_path)
    left_path = kitti_root_path / 'image_2'         # left camera images, not used here
    velo_path = kitti_root_path / 'velodyne_reduced'    # LiDAR point cloud files (.bin format)
    calib_path = kitti_root_path / 'calib'          # Camera and LiDAR calibration files (not used)

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / 'data'
    metrics_path = output_dir / 'metrics.json'
    
    predictions_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(filename=str(output_dir / 'log.txt'), filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)

    with open(args.validation_split_path, 'r') as split_file:
        validation_ids = split_file.readlines()
        
    validation_ids = [val_id.rstrip('\n') for val_id in validation_ids]     # Removes newline characters from each ID
    
    # Load model architecture from config and weights from checkpoint onto GPU
    detector3d = init_model(args.detector_config, args.checkpoint_path, device='cuda:0')
    
    detector3d.cfg.model.test_cfg['nms_thr'] = args.nms_thr
    detector3d.cfg.model.test_cfg['nms_pre'] = args.nms_pre
    detector3d.cfg.model.test_cfg['max_num'] = args.max_num
    detector3d.cfg.model.test_cfg['score_thr'] = args.score_thr
    
    logging.info(detector3d)
    
    cfg = detector3d.cfg

    test_pipeline_3d = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline_3d[0].type = 'LoadPointsFromDict'         # Change first step from loading from file to loading from memory dictionary
    test_pipeline_3d = Compose(test_pipeline_3d)            # Chains all preprocessing steps together
    box_type_3d, box_mode_3d = get_box_type(cfg.test_dataloader.dataset.box_type_3d)
    
    results_list = []

    for i, val_id in tqdm(enumerate(validation_ids)):       # tqdm() provides progress bar iteration
                                                            # enumerate() provides both index and sample ID
        
        points = np.fromfile(velo_path / f'{val_id}.bin', dtype=np.float32).reshape((-1, 4))    # Each row represents one 3D point with intensity value
       
        inputs = dict(
            points=points,
            timestamp=1,                    # Dummy Timestamp
            axis_align_matrix=np.eye(4),    # 4×4 identity matrix (no coordinate transformation)
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d
        )
        # inputs = dict(
        #     lidar_points=dict(lidar_path=str(velo_path / f'{val_id}.bin')),
        #     timestamp=1,
        #     axis_align_matrix=np.eye(4),
        #     box_type_3d=box_type_3d,
        #     box_mode_3d=box_mode_3d)

        collate_data = pseudo_collate([test_pipeline_3d(inputs)])       # psuedo collate: Batches the single sample for model input
                                    # Applies all preprocessing steps
        with torch.no_grad():
            detection_output = detector3d.test_step(collate_data)[0]        # 0: Gets first (and only) sample result from the batch
        
        result = {
            'pred_instances_3d': {
                'bboxes_3d': detection_output.pred_instances_3d.bboxes_3d.to('cpu'),
                'scores_3d': detection_output.pred_instances_3d.scores_3d.to('cpu'),
                'labels_3d': detection_output.pred_instances_3d.labels_3d.to('cpu'),
            },
            'sample_idx': i
        }
        results_list.append(result)
    
    # remap nuscenes class ids to kitti class ids
    logging.info('Remapping Nuscenes predicted classes to Kitti classes')
    
    nusc_class_dict = {
        0: 'car', 1: 'truck', 2: 'trailer', 3: 'bus', 4: 'construction_vehicle',
        5: 'bicycle', 6: 'motorcycle', 7: 'pedestrian', 8: 'traffic_cone', 9: 'barrier'}
    
    remap_dict = {
    0: 2,  # car -> Car
    # 1: 2,  # truck -> Car
    # 2: 2,  # trailer -> Car
    # 3: 2,  # bus -> Car
    # 4: 2,  # construction_vehicle -> Car
    5: 1,  # bicyle -> Cyclist
    6: 1,  # motorcycle -> Cyclist
    7: 0,  # pedestrian -> Pedestrian
    # 8 (traffic_cone) and 9 (barrier) neglected
    }

    remapped_results_list = []
    for result in results_list:
        original_labels = result['pred_instances_3d']['labels_3d']
        new_labels = []
        valid_indices = []

        for i, label_idx in enumerate(original_labels):
            if label_idx.item() in remap_dict:
                new_labels.append(remap_dict[label_idx.item()])
                valid_indices.append(i)
        
        # Filter the results to only keep relevant classes
        result['pred_instances_3d']['labels_3d'] = torch.tensor(new_labels, dtype=torch.long)
        result['pred_instances_3d']['bboxes_3d'] = result['pred_instances_3d']['bboxes_3d'][valid_indices]
        result['pred_instances_3d']['scores_3d'] = result['pred_instances_3d']['scores_3d'][valid_indices]
        
        remapped_results_list.append(result)
    
    logging.info('Transforming predictions from nuScenes-space back to KITTI-space...')

    # Inverse of KittiToNuscenes transform
    final_results_list = []
    for result in remapped_results_list:
        boxes_nus = result['pred_instances_3d']['bboxes_3d']    # This is a LiDARInstance3DBoxes object
        box_tensor_nus = boxes_nus.tensor                       # Get the (N, 7) tensor
        
        # Create a new tensor for KITTI-space boxes
        # The nuScenes tensor might be (N, 9) [with velocity], but KITTI is (N, 7) (remove velocities)
        box_tensor_kitti = box_tensor_nus.new_zeros((box_tensor_nus.shape[0], 7))
        
        # Center transform:
        box_tensor_kitti[:, 0] = box_tensor_nus[:, 1]  # x_kitti = y_nus
        box_tensor_kitti[:, 1] = -box_tensor_nus[:, 0] # y_kitti = -x_nus
        box_tensor_kitti[:, 2] = box_tensor_nus[:, 2]  # z_kitti = z_nus
        
        # Dimension transform (swap l and w):
        box_tensor_kitti[:, 3] = box_tensor_nus[:, 4]  # l_kitti = w_nus
        box_tensor_kitti[:, 4] = box_tensor_nus[:, 3]  # w_kitti = l_nus
        box_tensor_kitti[:, 5] = box_tensor_nus[:, 5]  # h_kitti = h_nus
        
        # Yaw transform:
        box_tensor_kitti[:, 6] = box_tensor_nus[:, 6] - (np.pi / 2)
        
        # Normalize yaw to [-pi, pi]
        box_tensor_kitti[:, 6] = torch.atan2(
            torch.sin(box_tensor_kitti[:, 6]),
            torch.cos(box_tensor_kitti[:, 6])
        )

        # Update the result dict with transformed boxes
        # We must use the original Box3DMode.LIDAR for KITTI
        result['pred_instances_3d']['bboxes_3d'] = LiDARInstance3DBoxes(
            box_tensor_kitti, 
            box_dim=7, 
            origin=(0.5, 0.5, 0)
            )
        
        final_results_list.append(result)
    
    # Use this final, transformed list for evaluation
    results_list = final_results_list
    
    logging.info('Starting evaluation with KittiMetric')
    
    kitti_metric = KittiMetric(
        ann_file=args.annotation_file_eval,         # Ground truth annotations for comparison
        metric='bbox',
        backend_args=None,
        submission_prefix=str(predictions_path)
    )
    # logging.info(f'Dataset meta: {detector3d.dataset_meta}')
    # kitti_metric._dataset_meta = detector3d.dataset_meta
    kitti_metric.dataset_meta = {'classes': ['Pedestrian', 'Cyclist', 'Car']}
    metrics_dict = kitti_metric.compute_metrics(results_list)       # computes metrics
    
    with open(metrics_path, 'w') as metrics_file:
        json.dump(metrics_dict, metrics_file, indent=4)

    with open(output_dir / 'results_list.pkl', 'wb') as f:
        pkl.dump(results_list, f)
        
    logging.info(f'Results: {metrics_dict}')

    notes = """# remapped car -> car , pedestrian -> pedestrian, motorcycle & bicycle -> cyclist
#score_thr (0.05), nms_pre (1500), max_num(500), nms_thr(0.3)

# transformed the kitti pointcloud to nus coordinates + performed inference + take back to kitti coord. sys for evaluation
# Transform Annotations (gt_bboxes) not used, while present in the transform imported class
    """
    with open(output_dir / 'notes.txt', "w") as f:
        f.write(notes)