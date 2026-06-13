"""Measure the data-optimal nuScenes<->KITTI z (sensor-height) shift.

Decides between 0.11 and 0.2 by measuring where cars actually sit (ground
level) in each dataset's NATIVE LiDAR frame, then taking the difference.

Logic
-----
A car's box bottom rests on the ground, and the ground sits ~sensor_height
below the LiDAR origin. So the median car-bottom-z in each native frame is a
direct estimate of -sensor_height:
    nuScenes  median bottom-z  ~ -1.84
    KITTI     median bottom-z  ~ -1.73

`KittiToNuscenes` maps KITTI into the nuScenes frame via  z -= shift.
To make the two ground planes coincide we need:
    z_bottom_kitti - shift = z_bottom_nusc
 => shift* = median(z_bottom_kitti) - median(z_bottom_nusc)   (native frames)

The printed nuScenes/KITTI medians double as a sanity check: if they land
near -1.84 / -1.73 the per-dataset box conventions were handled correctly and
the recommended shift is trustworthy.

Usage
-----
    conda activate thesis
    cd ~/mmdetection3d
    python tools/measure_z_shift.py \
        --nusc data/nuscenes/nuscenes_infos_train.pkl \
        --kitti data/kitti/kitti_infos_train.pkl
"""
import argparse
import pickle

import numpy as np
import torch

from mmdet3d.structures.bbox_3d import (Box3DMode, CameraInstance3DBoxes,
                                        LiDARInstance3DBoxes)


def _car_label_id(metainfo, fallback_names):
    """Resolve the integer label id for the 'car' class from infos metainfo."""
    cats = (metainfo or {}).get('categories', None)
    if isinstance(cats, dict):
        for name, idx in cats.items():
            if name.lower() in fallback_names:
                return idx
    return None  # fall back to name-based filtering below


def _frame_car_boxes(info, car_id, car_names):
    """Yield the 7-dim car boxes for one frame as a float32 array (or None)."""
    raw = []
    for ins in info.get('instances', []):
        box = ins.get('bbox_3d', None)
        if box is None or len(box) < 7:
            continue
        lbl = ins.get('bbox_label_3d', ins.get('bbox_label', None))
        name = ins.get('name', None)
        is_car = ((car_id is not None and lbl == car_id)
                  or (name is not None and str(name).lower() in car_names))
        if is_car:
            raw.append(box[:7])
    return np.asarray(raw, dtype=np.float32) if raw else None


def load_bottom_z(pkl_path, car_names, source='lidar', origin=(0.5, 0.5, 0.5)):
    """Return bottom-center z of all car boxes in the file's NATIVE LiDAR frame.

    source='lidar'      : bbox_3d already in LiDAR frame (nuScenes). `origin`
                          is its stored convention; mmdet3d converts to internal
                          bottom-center (0.5,0.5,0), so tensor[:,2] is bottom-z.
    source='kitti_cam'  : bbox_3d in CAMERA frame (KITTI). Convert per-frame to
                          LiDAR using that frame's `lidar2cam` calib.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    metainfo = data.get('metainfo', {}) if isinstance(data, dict) else {}
    infos = data['data_list'] if isinstance(data, dict) and 'data_list' in data else data
    car_id = _car_label_id(metainfo, car_names)

    bottoms, heights = [], []
    for info in infos:
        arr = _frame_car_boxes(info, car_id, car_names)
        if arr is None:
            continue
        if source == 'kitti_cam':
            l2c = np.asarray(info['images']['CAM2']['lidar2cam'], dtype=np.float32)
            cam = CameraInstance3DBoxes(arr, box_dim=7, origin=(0.5, 1.0, 0.5))
            lidar = cam.convert_to(Box3DMode.LIDAR,
                                   rt_mat=torch.from_numpy(np.linalg.inv(l2c)))
        else:
            lidar = LiDARInstance3DBoxes(arr, box_dim=7, origin=origin)
        bottoms.append(lidar.tensor[:, 2].numpy())
        heights.append(lidar.tensor[:, 5].numpy())   # z_size = height, frame-agnostic

    if not bottoms:
        raise RuntimeError(
            f'No car boxes found in {pkl_path}. car_id={car_id}; '
            f'check metainfo["categories"].')

    return np.concatenate(bottoms), np.concatenate(heights)


def summarize(tag, bottom_z, height):
    print(f'  [{tag}]  N={len(bottom_z):>7d}  '
          f'median bottom-z={np.median(bottom_z):+.4f}  '
          f'mean bottom-z={bottom_z.mean():+.4f}  '
          f'(median height={np.median(height):.3f})')
    return float(np.median(bottom_z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nusc', default='data/nuscenes/nuscenes_infos_train.pkl')
    ap.add_argument('--kitti', default='data/kitti/kitti_infos_train.pkl')
    # nuScenes annotations are gravity-center (stored in LiDAR frame).
    ap.add_argument('--nusc-origin', type=float, nargs=3, default=(0.5, 0.5, 0.5))
    args = ap.parse_args()

    print('Measuring car ground level (bottom-z) in each NATIVE LiDAR frame...\n')

    nz, nh = load_bottom_z(args.nusc, {'car', 'vehicle.car'},
                           source='lidar', origin=tuple(args.nusc_origin))
    kz, kh = load_bottom_z(args.kitti, {'car'}, source='kitti_cam')

    nusc_med = summarize('nuScenes', nz, nh)
    kitti_med = summarize('KITTI   ', kz, kh)

    shift = kitti_med - nusc_med
    print('\n' + '=' * 64)
    print(f'  Sanity check  -> nuScenes ~ -1.84 ?  KITTI ~ -1.73 ?')
    print(f'  Data-optimal shift  = median(KITTI) - median(nuScenes)')
    print(f'                      = {kitti_med:+.4f} - ({nusc_med:+.4f})')
    print(f'                      = {shift:.4f} m')
    print('-' * 64)
    print(f'  your current shift : 0.11 m   (|err| = {abs(shift - 0.11):.4f})')
    print(f'  ST3D/CMT shift     : 0.20 m   (|err| = {abs(shift - 0.20):.4f})')
    better = '0.11' if abs(shift - 0.11) <= abs(shift - 0.20) else '0.20'
    print(f'  -> data favors {better};  recommend shift = {shift:.3f} m')
    print('=' * 64)


if __name__ == '__main__':
    main()
