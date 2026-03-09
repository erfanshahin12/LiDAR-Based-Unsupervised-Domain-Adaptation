from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img)

# Import the original KittiMetric.  Adjust the path to match your project.
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric


@METRICS.register_module()
class NuScenesKittiMetric(KittiMetric):
    """Evaluate NuScenes predictions with the KITTI evaluation protocol.

    Inherits all formatting and evaluation logic from :class:`KittiMetric`.
    Only the two NuScenes-incompatible methods are overridden:

    * :meth:`convert_annos_to_kitti_annos` — converts LiDAR-frame NuScenes GT
      to camera-frame KITTI annotations and applies class remapping.
    * :meth:`convert_valid_bboxes` — derives ``lidar2cam`` from the
      NuScenes ``lidar2ego`` / ``cam2ego`` chain instead of reading it from a
      dedicated ``lidar2cam`` key that NuScenes infos do not have.

    Args:
        ann_file (str): Path to the NuScenes pkl annotation file.
        metric (str or List[str]): Metrics to evaluate.  Defaults to 'bbox'.
        pcd_limit_range (List[float]): XYZ range filter applied to predicted
            boxes in LiDAR frame.  **Use the full 360° range for NuScenes**,
            e.g. ``[-50, -50, -5, 50, 50, 3]``.  The KITTI default
            ``[0, -40, -3, 70.4, 40, 0]`` will silently discard most NuScenes
            predictions.
        prefix (str, optional): Metric name prefix. Defaults to None.
        pklfile_prefix (str, optional): Output pkl prefix. Defaults to None.
        default_cam_key (str): Camera key used for projection and image-plane
            filtering.  For NuScenes this should be ``'CAM_FRONT'``.
            Defaults to ``'CAM_FRONT'``.
        format_only (bool): Only format results, skip evaluation.
            Defaults to False.
        submission_prefix (str, optional): Submission output prefix.
            Defaults to None.
        label_mapping (dict, optional): Maps **NuScenes class names** to
            **shared KITTI class names**.  GT instances whose NuScenes class
            is absent from the mapping are silently dropped.  When ``None``
            the metric assumes labels in the pkl are already in the shared
            KITTI index space (i.e. the pipeline already did the remapping).
            Defaults to None.
        collect_device (str): Device for distributed result collection.
            Defaults to ``'cpu'``.
        backend_args (dict, optional): Backend arguments. Defaults to None.
    """

    def __init__(
            self,
            ann_file: str,
            metric: Union[str, List[str]] = 'bbox',
            pcd_limit_range: List[float] = [-50, -50, -5, 50, 50, 3],
            prefix: Optional[str] = None,
            pklfile_prefix: Optional[str] = None,
            default_cam_key: str = 'CAM_FRONT',
            format_only: bool = False,
            submission_prefix: Optional[str] = None,
            label_mapping: Optional[dict] = None,
            collect_device: str = 'cpu',
            backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'NuScenesKitti metric'
        super().__init__(
            ann_file=ann_file,
            metric=metric,
            pcd_limit_range=pcd_limit_range,
            prefix=prefix,
            pklfile_prefix=pklfile_prefix,
            default_cam_key=default_cam_key,
            format_only=format_only,
            submission_prefix=submission_prefix,
            collect_device=collect_device,
            backend_args=backend_args)

        self.label_mapping = label_mapping

        # Ordered KITTI-style class names for kitti_eval()
        if label_mapping is not None:
            seen: dict = {}
            for v in label_mapping.values():
                seen[v] = None
            self.kitti_classes = list(seen.keys())
        else:
            self.kitti_classes = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # NuScenes standard resolution for all 6 cameras.
    # Used as a fallback when the pkl does not store pixel dimensions.
    _NUSCENES_IMG_SHAPE: Dict[str, Tuple[int, int]] = {
        'CAM_FRONT':        (900, 1600),
        'CAM_FRONT_LEFT':   (900, 1600),
        'CAM_FRONT_RIGHT':  (900, 1600),
        'CAM_BACK':         (900, 1600),
        'CAM_BACK_LEFT':    (900, 1600),
        'CAM_BACK_RIGHT':   (900, 1600),
    }

    def _patch_image_shapes(self, data_annos: List[dict]) -> None:
        """Inject ``height`` / ``width`` into camera sub-dicts if absent.

        The parent :meth:`KittiMetric.bbox2result_kitti` reads
        ``info['images'][cam_key]['height']`` and ``['width']`` to clip
        projected 2-D boxes to the image boundary.  NuScenes pkl files do
        not store these fields, so we inject the known fixed resolution.

        The patch is applied in-place and only fills missing keys, so it
        is safe to call even if a future NuScenes pkl version does include
        the dimensions.

        Args:
            data_annos (List[dict]): Entries from ``data_list``, mutated
                in-place.
        """
        for info in data_annos:
            for cam_key, cam_info in info.get('images', {}).items():
                if 'height' not in cam_info or 'width' not in cam_info:
                    h, w = self._NUSCENES_IMG_SHAPE.get(
                        cam_key,
                        (900, 1600))   # safe default for unknown cameras
                    cam_info.setdefault('height', h)
                    cam_info.setdefault('width', w)

    def _get_lidar2cam(self, info: dict) -> np.ndarray:
        """Derive the 4×4 lidar-to-camera matrix from NuScenes info.

        NuScenes stores separate ``lidar2ego`` and ``cam2ego`` transforms.
        The combined lidar→camera matrix is::

            lidar2cam = inv(cam2ego) @ lidar2ego

        Args:
            info (dict): A single NuScenes ``data_list`` entry.

        Returns:
            np.ndarray: Shape (4, 4), dtype float32.
        """
        lidar2ego = np.array(
            info['lidar_points']['lidar2ego'], dtype=np.float32)  # (4,4)
        cam2ego = np.array(
            info['images'][self.default_cam_key]['cam2ego'],
            dtype=np.float32)                                      # (4,4)
        lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
        return lidar2cam

    def _build_nuscenes_label_remap(
            self,
            label2cat: dict) -> Dict[int, Optional[str]]:
        """Build a mapping from NuScenes integer label → KITTI class name.

        Args:
            label2cat (dict): ``{label_int: nuscenes_class_name}`` from
                the pkl metainfo (``categories`` dict, inverted).

        Returns:
            dict: ``{nuscenes_label_int: kitti_class_name_or_None}``.
                  A ``None`` value means "drop this instance".
        """
        remap: Dict[int, Optional[str]] = {}
        for nus_label_idx, nus_name in label2cat.items():
            remap[nus_label_idx] = self.label_mapping.get(nus_name, None)
        return remap

    # ------------------------------------------------------------------
    # Overridden: compute_metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        # Override to swap dataset_meta['classes'] for KITTI-style names
        # before calling the parent.  kitti_eval() has a hardcoded
        # name_to_class dict that KeyErrors on NuScenes lowercase names.
        if self.kitti_classes is not None:
            original_classes = self.dataset_meta.get('classes')
            self.dataset_meta['classes'] = self.kitti_classes
            try:
                return super().compute_metrics(results)
            finally:
                self.dataset_meta['classes'] = original_classes
        return super().compute_metrics(results)

    
    # ------------------------------------------------------------------
    # Overridden: GT annotation conversion
    # ------------------------------------------------------------------

    def convert_annos_to_kitti_annos(self, data_infos: dict) -> List[dict]:
        """Convert NuScenes GT annotations to KITTI-style camera-frame dicts.

        Each NuScenes ``instance`` contains:

        * ``bbox_label`` — integer index into the NuScenes class list.
        * ``bbox_3d`` — ``[x, y, z, l, w, h, yaw]`` in LiDAR frame
          (:class:`LiDARInstance3DBoxes` convention: bottom-centre origin).

        The KITTI evaluator expects per-sample dicts with keys
        ``name / truncated / occluded / alpha / bbox / dimensions /
        location / rotation_y / score``, all in **camera frame**.

        Because NuScenes has no 2-D bounding-box or visibility annotations:

        * ``truncated`` and ``occluded`` are set to 0.
        * ``bbox`` (2-D image box) is set to a dummy ``[0, 0, 50, 50]``;
          KITTI 3D / BEV evaluation does not use it.
        * ``alpha`` (observation angle) is derived analytically from the
          camera-frame 3-D box.
        * ``score`` is set to -1 (GT annotations have no confidence).

        Args:
            data_infos (dict): Loaded pkl with ``'data_list'`` and
                ``'metainfo'``.

        Returns:
            List[dict]: The same ``data_list`` entries, each augmented with
            a ``'kitti_annos'`` key.
        """
        data_annos = data_infos['data_list']
        if self.format_only:
            return data_annos

        cat2label = data_infos['metainfo']['categories']   # name -> label_int
        label2cat = dict((v, k) for (k, v) in cat2label.items())   # label_int -> name

        assert 'instances' in data_annos[0], (
            "NuScenes data_list entries must contain an 'instances' key. "
            "Make sure you are pointing at a NuScenes pkl annotation file.")

        # Build the integer-level remap once so we don't repeat dict look-ups.
        if self.label_mapping is not None:
            idx_remap = self._build_nuscenes_label_remap(label2cat)
        else:
            # Pipeline already remapped labels to the shared KITTI space.
            # label2cat then directly gives the KITTI class names.
            idx_remap = None

        _EMPTY_ANNO = dict(
            name=np.array([]),
            truncated=np.array([]),
            occluded=np.array([]),
            alpha=np.array([]),
            bbox=np.zeros([0, 4]),
            dimensions=np.zeros([0, 3]),
            location=np.zeros([0, 3]),
            rotation_y=np.array([]),
            score=np.array([]),
        )

        for i, annos in enumerate(data_annos):
            if len(annos['instances']) == 0:
                data_annos[i]['kitti_annos'] = _EMPTY_ANNO.copy()
                continue

            # Lidar→camera for this sample (derived from NuScenes extrinsics).
            lidar2cam = self._get_lidar2cam(annos)

            kitti_annos: Dict[str, list] = {
                'name': [], 'truncated': [], 'occluded': [], 'alpha': [],
                'bbox': [], 'location': [], 'dimensions': [],
                'rotation_y': [], 'score': [],
            }

            for instance in annos['instances']:
                nus_label: int = instance['bbox_label']

                # ---- label resolution --------------------------------
                if idx_remap is not None:
                    kitti_name = idx_remap.get(nus_label, None)
                else:
                    # Labels already in KITTI space; just resolve the name.
                    kitti_name = label2cat.get(nus_label, None)

                if kitti_name is None:
                    # Class not in the shared label space → skip.
                    continue

                # ---- 3-D box conversion: LiDAR → camera frame --------
                # NuScenes / mmdet3d LiDAR convention:
                #   bbox_3d = [x, y, z, l, w, h, yaw]
                #   origin at bottom-centre, yaw around Z-up
                bbox_3d_lidar = np.array(
                    instance['bbox_3d'], dtype=np.float32)   # (7,)

                lidar_box = LiDARInstance3DBoxes(
                    torch.tensor(bbox_3d_lidar[None], dtype=torch.float32))
                cam_box = lidar_box.convert_to(
                    Box3DMode.CAM, lidar2cam, correct_yaw=True)

                # cam_box.tensor layout: [x, y, z, l, w, h, ry]
                # (camera frame, bottom-centre origin)
                cam_t = cam_box.tensor[0].numpy()   # (7,)
                loc = cam_t[:3]                      # x, y, z  (cam)
                # KITTI dimension order: h (up), w (lateral), l (forward)
                dims_hwl = cam_t[[5, 4, 3]]          # h=cam_t[5], w=[4], l=[3]
                ry = cam_t[6]                        # yaw in camera frame

                # ---- derived KITTI fields ----------------------------
                # Alpha: signed angle between observation ray and vehicle front
                alpha = float(-np.arctan2(loc[0], loc[2]) + ry)

                # 2-D bbox: not available in NuScenes; KITTI 3D/BEV eval
                # ignores it, so a small dummy box is safe.
                dummy_bbox = np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32)

                kitti_annos['name'].append(kitti_name)
                kitti_annos['truncated'].append(0.0)
                kitti_annos['occluded'].append(0)
                kitti_annos['alpha'].append(alpha)
                kitti_annos['bbox'].append(dummy_bbox)
                kitti_annos['location'].append(loc.astype(np.float32))
                kitti_annos['dimensions'].append(dims_hwl.astype(np.float32))
                kitti_annos['rotation_y'].append(float(ry))
                # GT has no confidence score; -1 is ignored by kitti_eval.
                kitti_annos['score'].append(-1.0)

            if len(kitti_annos['name']) == 0:
                data_annos[i]['kitti_annos'] = _EMPTY_ANNO.copy()
            else:
                data_annos[i]['kitti_annos'] = {
                    k: np.array(v) for k, v in kitti_annos.items()
                }

        # Ensure height/width exist in every camera sub-dict so the parent
        # bbox2result_kitti can clip 2-D projected boxes without KeyError.
        self._patch_image_shapes(data_annos)

        return data_annos

    # ------------------------------------------------------------------
    # Overridden: predicted box conversion
    # ------------------------------------------------------------------

    def convert_valid_bboxes(self, box_dict: dict, info: dict) -> dict:
        """Convert predicted boxes to KITTI format using NuScenes extrinsics.

        Mirrors :meth:`KittiMetric.convert_valid_bboxes` but computes
        ``lidar2cam`` via :meth:`_get_lidar2cam` instead of reading a
        ``lidar2cam`` key that does not exist in NuScenes info dicts.

        Args:
            box_dict (dict): Model predictions for one sample:

                * ``bboxes_3d`` — :class:`LiDARInstance3DBoxes` or
                  :class:`CameraInstance3DBoxes`.
                * ``scores_3d`` — confidence scores.
                * ``labels_3d`` — class indices.

            info (dict): NuScenes sample info dict (one entry from
                ``data_list``).

        Returns:
            dict: Filtered predictions with keys:
            ``bbox / pred_box_type_3d / box3d_camera / box3d_lidar /
            scores / label_preds / sample_idx``.
        """
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['sample_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        _empty = dict(
            bbox=np.zeros([0, 4]),
            pred_box_type_3d=type(box_preds),
            box3d_camera=np.zeros([0, 7]),
            box3d_lidar=np.zeros([0, 7]),
            scores=np.zeros([0]),
            label_preds=np.zeros([0]),
            sample_idx=sample_idx,
        )

        if len(box_preds) == 0:
            return _empty

        # ---- Extrinsics / intrinsics from NuScenes info --------------
        lidar2cam = self._get_lidar2cam(info)
        cam_info = info['images'][self.default_cam_key]
        P2 = box_preds.tensor.new_tensor(
            np.array(cam_info['cam2img'], dtype=np.float32))
        img_shape = (cam_info['height'], cam_info['width'])

        # ---- Convert to camera frame --------------------------------
        if isinstance(box_preds, LiDARInstance3DBoxes):
            box_preds_camera = box_preds.convert_to(
                Box3DMode.CAM, lidar2cam, correct_yaw=True)
            box_preds_lidar = box_preds
        elif isinstance(box_preds, CameraInstance3DBoxes):
            box_preds_camera = box_preds
            box_preds_lidar = box_preds.convert_to(
                Box3DMode.LIDAR, np.linalg.inv(lidar2cam), correct_yaw=True)
        else:
            raise TypeError(
                f'Unsupported box type for NuScenesKittiMetric: '
                f'{type(box_preds)}.  Expected LiDARInstance3DBoxes or '
                f'CameraInstance3DBoxes.')

        # ---- Image-plane validity check -----------------------------
        box_corners = box_preds_camera.corners            # (N, 8, 3)
        box_corners_in_image = points_cam2img(box_corners, P2)  # (N, 8, 2)
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)  # (N, 4)

        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = (
            (box_2d_preds[:, 0] < image_shape[1]) &
            (box_2d_preds[:, 1] < image_shape[0]) &
            (box_2d_preds[:, 2] > 0) &
            (box_2d_preds[:, 3] > 0))

        # ---- Point-cloud range validity check -----------------------
        if isinstance(box_preds, LiDARInstance3DBoxes):
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_pcd_inds = (
                (box_preds_lidar.center > limit_range[:3]) &
                (box_preds_lidar.center < limit_range[3:]))
            valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
        else:
            valid_inds = valid_cam_inds

        if valid_inds.sum() == 0:
            return _empty

        return dict(
            bbox=box_2d_preds[valid_inds, :].numpy(),
            pred_box_type_3d=type(box_preds),
            box3d_camera=box_preds_camera[valid_inds].numpy(),
            box3d_lidar=box_preds_lidar[valid_inds].numpy(),
            scores=scores[valid_inds].numpy(),
            label_preds=labels[valid_inds].numpy(),
            sample_idx=sample_idx,
        )
