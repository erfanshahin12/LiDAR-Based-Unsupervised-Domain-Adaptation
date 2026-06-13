"""
Hard Instance Mining for Domain Adaptive 3D Object Detection

Implements hard instance mining (CMT-style, online):
1. Builds a bank of ALL source instances in-process from the standard dbinfos
   pkl — no separate offline build script needed, no static threshold.
2. The "hard" threshold (Q-th percentile of point counts inside pseudo-label
   boxes in the current epoch) is pushed into the transform each epoch by
   PseudoLabelRefreshHook, matching CMT's per-scene live computation.
3. Injects hard instances into target-domain strong-aug scenes during training,
   with CMT-style point carving so the inserted object replaces existing returns.
4. Optionally normalises inserted box/point sizes toward the target domain
   (e.g. shrinks nuScenes cars toward KITTI dimensions).
"""

import ctypes
import functools
import multiprocessing
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmcv.transforms import BaseTransform


# ---------------------------------------------------------------------------
# Module-level LRU cache so multiple dataloader workers in the same process
# share one bank object instead of rebuilding from disk each time.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _cached_build_bank(source_db_path: str,
                       classes_tuple: tuple,
                       mapping_items: Optional[tuple]) -> 'HardInstanceBank':
    """Build and cache a HardInstanceBank keyed by its construction parameters."""
    source_class_mapping = dict(mapping_items) if mapping_items else None
    return build_hard_instance_bank(
        source_db_path=source_db_path,
        classes=list(classes_tuple) if classes_tuple else None,
        source_class_mapping=source_class_mapping,
    )


# ---------------------------------------------------------------------------
# HardInstanceBank
# ---------------------------------------------------------------------------

class HardInstanceBank:
    """Bank storing ALL source instances for dynamic hard-instance filtering.

    The "hard" threshold (Q-th percentile of point counts inside the current
    epoch's pseudo-label boxes) is determined externally by
    PseudoLabelRefreshHook and pushed into HardInstanceSampling.  This bank
    just stores every instance so the threshold can be applied at runtime.
    """

    def __init__(self,
                 source_db_infos: Dict[str, List],
                 classes: List[str] = None):
        self.classes = classes or list(source_db_infos.keys())
        self.all_instances: Dict[str, List] = {}

        for cls_name in self.classes:
            if cls_name not in source_db_infos:
                print(f'Warning: {cls_name} not found in source database')
                continue
            self.all_instances[cls_name] = source_db_infos[cls_name]
            print(f'Hard Instance Bank - {cls_name}: '
                  f'{len(source_db_infos[cls_name])} instances loaded')

    def sample(self, cls_name: str, num_samples: int = 1,
               threshold: Optional[float] = None) -> List[Dict]:
        """Sample instances, optionally filtering to those with fewer than
        ``threshold`` interior points (the "hard" subset).

        When ``threshold`` is None all instances are eligible.
        """
        if cls_name not in self.all_instances:
            return []
        available = self.all_instances[cls_name]
        if threshold is not None:
            available = [s for s in available
                         if s['num_points_in_gt'] < threshold]
        if len(available) == 0:
            return []
        num_samples = min(num_samples, len(available))
        indices = np.random.choice(len(available), num_samples, replace=False)
        return [available[i] for i in indices]

    def get_stats(self) -> Dict[str, Dict]:
        stats = {}
        for cls_name, samples in self.all_instances.items():
            if len(samples) > 0:
                point_counts = [s['num_points_in_gt'] for s in samples]
                stats[cls_name] = {
                    'count': len(samples),
                    'mean_points': np.mean(point_counts),
                    'median_points': np.median(point_counts),
                    'min_points': np.min(point_counts),
                    'max_points': np.max(point_counts),
                }
        return stats


# ---------------------------------------------------------------------------
# HardInstanceSampling transform
# ---------------------------------------------------------------------------

@TRANSFORMS.register_module()
class HardInstanceSampling(BaseTransform):
    """Inject hard source-domain instances into target-domain scenes (CMT-style).

    Builds the hard-instance bank in-process from the standard dbinfos pkl
    (no separate offline build step).  Each call:
      1. Samples hard source instances by class.
      2. Optionally normalises instance size toward the target domain.
      3. Carves (removes) existing target points in the enlarged box region.
      4. Injects instance points (object-local → global via box pose).
      5. Appends injected boxes / labels to ``results['gt_bboxes_3d']`` /
         ``results['gt_labels_3d']`` so the student is supervised on them.

    The transform always emits ``gt_bboxes_3d`` / ``gt_labels_3d`` keys (empty
    when nothing is injected) so downstream ``Pack3DDetInputs`` never KeyErrors.

    Args:
        hard_instance_bank: Pre-built ``HardInstanceBank`` object (optional).
        hard_instance_bank_path: Path to a pickled ``HardInstanceBank`` (optional).
        source_db_path: Path to source dbinfos pkl — builds the bank in-process
            when neither of the above is supplied.
        source_class_mapping: Maps target class names to one-or-more source
            class names, e.g. ``dict(Car='car')`` or ``dict(Cyclist=['bicycle',
            'motorcycle'])``.
        db_path_prefix: Prepended to relative ``sample['path']`` entries from
            the dbinfos so point files can be located without an offline fix-up.
        sample_groups: ``{cls_name: num_samples}`` dict, e.g. ``dict(Car=5)``.
        use_pred_boxes_for_collision: Use ``pred_bboxes_3d`` (predictions) for
            collision detection instead of ``gt_bboxes_3d``.
        iou_thresh: IoU threshold above which an instance is skipped (collision).
        carve: Remove existing target points inside the (enlarged) injected-box
            region before concatenating instance points (CMT carving).
        carve_extra_width: ``(dl, dw, dh)`` — each dimension grows by
            ``2 * extra`` so the carved zone is slightly larger than the box.
        size_normalize: Optional dict with ``size_res`` (absolute residuals
            ``[dl, dw, dh]``) applied to *shrink* the injected box and its
            local points toward the target domain.  Matches the source pipeline's
            ``normalize_object_size SIZE_RES`` step.
        class_names: Ordered list of class names used for label-index lookup.
        points_loader: Config dict for a mmcv transform that loads points from
            a file path (e.g. ``dict(type='LoadPointsFromFile', ...)``).

    Note on the "hard" threshold:
        The threshold (Q-th percentile of point counts inside pseudo-label boxes
        for the current epoch) is NOT set here.  It is computed by
        ``PseudoLabelRefreshHook`` after each pseudo-label refresh and pushed
        via ``set_point_threshold()``.  Until the first refresh the bank
        accepts all source instances (threshold = +inf).
    """

    def __init__(self,
                 hard_instance_bank=None,
                 hard_instance_bank_path: Optional[str] = None,
                 # --- in-process build args ---
                 source_db_path: Optional[str] = None,
                 source_class_mapping: Optional[Dict[str, Union[str, List[str]]]] = None,
                 db_path_prefix: Optional[str] = None,
                 # --- injection args ---
                 sample_groups: Optional[Dict[str, int]] = None,
                 use_pred_boxes_for_collision: bool = False,
                 iou_thresh: float = 0.3,
                 carve: bool = True,
                 carve_extra_width: Tuple[float, float, float] = (1.0, 0.5, 0.5),
                 size_normalize: Optional[Dict] = None,
                 class_names: List[str] = None,
                 points_loader: Optional[Dict] = None):

        self.sample_groups = sample_groups or {}
        self.use_pred_boxes_for_collision = use_pred_boxes_for_collision
        self.iou_thresh = iou_thresh
        self.carve = carve
        self.carve_extra_width = np.asarray(carve_extra_width, dtype=np.float32)
        self.size_normalize = size_normalize
        self.class_names = class_names or []
        self.db_path_prefix = db_path_prefix

        # ── Build / load the bank ────────────────────────────────────────────
        if hard_instance_bank is not None:
            self.hard_instance_bank = hard_instance_bank

        elif hard_instance_bank_path is not None:
            print(f'Loading hard instance bank from: {hard_instance_bank_path}')
            with open(hard_instance_bank_path, 'rb') as f:
                self.hard_instance_bank = pickle.load(f)
            print(f'Loaded bank with classes: '
                  f'{list(self.hard_instance_bank.all_instances.keys())}')

        elif source_db_path is not None:
            # In-process build — uses lru_cache so worker processes don't rebuild.
            mapping_items = (tuple(sorted(source_class_mapping.items()))
                             if source_class_mapping else None)
            classes_tuple = (tuple(sorted(source_class_mapping.keys()))
                             if source_class_mapping else None)
            self.hard_instance_bank = _cached_build_bank(
                source_db_path=source_db_path,
                classes_tuple=classes_tuple,
                mapping_items=mapping_items,
            )
            # Resolve relative DB paths to absolute using db_path_prefix.
            if db_path_prefix is not None:
                prefix = Path(db_path_prefix)
                for cls_samples in self.hard_instance_bank.all_instances.values():
                    for s in cls_samples:
                        p = Path(s['path'])
                        if not p.is_absolute():
                            s['path'] = str(prefix / p)
        else:
            raise ValueError(
                'Provide hard_instance_bank, hard_instance_bank_path, '
                'or source_db_path to build the bank in-process.')

        # ── Shared-memory threshold (visible across forked worker processes) ──
        # PseudoLabelRefreshHook writes the Q-th percentile of pseudo-label
        # box point counts here each epoch.  +inf until the first refresh so
        # all source instances are eligible before pseudo-labels exist.
        self._point_threshold = multiprocessing.Value(ctypes.c_double, float('inf'))

        # ── Points loader ────────────────────────────────────────────────────
        if points_loader:
            self.points_loader_transform = TRANSFORMS.build(points_loader)
        else:
            self.points_loader_transform = None

    def set_point_threshold(self, threshold: float) -> None:
        """Update the hard-instance point-count threshold.

        Called by PseudoLabelRefreshHook after each pseudo-label refresh.
        Uses shared memory so the update is visible to all persistent dataloader
        worker processes (fork-based multiprocessing on Linux).
        """
        self._point_threshold.value = float(threshold)

    # ── Public transform entry point ─────────────────────────────────────────

    def transform(self, results: Dict) -> Dict:
        """Inject hard instances into the target scene."""
        if not self.sample_groups:
            # Nothing to inject — still emit empty GT keys.
            self._ensure_gt_keys(results)
            return results

        threshold = self._point_threshold.value
        if threshold == float('inf'):
            threshold = None  # no filtering before first pseudo-label refresh

        # Existing boxes for collision detection.
        if self.use_pred_boxes_for_collision and 'pred_bboxes_3d' in results:
            existing_boxes = results['pred_bboxes_3d']
        elif 'gt_bboxes_3d' in results:
            existing_boxes = results['gt_bboxes_3d']
        else:
            existing_boxes = None

        sampled_boxes: List[np.ndarray] = []
        sampled_labels: List[int] = []
        sampled_points_list: List[np.ndarray] = []
        skipped = 0

        for cls_name, num_samples in self.sample_groups.items():
            instances = self.hard_instance_bank.sample(cls_name, num_samples,
                                                       threshold=threshold)
            for inst in instances:
                pts = self._load_instance_points(inst)
                if pts is None:
                    skipped += 1
                    continue

                box = self._get_instance_box(inst)   # [1, 7+]
                label = self._get_instance_label(inst, cls_name)

                # Optional size normalisation toward target domain.
                if self.size_normalize is not None:
                    pts, box = self._apply_size_normalize(pts, box)

                if existing_boxes is not None and self._has_collision(box, existing_boxes):
                    continue

                sampled_boxes.append(box)
                sampled_labels.append(label)
                sampled_points_list.append(pts)

        if skipped > 0:
            print(f'[HardInstanceSampling] Skipped {skipped} instances '
                  f'(missing point files)')

        if len(sampled_boxes) > 0:
            results = self._merge_instances(
                results, sampled_boxes, sampled_labels, sampled_points_list)
        else:
            self._ensure_gt_keys(results)

        return results

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _ensure_gt_keys(self, results: Dict) -> None:
        """Emit empty GT keys when nothing is injected so Pack3DDetInputs works."""
        if 'gt_bboxes_3d' not in results:
            results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                torch.zeros((0, 7), dtype=torch.float32), origin=(0.5, 0.5, 0.5))
        if 'gt_labels_3d' not in results:
            results['gt_labels_3d'] = np.array([], dtype=np.int64)

    def _load_instance_points(self, sample: Dict) -> Optional[np.ndarray]:
        """Load per-instance point cloud from the database path."""
        pts_path = sample['path']
        if not Path(pts_path).exists():
            return None
        try:
            if self.points_loader_transform is not None:
                res = {'lidar_points': {'lidar_path': pts_path}}
                res = self.points_loader_transform(res)
                pts = res.get('points', None)
                if hasattr(pts, 'tensor'):
                    pts = pts.tensor
                if isinstance(pts, torch.Tensor):
                    pts = pts.numpy()
                return pts
            else:
                pts = np.fromfile(pts_path, dtype=np.float32)
                if pts.size == 0:
                    return None
                if 'num_points_in_gt' in sample:
                    n = sample['num_points_in_gt']
                    dim = len(pts) // max(n, 1)
                    pts = pts.reshape(-1, dim)
                else:
                    pts = pts.reshape(-1, 4)
                return pts
        except Exception as e:
            print(f'[HardInstanceSampling] Error loading {pts_path}: {e}')
            return None

    def _get_instance_box(self, sample: Dict) -> np.ndarray:
        box = sample['box3d_lidar']
        return np.array(box, dtype=np.float32).reshape(1, -1)  # [1, 7+]

    def _get_instance_label(self, sample: Dict, cls_name: str) -> int:
        if 'label' in sample:
            return int(sample['label'])
        if cls_name in self.class_names:
            return self.class_names.index(cls_name)
        print(f'[HardInstanceSampling] Warning: {cls_name} not in class_names, using 0')
        return 0

    def _apply_size_normalize(self,
                               pts: np.ndarray,
                               box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale instance points and box dims toward target domain.

        ``size_normalize`` must contain ``size_res: [dl, dw, dh]``, the same
        residuals used by the source pipeline's ``normalize_object_size`` step
        (negative values shrink).  Scaling is applied about the box centre
        so global placement (via ``_transform_points_to_global``) is unaffected.
        """
        size_res = np.asarray(self.size_normalize['size_res'], dtype=np.float32)
        # box[:, 3:6] = [l, w, h]
        orig_dims = box[0, 3:6].copy()                   # [l, w, h]
        new_dims  = np.maximum(orig_dims + size_res, 0.1)  # floor at 0.1 m
        scale = new_dims / np.maximum(orig_dims, 1e-6)     # [sl, sw, sh]

        # Scale local xyz (object-local coords are centred at 0,0,0 for db pts).
        if isinstance(pts, torch.Tensor):
            pts = pts.numpy()
        pts = pts.copy()
        pts[:, 0] *= scale[0]   # x → l-axis
        pts[:, 1] *= scale[1]   # y → w-axis
        pts[:, 2] *= scale[2]   # z → h-axis

        box = box.copy()
        box[0, 3:6] = new_dims
        return pts, box

    def _has_collision(self, new_box: np.ndarray,
                       existing_boxes: LiDARInstance3DBoxes) -> bool:
        if len(existing_boxes) == 0:
            return False
        origin = getattr(existing_boxes, 'origin', (0.5, 0.5, 0.5))
        new_box_obj = LiDARInstance3DBoxes(
            torch.from_numpy(new_box[:, :7]).float(),
            box_dim=7, origin=origin)
        ious = LiDARInstance3DBoxes.overlaps(new_box_obj, existing_boxes, mode='iou')
        return bool((ious > self.iou_thresh).any().item())

    def _transform_points_to_global(self,
                                     pts: np.ndarray,
                                     box: np.ndarray) -> np.ndarray:
        """Rotate + translate object-local pts to the scene's global frame."""
        if isinstance(box, torch.Tensor):
            box = box.numpy()
        box = np.asarray(box, dtype=np.float32).reshape(-1)
        x, y, z, _l, _w, _h, yaw = box[:7]

        if hasattr(pts, 'tensor'):
            pts = pts.tensor
        if isinstance(pts, torch.Tensor):
            pts = pts.numpy()
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[0] == 0:
            return pts

        local_xyz = pts[:, :3]
        rest      = pts[:, 3:]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_y, -sin_y, 0],
                        [sin_y,  cos_y, 0],
                        [0,      0,     1]], dtype=np.float32)
        global_xyz = local_xyz @ rot.T + np.array([x, y, z], dtype=np.float32)
        return np.concatenate([global_xyz, rest], axis=1)

    def _merge_instances(self,
                         results: Dict,
                         new_boxes: List[np.ndarray],
                         new_labels: List[int],
                         new_points_list: List[np.ndarray]) -> Dict:
        """Carve existing returns, inject instance points, append GT."""
        new_boxes_np = np.concatenate(new_boxes, axis=0)   # [N, 7+]
        new_labels_np = np.array(new_labels, dtype=np.int64)   # [N]

        # ── Convert scene points to numpy ─────────────────────────────────
        orig_pts = results['points']
        if hasattr(orig_pts, 'tensor'):
            pts_np = orig_pts.tensor.numpy().copy()
        elif isinstance(orig_pts, torch.Tensor):
            pts_np = orig_pts.numpy().copy()
        else:
            pts_np = np.asarray(orig_pts, dtype=np.float32).copy()

        # ── Determine box origin (consistent with scene after KittiToNuscenes) ─
        if 'gt_bboxes_3d' in results and hasattr(results['gt_bboxes_3d'], 'origin'):
            origin = results['gt_bboxes_3d'].origin
        else:
            origin = (0.5, 0.5, 0.5)   # nuScenes frame default

        # ── CMT carving: remove scene points inside enlarged injected boxes ──
        if self.carve and new_boxes_np.shape[0] > 0:
            from mmdet3d.structures.ops.box_np_ops import points_in_rbbox
            carved_boxes = new_boxes_np[:, :7].copy()
            carved_boxes[:, 3:6] += 2.0 * self.carve_extra_width   # grow l,w,h
            # points_in_rbbox expects bottom-centre origin (0.5, 0.5, 0).
            # After KittiToNuscenes the boxes sit at nuScenes origin (0.5,0.5,0.5).
            # Shift z-centre down by h/2 to match points_in_rbbox convention.
            carved_boxes_bc = carved_boxes.copy()
            carved_boxes_bc[:, 2] -= carved_boxes_bc[:, 5] / 2.0
            inside = points_in_rbbox(
                pts_np[:, :3], carved_boxes_bc, z_axis=2, origin=(0.5, 0.5, 0))
            keep = ~inside.any(axis=1)
            pts_np = pts_np[keep]

        # ── Inject instance points ────────────────────────────────────────
        for i, inst_pts in enumerate(new_points_list):
            global_pts = self._transform_points_to_global(inst_pts, new_boxes_np[i])
            pts_np = np.concatenate([pts_np, global_pts], axis=0)

        # ── Restore original points container type ─────────────────────────
        if hasattr(orig_pts, 'tensor'):
            results['points'] = orig_pts.__class__(
                torch.from_numpy(pts_np).float(),
                points_dim=pts_np.shape[-1],
                attribute_dims=getattr(orig_pts, 'attribute_dims', None))
        elif isinstance(orig_pts, torch.Tensor):
            results['points'] = torch.from_numpy(pts_np).float()
        else:
            results['points'] = pts_np

        # ── Merge boxes and labels ─────────────────────────────────────────
        new_boxes_obj = LiDARInstance3DBoxes(
            torch.from_numpy(new_boxes_np[:, :7]).float(),
            box_dim=7, origin=origin)

        if 'gt_bboxes_3d' in results and len(results['gt_bboxes_3d']) > 0:
            results['gt_bboxes_3d'] = results['gt_bboxes_3d'].cat(
                [results['gt_bboxes_3d'], new_boxes_obj])
            existing_labels = results['gt_labels_3d']
            if isinstance(existing_labels, torch.Tensor):
                results['gt_labels_3d'] = torch.cat(
                    [existing_labels,
                     torch.from_numpy(new_labels_np).long()])
            else:
                results['gt_labels_3d'] = np.concatenate(
                    [existing_labels, new_labels_np])
        else:
            results['gt_bboxes_3d'] = new_boxes_obj
            results['gt_labels_3d'] = new_labels_np

        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'sample_groups={self.sample_groups}, '
                f'carve={self.carve}, '
                f'size_normalize={self.size_normalize}, '
                f'iou_thresh={self.iou_thresh})')


# ---------------------------------------------------------------------------
# build_hard_instance_bank  (also used by _cached_build_bank above)
# ---------------------------------------------------------------------------

def build_hard_instance_bank(source_db_path: str,
                              classes: List[str] = None,
                              source_class_mapping: Optional[Dict] = None,
                              ) -> HardInstanceBank:
    """Build a HardInstanceBank from the source dbinfos pkl.

    All source instances are stored; the "hard" threshold is applied at
    runtime by ``HardInstanceSampling`` (pushed each epoch by
    ``PseudoLabelRefreshHook``).

    Args:
        source_db_path: Path to source domain dbinfos pkl (e.g. nuScenes).
        classes: List of class names to include (defaults to all in source db).
        source_class_mapping: Maps target class names to source class name(s),
            e.g. ``{'Car': 'car', 'Cyclist': ['bicycle', 'motorcycle']}``.

    Returns:
        A populated ``HardInstanceBank``.
    """
    with open(source_db_path, 'rb') as f:
        source_db_infos = pickle.load(f)

    if source_class_mapping:
        print('Remapping source database classes:')
        remapped: Dict[str, List] = {}
        for tgt_name, src_names in source_class_mapping.items():
            remapped[tgt_name] = []
            if isinstance(src_names, str):
                src_names = [src_names]
            for src in src_names:
                if src in source_db_infos:
                    remapped[tgt_name].extend(source_db_infos[src])
                    print(f"  '{src}' → '{tgt_name}': "
                          f"{len(source_db_infos[src])} samples")
                else:
                    print(f"  Warning: source class '{src}' not found in database")
        source_db_infos = remapped

    bank = HardInstanceBank(source_db_infos=source_db_infos, classes=classes)

    print('\n' + '=' * 60)
    print('Hard Instance Bank Statistics:')
    print('=' * 60)
    for cls_name, s in bank.get_stats().items():
        print(f'\n{cls_name}:')
        print(f"  Count: {s['count']}")
        print(f"  Mean points: {s['mean_points']:.1f}")
        print(f"  Median points: {s['median_points']:.1f}")
        print(f"  Range: [{s['min_points']}, {s['max_points']}]")
    print('=' * 60 + '\n')

    return bank
