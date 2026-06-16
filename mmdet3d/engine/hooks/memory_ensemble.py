"""Memory Ensemble & Voting (MEV) for pseudo-label refresh.

Port of ST3D's ``consistency_ensemble`` + memory-voting
(``ST3D/pcdet/utils/memory_ensemble_utils.py``), adapted to the
MMDetection3D pseudo-label store format used by
:class:`~mmdet3d.engine.hooks.PseudoLabelRefreshHook`.

The mechanism is **model-agnostic**: it operates only on per-frame box
dicts (``[N, 7]`` LiDAR boxes + parallel score/label/count arrays), with no
dependence on the detector or its head.  At every refresh it matches the new
teacher predictions against the historical memory bank by rotated-3D-IoU and:

* **matched** boxes (max-IoU ≥ ``iou_thresh``) keep the higher-confidence box
  and reset their unmatched counter to 0;
* **disappeared** memory boxes (no match) have their counter incremented, are
  flagged *ignore* (label → -1) at ``ignore_thresh`` and *removed* at
  ``rm_thresh``;
* **new** boxes that match nothing in memory are appended with counter 0.

This gives pseudo-labels temporal consistency: a box must persist across
refreshes to stay trusted, so the confidently-drifting boxes that drive
Mean-Teacher confirmation collapse fade out instead of being reinforced.

Store entry schema (one per frame, as produced by the refresh hook)::

    {
        'gt_boxes':   np.ndarray [N, 7]  float32  (x,y,z,dx,dy,dz,heading),
        'gt_labels':  np.ndarray [N]     int64,
        'scores':     np.ndarray [N]     float32  (ranking score),
        'iou_scores': np.ndarray [N]     float32,
        'cls_scores': np.ndarray [N]     float32  or None,
        'pt_counts':  np.ndarray [N]     int32    or None,
        'memory_counter': np.ndarray [N] int32,   # added/maintained by MEV
    }
"""

from typing import Optional

import numpy as np
import torch

from mmdet3d.structures.ops.iou3d_calculator import bbox_overlaps_3d

# Per-box array fields carried alongside ``gt_boxes`` through the merge.
# ``cls_scores``/``pt_counts`` may be None for a frame; handled per-key.
_ARRAY_KEYS = ('gt_labels', 'scores', 'iou_scores', 'cls_scores',
               'pt_counts', 'memory_counter')


def _ensure_counter(entry: dict) -> dict:
    """Return ``entry`` with a ``memory_counter`` array (zeros if absent)."""
    n = len(entry['gt_boxes'])
    if entry.get('memory_counter') is None:
        entry = dict(entry)
        entry['memory_counter'] = np.zeros(n, dtype=np.int32)
    return entry


def _index_entry(entry: dict, idx) -> dict:
    """Index every per-box array in ``entry`` by ``idx`` (None-safe)."""
    out = {'gt_boxes': entry['gt_boxes'][idx]}
    for k in _ARRAY_KEYS:
        v = entry.get(k)
        out[k] = v[idx] if v is not None else None
    return out


def _concat_entries(a: dict, b: dict) -> dict:
    """Concatenate two entries box-wise (None-safe per key)."""
    out = {'gt_boxes': np.concatenate([a['gt_boxes'], b['gt_boxes']], axis=0)}
    for k in _ARRAY_KEYS:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            out[k] = None
        else:
            out[k] = np.concatenate([va, vb], axis=0)
    return out


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray,
                device: str = 'cuda') -> np.ndarray:
    """Rotated 3D IoU matrix ``[Na, Nb]`` between two ``[N, 7]`` box sets."""
    use_cuda = device == 'cuda' and torch.cuda.is_available()
    dev = 'cuda' if use_cuda else 'cpu'
    ta = torch.from_numpy(boxes_a[:, :7].astype(np.float32)).to(dev)
    tb = torch.from_numpy(boxes_b[:, :7].astype(np.float32)).to(dev)
    iou = bbox_overlaps_3d(ta, tb, mode='iou', coordinate='lidar')
    return iou.cpu().numpy()


def memory_ensemble_frame(prev: Optional[dict], new: dict,
                          cfg: dict) -> dict:
    """Merge one frame's new predictions into its memory bank.

    Args:
        prev: Previous-refresh memory-bank entry for this frame, or ``None``
            (first refresh / frame unseen before).  Must carry
            ``memory_counter`` when not None.
        new: Current-refresh prediction entry for this frame.  ``memory_counter``
            is initialised to zeros if absent.
        cfg: Dict with keys ``iou_thresh`` (float), ``ignore_thresh`` (int),
            ``rm_thresh`` (int), ``weighted`` (bool), ``voting`` (bool).

    Returns:
        Merged memory-bank entry (same schema, with ``memory_counter``).
    """
    iou_thresh = float(cfg.get('iou_thresh', 0.1))
    ignore_thresh = int(cfg.get('ignore_thresh', 2))
    rm_thresh = int(cfg.get('rm_thresh', 3))
    weighted = bool(cfg.get('weighted', False))
    voting = bool(cfg.get('voting', True))

    new = _ensure_counter(new)

    # First sight of this frame → adopt the new predictions as the memory.
    if prev is None or len(prev['gt_boxes']) == 0:
        return _ensure_counter(new)

    prev = _ensure_counter(prev)

    # Teacher found nothing this round → age every memory box.
    if len(new['gt_boxes']) == 0:
        merged = _index_entry(prev, slice(None))
        merged['memory_counter'] = prev['memory_counter'] + 1
        return _apply_voting(merged, ignore_thresh, rm_thresh, voting)

    iou = _iou_matrix(prev['gt_boxes'], new['gt_boxes'])  # [Np, Nn]

    # ── For each memory box: best matching new box ──────────────────────────
    a2b_iou = iou.max(axis=1)
    a2b_idx = iou.argmax(axis=1)
    matched = a2b_iou >= iou_thresh

    merged = _index_entry(prev, slice(None))         # start from memory copy
    merged_counter = prev['memory_counter'].copy()

    # Matched: keep higher-confidence box; copy its fields; reset counter.
    m_prev = np.nonzero(matched)[0]
    if len(m_prev) > 0:
        m_new = a2b_idx[m_prev]
        prev_sc = prev['scores'][m_prev]
        new_sc = new['scores'][m_new]
        if weighted:
            w = prev_sc / np.clip(prev_sc + new_sc, 1e-6, None)
            merged['gt_boxes'][m_prev, :7] = (
                w[:, None] * prev['gt_boxes'][m_prev, :7]
                + (1.0 - w[:, None]) * new['gt_boxes'][m_new, :7])
            lo = np.minimum(prev_sc, new_sc)
            hi = np.maximum(prev_sc, new_sc)
            merged['scores'][m_prev] = w * (hi - lo) + lo
            take = np.ones(len(m_prev), dtype=bool)  # always refresh aux fields
        else:
            take = new_sc > prev_sc                  # replace only when better
            sel_prev = m_prev[take]
            sel_new = m_new[take]
            merged['gt_boxes'][sel_prev] = new['gt_boxes'][sel_new]
            merged['scores'][sel_prev] = new['scores'][sel_new]
        # Copy the auxiliary per-box fields for the boxes we refreshed.
        sel_prev = m_prev[take]
        sel_new = m_new[take]
        for k in ('gt_labels', 'iou_scores', 'cls_scores', 'pt_counts'):
            if merged.get(k) is not None and new.get(k) is not None:
                merged[k][sel_prev] = new[k][sel_new]
        merged_counter[m_prev] = 0

    # Disappeared memory boxes: age them.
    merged_counter[~matched] += 1
    merged['memory_counter'] = merged_counter

    merged = _apply_voting(merged, ignore_thresh, rm_thresh, voting)

    # ── New boxes matching nothing in memory → append as fresh tracks ───────
    b2a_iou = iou.max(axis=0)
    fresh = np.nonzero(b2a_iou < iou_thresh)[0]
    if len(fresh) > 0:
        merged = _concat_entries(merged, _index_entry(new, fresh))

    return merged


def _apply_voting(entry: dict, ignore_thresh: int, rm_thresh: int,
                  voting: bool) -> dict:
    """Flag *ignore* (label → -1) at ``ignore_thresh`` and drop at ``rm_thresh``."""
    if not voting or len(entry['gt_boxes']) == 0:
        return entry
    counter = entry['memory_counter']
    if entry.get('gt_labels') is not None:
        ignore = counter >= ignore_thresh
        if ignore.any():
            entry['gt_labels'] = entry['gt_labels'].copy()
            entry['gt_labels'][ignore] = -1
    keep = counter < rm_thresh
    if not keep.all():
        entry = _index_entry(entry, keep)
    return entry


def memory_ensemble(prev_store: Optional[dict], new_store: dict, cfg: dict,
                    logger=None) -> dict:
    """Apply :func:`memory_ensemble_frame` across every frame of a store.

    Iterates over ``new_store`` (the refresh covers the full target split, so
    its keys are the superset) and merges each frame against its ``prev_store``
    entry.  Single-class only — ST3D's per-class loop is unnecessary here.

    Args:
        prev_store: Previous-refresh memory bank (frame → entry), or ``None``
            on the first refresh.
        new_store: Current-refresh filtered predictions (frame → entry).
        cfg: See :func:`memory_ensemble_frame`.
        logger: Optional MMLogger for a one-line summary.

    Returns:
        Merged memory bank (frame → entry, each carrying ``memory_counter``).
    """
    prev_store = prev_store or {}
    merged_store: dict = {}
    n_in = n_out = n_ignored = n_removed = 0

    for key, new_entry in new_store.items():
        prev_entry = prev_store.get(key)
        n_before = len(new_entry['gt_boxes'])
        if prev_entry is not None:
            n_before = max(n_before, len(prev_entry['gt_boxes']))
        merged = memory_ensemble_frame(prev_entry, new_entry, cfg)
        merged_store[key] = merged
        n_in += len(new_entry['gt_boxes'])
        labels = merged.get('gt_labels')
        if labels is not None:
            n_ignored += int((labels < 0).sum())
        n_out += len(merged['gt_boxes'])

    if logger is not None:
        logger.info(
            f'[MEV] memory-ensemble: {n_in} new boxes → {n_out} in memory '
            f'({n_ignored} flagged ignore) across {len(merged_store)} frames '
            f"(iou_thresh={cfg.get('iou_thresh', 0.1)}, "
            f"ignore_thresh={cfg.get('ignore_thresh', 2)}, "
            f"rm_thresh={cfg.get('rm_thresh', 3)})")
    return merged_store


def active_store(memory_store: dict) -> dict:
    """Return a student-facing store with *ignore*-flagged boxes removed.

    The memory bank keeps ignore-flagged boxes (label -1, counter ≥
    ``ignore_thresh``) so they remain tracked, but they must not be fed to the
    student as positives.  This strips them, yielding only active pseudo-labels
    (label ≥ 0).  The full memory bank (with counters and ignored tracks) is
    what gets persisted to the pkl for resume + next-refresh matching.
    """
    out: dict = {}
    for key, entry in memory_store.items():
        labels = entry.get('gt_labels')
        if labels is None or len(labels) == 0:
            out[key] = entry
            continue
        keep = labels >= 0
        if keep.all():
            out[key] = entry
        else:
            out[key] = _index_entry(entry, keep)
    return out
