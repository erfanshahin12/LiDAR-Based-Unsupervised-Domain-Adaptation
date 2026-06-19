# Mean-Teacher nuScenes→KITTI Collapse — Root-Cause Analysis

**Date:** 2026-06-17
**Model:** CenterPoint (`configs/mean_teacher/mean_teacher_centerpoint_config.py`)
**Author:** debugging session (Claude)

---

## Executive summary

After four full runs, the collapse is **definitively not caused by pseudo-label store
quality, learning rate, or refresh interval.** A controlled experiment (Run 4) restored
ep3 pseudo-label precision to ep0 levels (0.60 → 0.80) and the run collapsed *harder*, not
softer. The collapse is a **teacher–student feedback loop with no target-domain ground-truth
anchor**, whose *onset* is clocked by the **EMA time constant** (τ ≈ 2.84 epochs at
`ema_momentum=0.9999`) and whose drift is transmitted into the student through an
**ungated contrastive/BEV-consistency channel** that all the pseudo-label QC bypasses.

The store-quality machinery (floor, consistency filter, soft-quality, keep-fraction) gates
**one** of the two teacher→student channels. The other channel (contrastive loss) and the
EMA weight coupling form a complete feedback loop on their own, which is why cleaning the
store cannot stop the collapse.

---

## 1. AP trajectories (3D AP40 moderate, strict = IoU 0.70)

| Epoch | Baseline | Run1 (no HIS, cont0.05) | Run2 (HIS, LR-ramp) | Run3 (LR-fix, adaptiveFloor) | Run4 (consistFilter+fixedFloor+soft) |
|------:|---------:|------------------------:|--------------------:|-----------------------------:|-------------------------------------:|
| 0     | 21.96    | —                       | —                   | —                            | —                                    |
| 1     | —        | 22.68                   | 21.33               | 38.47                        | 38.95                                |
| 2     | —        | **25.06**               | 20.57               | **46.09**                    | **47.45**                            |
| 3     | —        | 21.58                   | 15.45               | 41.21                        | 39.39                                |
| 4     | —        | 10.68                   | 7.45                | 16.85                        | **12.33**                            |
| 5     | —        | 2.56                    | 2.43                | 6.15                         | (stopped)                            |
| 6     | —        | 0.22                    | 0.34                | 4.51                         | —                                    |
| 7     | —        | 0.008                   | 0.02                | 2.77                         | —                                    |

**Invariant across every run:** peak at ep1–2, *moderate* drop ep2→3, *free-fall* ep3→4.
The onset epoch never moves, regardless of LR direction, refresh interval, or store quality.

---

## 2. The decisive controlled experiment (Run3 → Run4)

Run4 changed **only the pseudo-label store path** vs Run3. Everything else identical
(`ema_momentum=0.9999`, `lr=1e-4` cosine, `contrastive_weight=0.1`,
`target_loss_weight=0.5`, `use_bev_consistency=True`, `update_teacher_buffers=True`).

Store-path changes in Run4:
- `cls_percentile=60` (adaptive floor) → `cls_percentile=0` + **fixed `cls_thr=0.4`**
- `keep_frac=0.4` → `0.7`
- added `consistency_filter`
- added `pseudo_loss_cfg=dict(enable_soft_quality=True)`

### Pseudo-label quality (from `tools/compare_pseudo_label_runs.py`, BEV IoU vs KITTI GT)

| Run | Epoch | n_boxes | P@0.25 | R@0.25 | P@0.50 | mean_score |
|-----|------:|--------:|-------:|-------:|-------:|-----------:|
| Run3 (adaptive) | 0 | 11910 | 0.812 | 0.674 | 0.798 | 0.647 |
| Run3 (adaptive) | 3 | 18714 | **0.603** | 0.786 | 0.587 | 0.559 |
| Run3 (adaptive) | 6 | 15891 | 0.663 | 0.734 | 0.627 | 0.346 |
| Run4 (consistFilter) | 0 | 12162 | 0.836 | 0.708 | 0.822 | 0.654 |
| Run4 (consistFilter) | 3 | 13605 | **0.798** | 0.756 | 0.785 | 0.668 |

**Run4's ep3 store is excellent** — precision 0.798 (vs 0.603), recall *higher* (0.756),
no box explosion (13.6k vs 18.7k), mean score up (0.668 vs 0.559). The QC fix worked exactly
as designed.

### Yet the model collapsed *harder*

Run4 ep4 AP = **12.33** vs Run3 ep4 = **16.85**. Better pseudo-labels, deeper collapse.

**Conclusion: pseudo-label store quality is decoupled from the collapse.** This rules out
root-cause #2 from the original Stage-1 plan as the *collapse driver* (it is still worth
having for label hygiene, just not the cure).

### Why *harder*?
Tightening the store (higher floor, keep_frac 0.7, soft-quality down-weighting low-conf
boxes) **shrinks the one gated, supervised target channel** (`loss_bbox_target`,
`loss_heatmap_target`). The ungated contrastive channel runs at full strength regardless.
So the *relative* weight of the ungated drift-transmitting channel **increases** → deeper
collapse. Loss evidence (Run4 end-of-epoch): `loss_bbox_target` ≈ 0.22 (vs Run3 ≈ 0.26) —
weaker box supervision — while `loss_contrastive` is unchanged (~0.10).

---

## 3. The clock: EMA anchor decay

`ema_momentum = 0.9999` in **all four runs**. Time constant:

```
τ = 1 / (1 - 0.9999) = 10,000 iters ÷ 3,517 iters/epoch ≈ 2.84 epochs
```

The pretrained checkpoint is the **only** stabilizing anchor in the system (there is no
target-domain GT). It lives in the teacher's weights and decays out of the EMA with τ ≈ 2.84
epochs:
- Epochs 0–2 (< 1τ): teacher ≈ pretrained → good pseudo-labels & features → student rises to
  peak (ep2).
- Epoch ~3 (≈ 1τ): pretrained contribution decays to ~1/e ≈ 37%; accumulated student drift
  takes over the teacher → anchor gone → free-fall.

**Why this is the clock (evidence):** onset is invariant across runs where LR *ramped up*
(Run1/2) vs *decayed* (Run3/4) — a ~15× LR difference by ep4 — and across refresh intervals
(user's earlier 2 vs 3 test) and across store quality (Run3 vs Run4). The only parameter with
the right (~3-epoch) timescale that was held fixed across all of them is `ema_momentum`.

LR sets the **depth** of the collapse (the LR fix tripled the peak: 25 → 46), not the onset.

---

## 4. The transmission channel: ungated contrastive / BEV consistency

Two teacher→student channels exist; only one is gated.

**Channel A — box pseudo-labels (GATED).** Store → `loss_bbox_target` / `loss_heatmap_target`.
Floor, consistency filter, soft-quality, keep-fraction all act here. Run4 cleaned it; no effect.

**Channel B — contrastive / BEV consistency (UNGATED).**
`mmdet3d/models/detectors/mean_teacher_detector.py:882-931`:
- L884: iterates the **`*unfiltered*` `teacher_pred`**.
- L911: foreground = `all_scores_t > fg_threshold (0.5)`, explicitly *independent of the
  pseudo-label conf_threshold*.
- L921: pulls `neck_student[i]` toward the **live teacher** `bev_t` at every teacher fg box,
  every iteration.

No QC touches Channel B. As the teacher drifts (Channel C, below), B faithfully transmits the
drift into the student's neck representation — the very features the detection head regresses
from. Evidence: in Run4, `loss_heatmap_target` *rises at ep4 (0.785→0.801) even though the
ep3 store is clean* — the corruption is entering through the representation, not the targets.

**Channel C — EMA (student→teacher), `momentum=0.9999`.** Closes the loop.

Channels B + C form a complete feedback loop with no GT anchor on target geometry. Cleaning
Channel A cannot break it.

### Note on soft-quality (`enable_soft_quality=True`)
`mmdet3d/models/dense_heads/centerpoint_head.py:688`: the per-box quality weight scales
**`bbox_weights` only** → `loss_bbox`. It never touches `loss_heatmap` (L691). So soft-quality
down-weights a low-confidence box's *shape* regression but still trains the detection heatmap
to fire at that location at **full strength**. It cannot suppress false-positive *detections*,
only soften their geometry — a structural reason it "had no effect" on the collapse.

---

## 5. Evaluation of the three proposed changes

### (1) contrastive_weight 0.1 → 0.05
**Right channel, likely too timid.** This is the only knob that directly attacks Channel B
(the ungated drift transmitter). But if B is a primary driver, halving it may only soften the
collapse. **Recommend instead: run `contrastive_weight=0` as a diagnostic first.** If the
collapse softens materially → B is confirmed as a driver and we tune the weight. If the
collapse is unchanged → the driver is pure EMA weight drift (Channel C) and the contrastive
weight is a side-show. One clean experiment answers this.

### (2) decrease LR / different scheduler
**Depth lever, not an onset fix.** Proven: LR sets collapse depth, not timing. Lowering LR or
decaying faster reduces magnitude and can *lock in* the ep2 peak. Highest-value version: drop
LR hard right after the ep2 peak (e.g. step to ~1e-5 at ep2, or cosine with `T_max=3`) to
freeze the model near its best state. This is a useful **band-aid / safety net**, not a cure —
it stops learning before the loop diverges rather than fixing the loop.

### (3) EMA every 10–15 iters instead of every iter
**Correct intuition — it delays, doesn't prevent.** Updating the teacher every K iters with
the same momentum lengthens the effective time constant ≈ K×τ, pushing the onset later without
adding an anchor. **Its real value is as the decisive diagnostic for the clock:** if onset
moves later in proportion to K, Channel C is confirmed as the clock. The *preventive* version
of this lever is not "slow the EMA" but "**stop the teacher from drifting at all**" — freeze or
re-anchor (below).

---

## 6. Recommended decisive experiments (ranked)

**E1 — Confirm Channel B (cheapest, 4 epochs).** Run4 config + `contrastive_weight=0`.
Reveals whether the ungated contrastive loss is a collapse driver or a passenger.

**E2 — Confirm/break the clock (4–5 epochs).** Freeze the teacher at the pretrained checkpoint
(disable EMA update) — "offline self-training." If the collapse *vanishes* (AP plateaus instead
of falling), the entire collapse is teacher drift, full stop. Caveat: a frozen teacher also
stops improving the store; treat as a diagnostic, then relax to **periodic re-anchoring** or
`ema_momentum=0.99999` (τ ≈ 28 ep ≈ effectively frozen over a 10-ep run).

**E3 — The actual cure: keep the anchor permanently.** Add an L2-SP / weight-anchor
regularizer pulling student (and hence EMA teacher) toward the pretrained weights, OR
periodically reset the teacher toward the pretrained checkpoint (ST3D-style discrete updates
rather than continuous EMA). This restores the missing target-domain stabilizer without
freezing adaptation.

**Best single next run:** combine the safety net + the suspected fix —
`contrastive_weight=0` (E1) **and** a much higher `ema_momentum` / periodic teacher update (E2),
on the Run4 base (which already has the good store). If that holds past ep4, re-introduce the
contrastive loss at low weight to confirm its marginal value.

---

## 7. One-paragraph causal model

The pretrained checkpoint is the only thing telling the model what a real KITTI car's
geometry is. It decays out of the EMA teacher over ~2.84 epochs. Once it's gone (~ep3), the
teacher and student form a closed loop — the contrastive loss pulls student features toward the
(now drifting) teacher every iteration, the EMA pulls the teacher toward the student — with no
external signal anchoring target-domain box geometry. The loop diverges (free-fall ep3→4). The
pseudo-label store quality, the LR, and the refresh interval modulate the *height* and
*steepness* of the curve but cannot move its *onset* or *prevent* it, because none of them
restore the missing anchor or cut the ungated feature-coupling that carries the drift.
