"""
plot_losses.py — Visualize mmdet3d training logs
Usage:
    python plot_losses.py run1.json [run2.json ...] [--smooth 0.6] [--x step] [--out losses.png]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def load_log(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def smooth_ema(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average smoothing (alpha=0 → no smoothing)."""
    if alpha <= 0:
        return values
    out, s = [], None
    for v in values:
        s = v if s is None else alpha * s + (1 - alpha) * v
        out.append(s)
    return out


# LOSS_KEYS = ["loss", "loss_cls", "loss_bbox", "loss_dir",
#              "loss_velo", "loss_iou", "loss_mask", "loss_depth"]  # extend as needed

LOSS_KEYS = ["loss", "loss_cls_source", "loss_bbox_source", "loss_dir_source",
             "loss_cls_target", "loss_bbox_target", "loss_dir_target", "loss_contrastive"]

EXTRA_KEYS = ["lr", "grad_norm"]

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot mmdet3d training losses from JSON logs.")
    p.add_argument("logs", nargs="+", help="Path(s) to training log JSON file(s).")
    p.add_argument("--x", default="step", choices=["step", "iter", "epoch"],
                   help="X-axis variable (default: step).")
    p.add_argument("--smooth", type=float, default=0.6,
                   help="EMA smoothing factor 0–1 (0 = off, 0.9 = heavy). Default: 0.6.")
    p.add_argument("--out", default="losses.png",
                   help="Output image path. Default: losses.png.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-extras", action="store_true",
                   help="Skip LR and grad_norm subplots.")
    return p.parse_args()


def collect(records: list[dict], x_key: str) -> dict[str, tuple[list, list]]:
    """Return {metric: (xs, ys)} for each numeric metric found in records."""
    buckets: dict[str, dict] = defaultdict(dict)
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    plot_keys = [k for k in LOSS_KEYS + EXTRA_KEYS if k in all_keys]

    for r in records:
        x_val = r.get(x_key) or r.get("step") or r.get("iter")
        if x_val is None:
            continue
        for k in plot_keys:
            if k in r:
                buckets[k][x_val] = r[k]

    result = {}
    for k, xmap in buckets.items():
        xs = sorted(xmap)
        ys = [xmap[x] for x in xs]
        result[k] = (xs, ys)
    return result


def plot_run(ax, xs, ys, label, color, smooth, linestyle="-"):
    ys_raw = np.array(ys, dtype=float)
    ys_smooth = smooth_ema(list(ys_raw), smooth)
    ax.plot(xs, ys_raw, color=color, alpha=0.2, linewidth=0.8, linestyle=linestyle)
    ax.plot(xs, ys_smooth, color=color, linewidth=1.8, label=label, linestyle=linestyle)


def main():
    args = parse_args()
    multi_run = len(args.logs) > 1

    # ── figure layout ─────────────────────────────────────────────────────────
    loss_keys_present = []
    all_data = []
    for path in args.logs:
        records = load_log(path)
        data = collect(records, args.x)
        all_data.append((Path(path).stem, data))
        for k in LOSS_KEYS:
            if k in data and k not in loss_keys_present:
                loss_keys_present.append(k)

    n_loss = len(loss_keys_present)
    n_extra = 0 if args.no_extras else sum(1 for k in EXTRA_KEYS
                                           if any(k in d for _, d in all_data))
    n_plots = n_loss + n_extra

    if n_plots == 0:
        print("No plottable keys found. Check your log files.")
        return

    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle("Training Metrics", fontsize=14, fontweight="bold", y=1.01)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    linestyles = ["-", "--", "-.", ":"]

    ax_idx = 0

    # ── loss subplots ─────────────────────────────────────────────────────────
    for loss_key in loss_keys_present:
        ax = axes_flat[ax_idx]
        ax_idx += 1
        for run_idx, (run_name, data) in enumerate(all_data):
            if loss_key not in data:
                continue
            xs, ys = data[loss_key]
            label = f"{run_name}" if multi_run else loss_key
            color = COLORS[run_idx % len(COLORS)] if multi_run else COLORS[ax_idx % len(COLORS)]
            plot_run(ax, xs, ys, label, color, args.smooth,
                     linestyle=linestyles[run_idx % len(linestyles)])

        ax.set_title(loss_key, fontsize=11)
        ax.set_xlabel(args.x.capitalize())
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # ── extra subplots (LR, grad_norm) ────────────────────────────────────────
    if not args.no_extras:
        for extra_key in EXTRA_KEYS:
            if not any(extra_key in d for _, d in all_data):
                continue
            ax = axes_flat[ax_idx]
            ax_idx += 1
            for run_idx, (run_name, data) in enumerate(all_data):
                if extra_key not in data:
                    continue
                xs, ys = data[extra_key]
                label = run_name if multi_run else extra_key
                color = COLORS[run_idx % len(COLORS)]
                ls = linestyles[run_idx % len(linestyles)]
                if extra_key == "lr":
                    # LR: no smoothing, just a step plot
                    ax.plot(xs, ys, color=color, linewidth=1.6,
                            label=label, linestyle=ls)
                else:
                    plot_run(ax, xs, ys, label, color, args.smooth, linestyle=ls)

            title = {"lr": "Learning Rate", "grad_norm": "Gradient Norm"}.get(extra_key, extra_key)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(args.x.capitalize())
            ax.set_ylabel(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if extra_key == "lr":
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda v, _: f"{v:.2e}"))

    # hide unused axes
    for i in range(ax_idx, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()