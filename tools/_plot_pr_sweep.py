"""Scatter plots of keep_frac × floor_value coloured by precision / recall."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# keep_frac, floor_value, precision, recall  (min_pt_count column ignored)
DATA = [
    (0.7, 0.40, 0.8096, 0.7129),
    (1.0, 0.40, 0.8089, 0.7148),
    (0.9, 0.40, 0.8089, 0.7148),
    (0.8, 0.40, 0.8089, 0.7142),
    (0.6, 0.40, 0.8113, 0.7060),
    (0.5, 0.40, 0.8182, 0.6850),
    (0.4, 0.40, 0.8311, 0.6490),
    (0.4, 0.35, 0.8041, 0.6593),
    (0.5, 0.35, 0.7870, 0.7002),
    (0.6, 0.35, 0.7763, 0.7242),
    (0.7, 0.35, 0.7736, 0.7334),
    (1.0, 0.35, 0.7721, 0.7361),
    (0.9, 0.35, 0.7721, 0.7361),
    (0.8, 0.35, 0.7721, 0.7353),
]

kf = np.array([d[0] for d in DATA])
fl = np.array([d[1] for d in DATA])
prec = np.array([d[2] for d in DATA])
rec = np.array([d[3] for d in DATA])

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_plot(values, label, cmap, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(kf, fl, c=values, cmap=cmap, s=320,
                    edgecolors='black', linewidths=0.8, zorder=3)
    # annotate each point with its value
    for x, y, v in zip(kf, fl, values):
        ax.text(x, y + 0.004, f'{v:.3f}', ha='center', va='bottom',
                fontsize=8, color='black', zorder=4)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(label, fontsize=11)
    ax.set_xlabel('keep_frac', fontsize=12)
    ax.set_ylabel('floor_value', fontsize=12)
    ax.set_title(f'{label} vs keep_frac × floor_value', fontsize=13)
    ax.set_yticks([0.35, 0.40])
    ax.set_xticks(sorted(set(kf)))
    ax.grid(True, alpha=0.25, zorder=0)
    ax.margins(x=0.08, y=0.25)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved -> {out}')


make_plot(prec, 'precision', 'viridis', 'pr_sweep_precision.png')
make_plot(rec, 'recall', 'plasma', 'pr_sweep_recall.png')
