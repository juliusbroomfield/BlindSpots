"""
Figure 11 — pairwise correlation between per-need recall vectors.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blindspot import config, data
from blindspot.figures import style
from blindspot.metrics import need_map, pairwise_correlation

NAME = "fig11_model_correlation"
TITLE = "Do models miss the same needs?"
REQUIRES = config.BASELINES


def render(out_dir):
    style.use_style()

    vectors = {r.label: need_map(data.load_run(r), "base") for r in REQUIRES}
    labels, matrix = pairwise_correlation(vectors, method="spearman")

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    masked = np.array(matrix, dtype=float)
    np.fill_diagonal(masked, np.nan)
    finite = masked[np.isfinite(masked)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0

    ax.imshow(np.where(np.isnan(masked), vmin, masked), cmap=style.SEQUENTIAL,
              vmin=vmin, vmax=vmax, aspect="equal")

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color=style.RULE, fontsize=13)
                continue
            value = matrix[i][j]
            shade = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=12,
                    color=style.INK if shade < 0.55 else "white")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_shared = len(set.intersection(*(set(v) for v in vectors.values())))
    off_diagonal = [matrix[i][j] for i in range(len(labels))
                    for j in range(len(labels)) if i < j]
    ax.set_title(
        f"Spearman ρ over {n_shared} shared needs · "
        f"mean {np.mean(off_diagonal):.2f}, range {min(off_diagonal):.2f}–{max(off_diagonal):.2f}",
        fontsize=style.FS_TICK, color=style.INK_MID, pad=14,
    )

    bar = fig.colorbar(ax.images[0], ax=ax, fraction=0.035, pad=0.03)
    bar.outline.set_visible(False)
    return style.save(fig, NAME, out_dir)
