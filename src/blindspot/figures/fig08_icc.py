"""
Figure 8 — intraclass correlation across the three scenarios per need.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blindspot import config, data
from blindspot.figures import style
from blindspot.figures.taxonomy import GROUP_CATEGORIES, condense_group
from blindspot.metrics import icc, scored

NAME = "fig08_icc"
TITLE = "Within-need consistency across scenarios, by group"
REQUIRES = config.BASELINES

MIN_NEEDS = 4  # a column needs this many needs before its ICC means anything


def render(out_dir):
    style.use_style()
    columns = [g for groups in GROUP_CATEGORIES.values() for g in groups]
    models = [r.label for r in REQUIRES]

    matrix = np.full((len(models), len(columns)), np.nan)
    for row, run in enumerate(REQUIRES):
        # need -> its per-scenario outcomes, kept per group column
        buckets: dict[str, dict[str, list[float]]] = {}
        for record in scored(data.load_run(run), "base"):
            hit = float(bool(record["need_accounted_for"]))
            need = record.get("need", "")
            for raw in record.get("groups") or []:
                column = condense_group(raw)
                if column:
                    buckets.setdefault(column, {}).setdefault(need, []).append(hit)

        for col, column in enumerate(columns):
            per_need = list(buckets.get(column, {}).values())
            if len(per_need) >= MIN_NEEDS:
                matrix[row, col] = icc(per_need)

    fig, ax = plt.subplots(figsize=(max(12, 0.55 * len(columns)), 0.7 * len(models) + 2.6))
    ax.imshow(matrix, cmap=style.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")

    for i in range(len(models)):
        for j in range(len(columns)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=9, color=style.INK if matrix[i, j] < 0.55 else "white")

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(ax.images[0], ax=ax, fraction=0.02, pad=0.02)
    bar.set_label("ICC", fontsize=style.FS_TICK)
    bar.outline.set_visible(False)

    return style.save(fig, NAME, out_dir)
