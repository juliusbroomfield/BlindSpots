"""
Figure 4 — target need recall by group (columns) and domain (rows).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blindspot import config, data
from blindspot.figures import style
from blindspot.figures.taxonomy import GROUP_CATEGORIES, condense_group, domain_order
from blindspot.metrics import scored

NAME = "fig04_group_domain"
TITLE = "Recall by group and domain, averaged over models (base condition)"
REQUIRES = config.BASELINES

MIN_CELL = 3  # cells thinner than this are left blank rather than shown as noise


def render(out_dir):
    style.use_style()

    cells: dict[tuple[str, str], list[float]] = {}
    for run in REQUIRES:
        for record in scored(data.load_run(run), "base"):
            hit = float(bool(record["need_accounted_for"]))
            domains = record.get("domains") or ["Unknown"]
            for group in record.get("groups") or []:
                category = condense_group(group)
                if category:
                    cells.setdefault((domains[0], category), []).append(hit)

    groups = [g for cat in GROUP_CATEGORIES.values() for g in cat]
    groups = [g for g in groups if any((d, g) in cells for d, _ in cells)]
    domains = domain_order([d for d, _ in cells])

    matrix = np.full((len(domains), len(groups)), np.nan)
    for i, domain in enumerate(domains):
        for j, group in enumerate(groups):
            vals = cells.get((domain, group), [])
            if len(vals) >= MIN_CELL:
                matrix[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(max(12, 0.52 * len(groups)), 0.62 * len(domains) + 3))
    ax.imshow(matrix, cmap=style.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")

    for i in range(len(domains)):
        for j in range(len(groups)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=9, color=style.INK if matrix[i, j] < 0.55 else "white")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(ax.images[0], ax=ax, fraction=0.02, pad=0.02)
    bar.set_label("Target need recall", fontsize=style.FS_TICK)
    bar.outline.set_visible(False)

    return style.save(fig, NAME, out_dir)
