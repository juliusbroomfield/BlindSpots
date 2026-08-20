"""
Figure 9 — recall by task type under the base condition.

creation prompts ask the model to produce an artifact; advice prompts ask for
guidance on a decision.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blindspot import config, data
from blindspot.figures import style
from blindspot.metrics import cluster_bootstrap_ci, scored

NAME = "fig09_task"
TITLE = "Recall by task type (creation vs. advice), base condition"
REQUIRES = config.BASELINES

TASK_COLORS = {"creation": style.C_DARK, "advice": style.C_WARM}


def render(out_dir):
    style.use_style()
    models = [r.label for r in REQUIRES]

    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(models))
    width = 0.36

    for i, task in enumerate(("creation", "advice")):
        means, errs = [], [[], []]
        for run in REQUIRES:
            rows = [r for r in data.load_run(run) if r.get("task") == task]
            subset = scored(rows, "base")
            mean = float(np.mean([bool(r["need_accounted_for"]) for r in subset])) if subset else np.nan
            lo, hi = cluster_bootstrap_ci(rows, "base")
            means.append(mean)
            errs[0].append(mean - lo)
            errs[1].append(hi - mean)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, color=TASK_COLORS[task],
                      label=task.capitalize(), edgecolor="none", zorder=3)
        ax.errorbar(x + offset, means, yerr=errs, fmt="none", ecolor=style.INK_MID,
                    elinewidth=1.4, capsize=4, zorder=4)
        for bar, mean in zip(bars, means, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02, f"{mean:.0%}",
                    ha="center", fontsize=11, color=style.INK_MID)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    style.pct_yaxis(ax)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=style.FS_LEG, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, 1.12))

    return style.save(fig, NAME, out_dir)
