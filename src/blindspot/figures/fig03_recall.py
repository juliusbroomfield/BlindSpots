"""
Figure 3 — how often models surface the need, and why they miss it.

left:  the paired decomposition. every scenario lands in exactly one of four
       buckets, ordered by the weakest cue that recovers the need.
right: recall in each condition, with 95% CIs clustered by need.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blindspot import config, data
from blindspot.figures import style
from blindspot.metrics import CONDITION_LABELS, CONDITIONS, cluster_bootstrap_ci, decompose, recall

NAME = "fig03_recall"
TITLE = "Recall by condition, and the detection/operationalization split"
REQUIRES = config.BASELINES


def render(out_dir):
    style.use_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(17, 6.5),
                                      gridspec_kw={"width_ratios": [1.15, 1]})

    models = [r.label for r in REQUIRES]
    records = {r.label: data.load_run(r) for r in REQUIRES}

    # left: paired decomposition
    segments = [
        ("baseline", style.C_DARK, "Base"),
        ("detection", style.C_MID, "Detection"),
        ("operationalization", style.C_WARM, "Operationalization"),
        ("irreducible", style.C_NONE, "Irreducible"),
    ]
    for row, model in enumerate(reversed(models)):
        parts = decompose(records[model]).as_dict()
        style.stacked_row(
            left, row,
            [(parts[key], color, f"{parts[key]:.0%}") for key, color, _ in segments],
        )
        style.row_label(left, row, model)

    style.pct_axis(left)
    left.set_ylim(-0.7, len(models) - 0.3)
    left.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=lab) for _, c, lab in segments],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
        frameon=False, fontsize=style.FS_LEG,
    )

    # right: recall across cue conditions
    x = np.arange(len(CONDITIONS))
    for model in models:
        rows = records[model]
        means = [recall(rows, cond) for cond in CONDITIONS]
        cis = [cluster_bootstrap_ci(rows, cond) for cond in CONDITIONS]
        color = style.MODEL_COLORS.get(model, style.C_DARK)
        right.plot(x, means, marker="o", color=color, label=model)
        right.fill_between(x, [c[0] for c in cis], [c[1] for c in cis],
                           alpha=0.12, color=color, linewidth=0)

    right.set_xticks(x)
    right.set_xticklabels([CONDITION_LABELS[cond] for cond in CONDITIONS])
    style.pct_yaxis(right)
    right.set_ylim(0.40, 1.02)
    right.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4)
    right.tick_params(length=0)
    right.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                 frameon=False, fontsize=style.FS_LEG)

    return style.save(fig, NAME, out_dir)
