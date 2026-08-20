"""
Figure 6 — retrieval on Llama 3.1 8B.

left:  a 2x2 crossing RAG with the group cue
right: item-level overlap
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blindspot import config, data
from blindspot.figures import style
from blindspot.metrics import cluster_bootstrap_ci, contingency, item_map, mcnemar, scored

NAME = "fig06_rag"
TITLE = "RAG crossed with the group cue, and item overlap"
REQUIRES = [config.parse("llama-3.1-8b"), config.parse("llama-3.1-8b+rag")]


def render(out_dir):
    style.use_style()
    baseline = data.load_run(config.parse("llama-3.1-8b"))
    rag = data.load_run(config.parse("llama-3.1-8b+rag"))

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.4),
                                      gridspec_kw={"width_ratios": [1.3, 1]})

    # left: 2x2
    conditions = [
        ("Baseline", style.C_NONE, [(baseline, "base"), (baseline, "group")]),
        ("RAG", style.C_DARK, [(rag, "base"), (rag, "group")]),
    ]
    x = np.arange(2)
    width = 0.36
    for i, (label, color, cells) in enumerate(conditions):
        means, errs = [], [[], []]
        for records, cue in cells:
            subset = scored(records, cue)
            mean = float(np.mean([bool(r["need_accounted_for"]) for r in subset]))
            lo, hi = cluster_bootstrap_ci(records, cue)
            means.append(mean)
            errs[0].append(mean - lo)
            errs[1].append(hi - mean)
        offset = (i - 0.5) * width
        bars = left.bar(x + offset, means, width, color=color, label=label,
                        edgecolor="none", zorder=3)
        left.errorbar(x + offset, means, yerr=errs, fmt="none",
                      ecolor=style.INK_MID, elinewidth=1.4, capsize=4, zorder=4)
        for bar, mean in zip(bars, means, strict=False):
            left.text(bar.get_x() + bar.get_width() / 2, mean + 0.03, f"{mean:.0%}",
                      ha="center", fontsize=style.FS_INBAR, fontweight="bold",
                      color=style.INK)

    left.set_xticks(x)
    left.set_xticklabels(["Base", "Group"])
    style.pct_yaxis(left)
    left.set_ylim(0, 1.05)
    left.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    left.set_axisbelow(True)
    left.tick_params(length=0)
    left.legend(frameon=False, fontsize=style.FS_LEG, loc="upper left")

    # right: overlap
    rag_hits = item_map(rag, "base")
    group_hits = item_map(baseline, "group")
    counts, n = contingency(rag_hits, group_hits)
    # contingency() returns [[both, rag_only], [group_only, neither]]; the panel
    # is drawn as rows = RAG hit/miss, columns = Group hit/miss.
    style.contingency(
        right,
        [[counts[0][1], counts[0][0]], [counts[1][1], counts[1][0]]],
        row_labels=["RAG hit", "RAG miss"],
        col_labels=["Group miss", "Group hit"],
        denom=n,
    )
    test = mcnemar(rag_hits, group_hits)
    right.set_title(
        f"n = {n}   RAG-only {test['a_only']} vs group-only {test['b_only']}   "
        f"p = {test['p']:.1e}",
        fontsize=style.FS_TICK, color=style.INK_MID, pad=12,
    )

    return style.save(fig, NAME, out_dir)
