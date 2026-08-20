"""
Figure 10 — clarifying question rates under the base and guidance conditions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blindspot import config, data
from blindspot.figures import style
from blindspot.metrics import scored

NAME = "fig10_clarifying"
TITLE = "Clarifying question rates, base vs. guidance"
REQUIRES = config.BASELINES

PANELS = [("base", "Base"), ("guidance", "Guidance")]


def _rates(records, cue):
    """(resolved, unresolved, irrelevant) as fractions of all scored prompts."""
    rows = scored(records, cue)
    if not rows:
        return 0.0, 0.0, 0.0

    resolved = unresolved = irrelevant = 0
    for record in rows:
        if not record.get("asks_clarifying_question"):
            continue
        targets = record.get("clarifying_targets") or []
        relevant = {"group", "need"} & set(targets)
        if not relevant:
            irrelevant += 1
            continue
        # did supplying that cue actually get the need surfaced?
        flags = [
            record.get("resolved_by_group_cue") if "group" in relevant else None,
            record.get("resolved_by_need_cue") if "need" in relevant else None,
        ]
        if any(flag is True for flag in flags):
            resolved += 1
        else:
            unresolved += 1

    n = len(rows)
    return resolved / n, unresolved / n, irrelevant / n


def render(out_dir):
    style.use_style()
    models = [r.label for r in REQUIRES]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    x = np.arange(len(models))
    width = 0.36
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for i, (cue, label) in enumerate(PANELS):
        offset = (i - 0.5) * width
        resolved, unresolved, irrelevant = zip(
            *(_rates(data.load_run(run), cue) for run in REQUIRES), strict=True
        )
        resolved = np.array(resolved)
        unresolved = np.array(unresolved)
        irrelevant = np.array(irrelevant)

        alpha = 1.0 if cue == "base" else 0.62
        ax.bar(x + offset, resolved, width, color=style.C_DARK, alpha=alpha,
               edgecolor="none", zorder=3)
        ax.bar(x + offset, unresolved, width, bottom=resolved, color=style.C_MID,
               alpha=alpha, hatch="///", edgecolor="white", linewidth=0, zorder=3)
        ax.bar(x + offset, irrelevant, width, bottom=resolved + unresolved,
               color=style.C_NONE, alpha=alpha, edgecolor="none", zorder=3)

        tick_positions.extend(x + offset)
        tick_labels.extend([label] * len(models))

    # model names on the major ticks, condition on the minor ticks under each
    # bar — rotated labels floating above the bars collided with the legend.
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_xticks(tick_positions, minor=True)
    ax.set_xticklabels(tick_labels, minor=True, fontsize=9, color=style.INK_MID)
    ax.tick_params(axis="x", which="minor", length=0, pad=2)
    ax.tick_params(axis="x", which="major", length=0, pad=20)
    style.pct_yaxis(ax)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=style.C_DARK, label="relevant, resolved by the cue"),
            plt.Rectangle((0, 0), 1, 1, color=style.C_MID, hatch="///",
                          label="relevant, still unresolved"),
            plt.Rectangle((0, 0), 1, 1, color=style.C_NONE, label="off-target question"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3,
        frameon=False, fontsize=11,
    )

    return style.save(fig, NAME, out_dir)
