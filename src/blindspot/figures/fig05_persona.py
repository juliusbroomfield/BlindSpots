"""
Figure 5 — recall under persona conditioning, against matched baselines.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from blindspot import config, data
from blindspot.figures import style
from blindspot.metrics import need_map, paired_delta_ci

NAME = "fig05_persona"
TITLE = "Persona, OmniPersona and MoS relative to matched baselines"

PANELS = [
    ("llama-3.1-8b", ["persona", "omni", "mos"]),
    ("gpt-5-mini", ["persona", "omni", "mos"]),
]

REQUIRES = [
    config.parse(spec)
    for model, methods in PANELS
    for spec in [model, *(f"{model}+{m}" for m in methods)]
]


def render(out_dir):
    style.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)

    for ax, (model, methods) in zip(axes, PANELS, strict=True):
        base_run = config.parse(model)
        baseline = need_map(data.load_run(base_run), "base")

        for i, method in enumerate(reversed(methods)):
            treated = need_map(data.load_run(config.parse(f"{model}+{method}")), "base")
            shared = sorted(set(baseline) & set(treated))
            if not shared:
                continue
            before = sum(baseline[n] for n in shared) / len(shared)
            after = sum(treated[n] for n in shared) / len(shared)
            delta, lo, hi = paired_delta_ci(
                {n: baseline[n] for n in shared}, {n: treated[n] for n in shared}
            )
            style.dumbbell(ax, i, before, after, style.C_DARK,
                           ci=(after - (delta - lo), after + (hi - delta)))
            style.row_label(ax, i, method)
            ax.text(after + 0.015, i + 0.24, f"{after:.0%}", fontsize=style.FS_INBAR,
                    color=style.INK_MID, va="center")

        model = base_run.label
        rows = methods
        ax.set_title(model, fontsize=style.FS_ROW, color=style.INK, pad=12)
        ax.set_xlim(0.4, 1.02)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_yticks([])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(style.RULE)
        ax.tick_params(left=False, length=0)
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_axisbelow(True)

    fig.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="none", ms=11, mfc="white",
                       mec=style.C_DARK, mew=2.2, label="matched baseline"),
            plt.Line2D([], [], marker="o", ls="none", ms=11, color=style.C_DARK,
                       label="mitigation"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=style.FS_LEG,
        bbox_to_anchor=(0.5, -0.06),
    )
    return style.save(fig, NAME, out_dir)
