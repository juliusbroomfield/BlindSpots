"""
Figure 7 — LoRA distillation of GPT-5 into Llama 3.1 8B.

left:  item overlap between the student and the teacher
right: recall across all four conditions for base Llama, the fine-tune, and
       the teacher as a ceiling on the same needs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blindspot import config, data
from blindspot.figures import style
from blindspot.metrics import CONDITION_LABELS, CONDITIONS, cluster_bootstrap_ci, item_map, recall

NAME = "fig07_sft"
TITLE = "SFT overlap with the teacher, and recall across cue conditions"
REQUIRES = ["llama-3.1-8b", "sft", "gpt-5"]

SERIES = [
    ("Llama 3.1 8B", "llama-3.1-8b", style.C_WARM),
    ("Llama + SFT", "llama-3.1-8b-sft", style.C_DARK),
    ("GPT-5 (teacher)", "gpt-5", style.C_GREEN),
]


def render(out_dir):
    style.use_style()
    records = {spec: data.load_run(config.parse(spec)) for _, spec, _ in SERIES}

    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5.6),
                                      gridspec_kw={"width_ratios": [1, 1.15]})

    # Left: base vs SFT, conditioned on whether the teacher succeeded
    base = item_map(records["llama-3.1-8b"], "base")
    student = item_map(records["llama-3.1-8b-sft"], "base")
    teacher = item_map(records["gpt-5"], "base")
    shared = set(base) & set(student) & set(teacher)

    inner = left.get_subplotspec().subgridspec(1, 2, wspace=0.35)
    left.remove()
    for panel, (teacher_hit, title) in enumerate([(False, "GPT-5 miss"), (True, "GPT-5 hit")]):
        ax = fig.add_subplot(inner[panel])
        keys = [k for k in shared if teacher[k] is teacher_hit]
        counts = [
            [sum(1 for k in keys if student[k] and not base[k]),
             sum(1 for k in keys if student[k] and base[k])],
            [sum(1 for k in keys if not student[k] and not base[k]),
             sum(1 for k in keys if not student[k] and base[k])],
        ]
        style.contingency(
            ax, counts,
            row_labels=["SFT hit", "SFT miss"],
            col_labels=["base miss", "base hit"],
            title=title, denom=len(shared), vmax=0.4,
            show_row_labels=(panel == 0),
        )

    # right: recall across cue conditions
    x = np.arange(len(CONDITIONS))
    for label, spec, color in SERIES:
        rows = records[spec]
        means = [recall(rows, cond) for cond in CONDITIONS]
        cis = [cluster_bootstrap_ci(rows, cond) for cond in CONDITIONS]
        right.plot(x, means, marker="o", color=color, label=label)
        right.fill_between(x, [c[0] for c in cis], [c[1] for c in cis],
                           alpha=0.12, color=color, linewidth=0)

    right.set_xticks(x)
    right.set_xticklabels([CONDITION_LABELS[cond] for cond in CONDITIONS])
    style.pct_yaxis(right)
    right.set_ylim(0.3, 1.02)
    right.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4)
    right.tick_params(length=0)
    right.legend(frameon=False, fontsize=style.FS_LEG, loc="lower right")

    return style.save(fig, NAME, out_dir)
