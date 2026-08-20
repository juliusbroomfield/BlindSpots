"""
Figure 12 — logistic regression coefficients for group membership.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from blindspot import config, data
from blindspot.figures import style
from blindspot.figures.taxonomy import condense_group
from blindspot.metrics import scored

NAME = "fig12_logistic"
TITLE = "Logistic regression coefficients, group membership controlling for domain"
REQUIRES = config.BASELINES

MIN_OBSERVATIONS = 20   # a group needs this many rows to get a stable coefficient

CATEGORY_COLORS = {
    "Mobility": style.C_DARK, "Vision": style.C_DARK, "Hearing": style.C_DARK,
    "Religion": style.C_WARM, "Allergies": style.C_MID,
    "Socioeconomic": style.C_GREEN, "Language": style.C_MID, "Cultural": style.C_GREEN,
}


def _design_matrix():
    """
    one row per (scored prompt, group) pair.

    a need can belong to several groups, so a prompt contributes one row per
    group it is annotated with. that inflates n relative to independent
    observations; the coefficients are point estimates and the CIs should be
    read as approximate.
    """
    rows = []
    for run in REQUIRES:
        for record in scored(data.load_run(run), "base"):
            hit = int(bool(record["need_accounted_for"]))
            domain = (record.get("domains") or ["Unknown"])[0]
            for raw in record.get("groups") or []:
                group = condense_group(raw)
                if group:
                    rows.append((hit, group, domain))
    return rows


def render(out_dir):
    import pandas as pd
    import statsmodels.formula.api as smf

    style.use_style()
    frame = pd.DataFrame(_design_matrix(), columns=["hit", "group", "domain"])

    keep = frame["group"].value_counts()
    frame = frame[frame["group"].isin(keep[keep >= MIN_OBSERVATIONS].index)]
    if frame["group"].nunique() < 2:
        raise ValueError("Not enough groups with observations to fit the model")

    model = smf.logit("hit ~ C(group) + C(domain)", data=frame).fit(disp=False)

    coefficients = []
    for term, value in model.params.items():
        if not term.startswith("C(group)"):
            continue
        label = term.split("T.", 1)[1].rstrip("]")
        low, high = model.conf_int().loc[term]
        coefficients.append((label, value, low, high))
    coefficients.sort(key=lambda row: row[1])

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(coefficients) + 2.6))
    for i, (label, value, low, high) in enumerate(coefficients):
        color = CATEGORY_COLORS.get(_category_for(label), style.C_DARK)
        ax.plot([low, high], [i, i], color=color, lw=2.0, alpha=0.45, zorder=2)
        ax.plot(value, i, "o", ms=8, color=color, zorder=3)

    ax.axvline(0, color=style.RULE, ls="--", lw=1.2, zorder=1)
    ax.set_yticks(range(len(coefficients)))
    ax.set_yticklabels([label for label, *_ in coefficients], fontsize=11)
    ax.set_xlabel("Log-odds relative to the reference group", fontsize=style.FS_TICK)
    ax.set_ylim(-0.8, len(coefficients) - 0.2)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(style.RULE)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        f"reference: {_reference_group(model, frame)} · "
        f"pseudo R² = {model.prsquared:.3f} · n = {len(frame):,}",
        fontsize=style.FS_TICK, color=style.INK_MID, pad=12,
    )

    return style.save(fig, NAME, out_dir)


def _category_for(label: str) -> str | None:
    from blindspot.figures.taxonomy import GROUP_CATEGORIES

    for category, members in GROUP_CATEGORIES.items():
        if label in members:
            return category
    return None


def _reference_group(model, frame) -> str:
    """the condition statsmodels dropped — alphabetically first by default."""
    present = {t.split("T.", 1)[1].rstrip("]") for t in model.params.index
               if t.startswith("C(group)")}
    return next((g for g in sorted(frame["group"].unique()) if g not in present), "?")
