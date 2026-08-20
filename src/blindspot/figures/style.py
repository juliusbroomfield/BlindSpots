"""
Consistent style formatting for figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from blindspot import config

_style = config.load("style.yaml")
_palette = _style["palette"]

C_DARK = _palette["dark"]     # navy, the baseline or primary series
C_MID = _palette["mid"]       # blue, secondary recovery
C_WARM = _palette["warm"]     # orange, the interesting or adverse segment
C_GREEN = _palette["green"]   # a third hue, where one is unavoidable
C_NONE = _palette["none"]     # neutral: "not recovered", not a series
INK = _palette["ink"]
INK_MID = _palette["ink_mid"]
RULE = _palette["rule"]

SEQUENTIAL = mcolors.LinearSegmentedColormap.from_list("blindspot_seq", ["#eef3fb", C_DARK])
# blue to muted gold: gentler than plasma, still monotone in lightness
DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "blindspot_div", ["#2f5aa8", "#7fa8d9", "#f2f0e6", "#e8c46a", "#b8860b"]
)

# a model reads the same colour in every figure
MODEL_COLORS = dict(_style["models"])

BAR_H = _style["sizes"]["bar_height"]
FS_ROW = _style["sizes"]["row"]
FS_INBAR = _style["sizes"]["in_bar"]
FS_TICK = _style["sizes"]["tick"]
FS_LEG = _style["sizes"]["legend"]

# stable colours per model, so a model reads the same across every figure.
MODEL_COLORS = {
    "GPT-5": "#1a3a7a",
    "GPT-5-mini": "#2f5aa8",
    "GPT-5-nano": "#48a0e0",
    "GPT-4.1": "#7fb2e0",
    "Llama 3.1 8B": "#e8842e",
    "Qwen 2.5 7B": "#b8860b",
}


def use_style() -> None:
    """apply the shared rcParams. call once at the top of a figure module."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 17,
        "axes.labelsize": 17,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "lines.linewidth": 2.8,
        "lines.markersize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,   # embed as TrueType so text stays selectable
        "ps.fonttype": 42,
    })


def save(fig: mpl.figure.Figure, stem: str, out_dir: str | Path) -> list[Path]:
    """write png/pdf/svg and close. returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in ((".png", {"dpi": 200}), (".pdf", {}), (".svg", {})):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def pct_axis(ax: plt.Axes, xmax: float = 1.02) -> None:
    """0-100% x axis with recessive chrome."""
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(left=False, length=0, labelsize=FS_TICK)
    ax.set_axisbelow(True)


def pct_yaxis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))


def stacked_row(ax, y, segments, min_label: float = 0.055) -> float:
    """
    one 100% stacked row. `segments` is [(width, color, label_or_None), ...].

    labels drop out when a segment is too narrow to hold them — the relief the
    palette's contrast warning requires.
    """
    x = 0.0
    for width, color, label in segments:
        ax.barh(y, width, height=BAR_H, left=x, color=color, edgecolor="none", zorder=3)
        if label and width >= min_label:
            ax.text(x + width / 2, y, label, ha="center", va="center",
                    fontsize=FS_INBAR, fontweight="bold", zorder=5,
                    color=INK_MID if color == C_NONE else "white")
        x += width
    return x


def row_label(ax, y, text: str, size: int = FS_ROW) -> None:
    ax.text(-0.015, y, text, ha="right", va="center", size=size, color=INK)


def contingency(
    ax,
    counts: Sequence[Sequence[int]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    row_title: str = "",
    col_title: str = "",
    title: str = "",
    denom: int | None = None,
    vmax: float | None = None,
    show_row_labels: bool = True,
) -> None:
    """
    2x2 contingency heatmap. counts[i][j] is (row_labels[i], col_labels[j]).

    sequential shading on count/denom — pass a shared denom and vmax to facet
    several panels onto one comparable scale. cell text is count plus percent.
    """
    m = np.asarray(counts, dtype=float)
    d = denom if denom is not None else m.sum()
    frac = m / d if d else m
    vmax = vmax if vmax is not None else float(frac.max()) * 1.15

    ax.imshow(frac, cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="equal", zorder=1)
    for i in range(2):
        for j in range(2):
            light = frac[i, j] < 0.55 * vmax
            ax.text(j, i, f"{int(m[i, j])}\n{frac[i, j]:.0%}", ha="center", va="center",
                    fontsize=FS_ROW, color=INK if light else "white", zorder=3, linespacing=1.5)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(col_labels, fontsize=FS_TICK, color=INK)
    ax.set_yticklabels(row_labels if show_row_labels else ["", ""], fontsize=FS_TICK, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    if col_title:
        ax.set_xlabel(col_title, fontsize=FS_ROW, color=INK, labelpad=8)
    if row_title and show_row_labels:
        ax.set_ylabel(row_title, fontsize=FS_ROW, color=INK, labelpad=8)
    if title:
        ax.set_title(title, fontsize=FS_TICK, color=INK_MID, pad=10)
    ax.axhline(0.5, color="white", lw=3, zorder=2)
    ax.axvline(0.5, color="white", lw=3, zorder=2)


def dumbbell(ax, y, before: float, after: float, color: str = C_DARK,
             ci: tuple[float, float] | None = None) -> None:
    """
    one dumbbell row: hollow marker for the baseline, filled for the condition.

    every row is paired against its own matched baseline, so a condition run
    over a subset of needs stays honest — its hollow marker just lands
    somewhere else.
    """
    ax.plot([before, after], [y, y], color=color, lw=2.2, zorder=2, alpha=0.55)
    ax.plot(before, y, "o", ms=11, mfc="white", mec=color, mew=2.2, zorder=3)
    ax.plot(after, y, "o", ms=11, color=color, zorder=4)
    if ci is not None:
        ax.plot(ci, [y, y], color=color, lw=1.2, alpha=0.8, zorder=1)
