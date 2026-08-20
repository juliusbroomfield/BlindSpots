"""
three kinds of agreement: judge against human, scenario against scenario, model
against model.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """chance-corrected agreement between two binary raters."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b, strict=False) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def kappa_ci(a: Sequence[bool], b: Sequence[bool]) -> tuple[float, float]:
    """
    normal-approximation 95% CI for kappa (Fleiss et al.), clipped to [-1, 1].
    """
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for x, y in zip(a, b, strict=False) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return float("nan"), float("nan")
    k = (po - pe) / (1 - pe)
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    return max(-1.0, k - 1.96 * se), min(1.0, k + 1.96 * se)


def bootstrap_kappa_ci(
    a: Sequence[bool], b: Sequence[bool], n_boot: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    """percentile bootstrap CI for kappa, resampling items with replacement."""
    rng = random.Random(seed)
    n = len(a)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohens_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k == k:
            stats.append(k)
    if not stats:
        return float("nan"), float("nan")
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]


def icc(groups: list[list[float]]) -> float:
    """
    ICC(1) over ragged groups — here, the scenarios belonging to one need.

    high ICC means whether a need gets surfaced is a property of the need, not
    noise from which scenario it was instantiated in. returns the one-way
    random-effects consistency estimate, floored at 0.
    """
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")

    n_total = sum(len(g) for g in groups)
    k = n_total / len(groups)  # mean group size, for unbalanced designs
    grand = sum(v for g in groups for v in g) / n_total

    ms_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups) / (len(groups) - 1)
    within_df = n_total - len(groups)
    if within_df <= 0:
        return float("nan")
    ms_within = sum((v - np.mean(g)) ** 2 for g in groups for v in g) / within_df

    denom = ms_between + (k - 1) * ms_within
    if denom == 0:
        return float("nan")
    return max(0.0, float((ms_between - ms_within) / denom))


def pairwise_correlation(
    vectors: dict[str, dict[str, float]], method: str = "spearman"
) -> tuple[list[str], np.ndarray]:
    """
    correlation matrix over per-need recall vectors, on the needs all models share.
    """
    from scipy import stats as sps

    labels = list(vectors)
    shared = set.intersection(*(set(v) for v in vectors.values())) if vectors else set()
    needs = sorted(shared)
    if len(needs) < 3:
        return labels, np.full((len(labels), len(labels)), np.nan)

    data = np.array([[vectors[m][n] for n in needs] for m in labels])
    size = len(labels)
    out = np.eye(size)
    for i in range(size):
        for j in range(i + 1, size):
            if method == "spearman":
                r = sps.spearmanr(data[i], data[j]).statistic
            else:
                r = sps.pearsonr(data[i], data[j]).statistic
            out[i, j] = out[j, i] = r
    return labels, out
