from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

import numpy as np

from blindspot.data import CONDITION_LABELS, CONDITIONS, normalise_condition

__all__ = [
    "CONDITIONS",
    "CONDITION_LABELS",
    "base_prompts",
    "cluster_bootstrap_ci",
    "contingency",
    "item_key",
    "item_map",
    "keyed",
    "mcnemar",
    "need_map",
    "paired_delta_ci",
    "recall",
    "recall_by_condition",
    "scored",
]

# only used for a scenario with no base-condition sibling to anchor to.
_PROMPT_PREFIX = 80


def item_key(record: dict[str, Any], base_prompt: str | None = None) -> tuple[str, str, str]:
    prompt = record.get("prompt") or ""
    return (
        record.get("need", ""),
        record.get("task", "") or record.get("task_type", ""),
        base_prompt if base_prompt is not None else prompt[:_PROMPT_PREFIX],
    )


def base_prompts(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    """the base-condition prompt of each scenario, grouped by (need, task)."""
    families: dict[tuple[str, str], set[str]] = {}
    for record in records:
        if normalise_condition(record.get("condition", "")) == "base" and record.get("prompt"):
            key = (record.get("need", ""), record.get("task", "") or record.get("task_type", ""))
            families.setdefault(key, set()).add(record["prompt"])
    return {key: sorted(values, key=len, reverse=True) for key, values in families.items()}


def keyed(records: Iterable[dict[str, Any]]) -> list[tuple[tuple[str, str, str], dict[str, Any]]]:
    records = list(records)
    families = base_prompts(records)
    out = []
    for record in records:
        family = (record.get("need", ""), record.get("task", "") or record.get("task_type", ""))
        prompt = record.get("prompt") or ""
        anchor = next((b for b in families.get(family, ()) if prompt.startswith(b)), None)
        out.append((item_key(record, anchor), record))
    return out


def scored(records: Iterable[dict[str, Any]], condition: str | None = None) -> list[dict[str, Any]]:
    """the records the judge actually labelled, optionally in one condition."""
    wanted = normalise_condition(condition) if condition else None
    return [
        r
        for r in records
        if r.get("need_accounted_for") is not None
        and (wanted is None or normalise_condition(r.get("condition", "")) == wanted)
    ]


def item_map(records: Iterable[dict[str, Any]], condition: str = "base") -> dict[tuple, bool]:
    wanted = normalise_condition(condition)
    return {
        key: bool(record["need_accounted_for"])
        for key, record in keyed(list(records))
        if record.get("need_accounted_for") is not None
        and normalise_condition(record.get("condition", "")) == wanted
    }


def need_map(records: Iterable[dict[str, Any]], condition: str = "base") -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for record in scored(records, condition):
        buckets.setdefault(record.get("need", ""), []).append(
            float(bool(record["need_accounted_for"]))
        )
    return {need: float(np.mean(values)) for need, values in buckets.items()}


def recall(records: Iterable[dict[str, Any]], condition: str | None = None) -> float:
    """share of scored prompts where the need got surfaced."""
    values = [bool(r["need_accounted_for"]) for r in scored(records, condition)]
    return float(np.mean(values)) if values else float("nan")


def recall_by_condition(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    records = list(records)
    return {c: recall(records, c) for c in CONDITIONS}


def cluster_bootstrap_ci(
    records: Iterable[dict[str, Any]],
    condition: str | None = None,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    clusters: dict[str, list[float]] = {}
    for record in scored(records, condition):
        clusters.setdefault(record.get("need", ""), []).append(
            float(bool(record["need_accounted_for"]))
        )
    groups = list(clusters.values())
    if not groups:
        return float("nan"), float("nan")

    rng = random.Random(seed)
    n = len(groups)
    stats = []
    for _ in range(n_boot):
        drawn = [groups[rng.randrange(n)] for _ in range(n)]
        flat = [v for group in drawn for v in group]
        if flat:
            stats.append(sum(flat) / len(flat))
    if not stats:
        return float("nan"), float("nan")
    stats.sort()
    return (
        stats[int((alpha / 2) * len(stats))],
        stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))],
    )


def paired_delta_ci(
    before: dict[Any, float],
    after: dict[Any, float],
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    keys = sorted(set(before) & set(after), key=str)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    diffs = np.array([float(after[k]) - float(before[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[draws].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def contingency(a: dict[Any, bool], b: dict[Any, bool]) -> tuple[list[list[int]], int]:
    """2x2 counts over shared keys as [[both, a_only], [b_only, neither]], plus n."""
    keys = set(a) & set(b)
    both = sum(1 for k in keys if a[k] and b[k])
    a_only = sum(1 for k in keys if a[k] and not b[k])
    b_only = sum(1 for k in keys if not a[k] and b[k])
    neither = sum(1 for k in keys if not a[k] and not b[k])
    return [[both, a_only], [b_only, neither]], len(keys)


def mcnemar(a: dict[Any, bool], b: dict[Any, bool]) -> dict[str, float]:
    from scipy import stats as sps

    keys = set(a) & set(b)
    a_wins = sum(1 for k in keys if a[k] and not b[k])
    b_wins = sum(1 for k in keys if b[k] and not a[k])
    n = a_wins + b_wins
    p = 1.0 if n == 0 else float(sps.binomtest(a_wins, n, 0.5).pvalue)
    return {"a_only": a_wins, "b_only": b_wins, "n_discordant": n, "p": p}
