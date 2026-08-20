"""
checking the judge against human labels, and checking recall against precision.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from blindspot import config, prompts
from blindspot.data import write
from blindspot.eval.judge import parse_json
from blindspot.llm.client import GenConfig, check_credentials, complete_many

# judge vs. human


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters over the same items."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b, strict=False) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def kappa_ci(a: Sequence[bool], b: Sequence[bool]) -> tuple[float, float]:
    """
    normal approximation 95% CI for kappa (Fleiss et al.), clipped to [-1, 1].
    """
    import math

    n = len(a)
    po = sum(1 for x, y in zip(a, b, strict=False) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return float("nan"), float("nan")
    k = (po - pe) / (1 - pe)
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    return max(-1.0, k - 1.96 * se), min(1.0, k + 1.96 * se)


def bootstrap_ci(
    a: Sequence[bool], b: Sequence[bool], n_boot: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    """percentile bootstrap CI for kappa, resampling items with replacement."""
    rng = random.Random(seed)
    n = len(a)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohens_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k == k:  # skip degenerate resamples
            stats.append(k)
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[int(0.975 * len(stats))]
    return lo, hi


def judge_agreement(
    path: str | Path | None = None, n_boot: int = 10_000, seed: int = 42
) -> dict[str, Any]:
    """agreement, kappa, and the confusion matrix between the judge and the humans."""
    path = Path(path) if path else config.ANNOTATIONS
    records = json.loads(Path(path).read_text(encoding="utf-8"))

    paired = [
        (bool(r["llm_judge_verdict"]), bool(r["human_verdict"]))
        for r in records
        if r.get("human_verdict") is not None and r.get("llm_judge_verdict") is not None
    ]
    if not paired:
        raise ValueError(f"No annotated items in {path}")

    judge = [p[0] for p in paired]
    human = [p[1] for p in paired]
    n = len(paired)

    tp = sum(1 for j, h in paired if j and h)
    fp = sum(1 for j, h in paired if j and not h)
    fn = sum(1 for j, h in paired if not j and h)
    tn = sum(1 for j, h in paired if not j and not h)

    agreement = (tp + tn) / n
    kappa = cohens_kappa(judge, human)
    lo, hi = kappa_ci(judge, human)
    blo, bhi = bootstrap_ci(judge, human, n_boot, seed)

    print(f"Judge validation — {n} of {len(records)} sampled items annotated\n")
    print(f"  Raw agreement       {agreement:.1%}")
    print(f"  Cohen's kappa       {kappa:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"                            bootstrap [{blo:.2f}, {bhi:.2f}] "
          f"({n_boot:,} resamples)")
    print()
    print("                 human: yes   human: no")
    print(f"  judge: yes  {tp:>10}   {fp:>10}   <- false positives")
    print(f"  judge: no   {fn:>10}   {tn:>10}")
    print()
    print(f"  Precision {tp / (tp + fp):.1%}   Recall {tp / (tp + fn):.1%}"
          if (tp + fp) and (tp + fn) else "")

    return {
        "n": n,
        "agreement": agreement,
        "kappa": kappa,
        "kappa_ci": [lo, hi],
        "kappa_ci_bootstrap": [blo, bhi],
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# recall vs. precision


def false_needs(
    runs: Sequence[config.Run],
    model: str = "anthropic/claude-sonnet-4-5",
    n_sample: int = 20,
    seed: int = 42,
    output_path: str | Path = "false_needs_results.json",
    max_workers: int = 8,
) -> dict[str, Any]:
    """
    count irrelevant needs each condition surfaces, on prompts they all share.
    """
    check_credentials(model)
    from blindspot.data import load_run

    tables: dict[str, dict[tuple, dict[str, str]]] = {}
    for run in runs:
        records = load_run(run)
        tables[run.name] = {
            (r.get("need", ""), r.get("task", ""), r.get("condition") or ""): {
                "prompt": r.get("prompt", ""),
                "response": r.get("response", ""),
            }
            for r in records
            if r.get("response")
        }
        print(f"  {run.label:<28} {len(tables[run.name]):>6} scored responses")

    shared = set.intersection(*(set(t) for t in tables.values())) if tables else set()
    print(f"\nPrompts present in all {len(tables)} conditions: {len(shared)}")
    if not shared:
        raise ValueError("No prompts are shared across the requested conditions")

    rng = random.Random(seed)
    sampled = rng.sample(sorted(shared), min(n_sample, len(shared)))
    print(f"Scoring {len(sampled)} prompts × {len(tables)} conditions "
          f"= {len(sampled) * len(tables)} calls with {model}\n")

    cfg = GenConfig(model=model, max_tokens=512, temperature=0.0)
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for name, table in tables.items():
        replies = complete_many(
            [
                prompts.PRECISION.format(
                    prompt=table[k]["prompt"], response=table[k]["response"]
                )
                for k in sampled
            ],
            cfg,
            system=prompts.PRECISION_SYSTEM,
            max_workers=max_workers,
            desc=name,
        )
        for k, reply in zip(sampled, replies, strict=True):
            parsed = parse_json(reply or "")
            results[name].append({
                "prompt": table[k]["prompt"],
                "count": int(parsed.get("count", 0) or 0),
                "irrelevant_needs": parsed.get("irrelevant_needs", []),
                "reasoning": parsed.get("reasoning", ""),
            })

    write(output_path, [{"run": k, **row} for k, rows in results.items() for row in rows])

    print(f"\n{'Run':<28} {'Total':>7} {'Any':>10} {'Mean':>8}")
    print("-" * 56)
    for run in runs:
        entries = results[run.name]
        total = sum(e["count"] for e in entries)
        any_count = sum(1 for e in entries if e["count"] > 0)
        print(f"{run.label:<28} {total:>7} {any_count:>6}/{len(entries):<3} "
              f"{total / max(1, len(entries)):>8.2f}")
    return dict(results)
