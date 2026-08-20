"""
splitting failures into detection and operationalization.

the four prompt conditions are meant to tell two failure modes apart (paper section 2.3,
following Hou et al. 2024):

    detection           the model can produce the need once you name the group,
                        but not from the underspecified prompt. the group never
                        entered its prior about who is affected.
    operationalization  a genuine knowledge gap about that group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from blindspot.metrics.recall import item_map

__all__ = ["Decomposition", "decompose", "gap_sizes"]


@dataclass(frozen=True)
class Decomposition:
    """the four mutually exclusive outcomes in Figure 3 (left)."""

    baseline: float             # already surfaced from the base prompt
    detection: float            # recovered by naming the group
    operationalization: float   # recovered only by naming the need
    irreducible: float          # never surfaced, even with the need spelled out
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "baseline": self.baseline,
            "detection": self.detection,
            "operationalization": self.operationalization,
            "irreducible": self.irreducible,
            "n": self.n,
        }


def _paired(records: list[dict[str, Any]]) -> tuple[dict, dict, dict, set]:
    base = item_map(records, "base")
    group = item_map(records, "group")
    need = item_map(records, "need")
    return base, group, need, set(base) & set(group) & set(need)


def decompose(records: list[dict[str, Any]]) -> Decomposition:
    base, group, need, keys = _paired(records)
    if not keys:
        return Decomposition(*(float("nan"),) * 4, n=0)

    n = len(keys)
    baseline = sum(1 for k in keys if base[k])
    detection = sum(1 for k in keys if not base[k] and group[k])
    operationalization = sum(1 for k in keys if not base[k] and not group[k] and need[k])
    irreducible = n - baseline - detection - operationalization
    return Decomposition(
        baseline / n, detection / n, operationalization / n, irreducible / n, n
    )


def gap_sizes(records: list[dict[str, Any]]) -> dict[str, float]:
    base, group, need, keys = _paired(records)
    if not keys:
        return {"detection": float("nan"), "operationalization": float("nan"), "n": 0}

    n = len(keys)
    r_base = sum(base[k] for k in keys) / n
    r_group = sum(group[k] for k in keys) / n
    r_need = sum(need[k] for k in keys) / n
    return {
        "base": r_base,
        "group": r_group,
        "need": r_need,
        "detection": r_group - r_base,
        "operationalization": r_need - r_group,
        "n": n,
    }
