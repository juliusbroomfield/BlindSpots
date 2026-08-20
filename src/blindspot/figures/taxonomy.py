from __future__ import annotations

from blindspot import config

__all__ = [
    "CATEGORY_ORDER",
    "CONDENSED",
    "DOMAIN_ORDER",
    "GROUP_CATEGORIES",
    "category_of",
    "condense_group",
    "domain_order",
]

_taxonomy = config.load("taxonomy.yaml")

# raw label -> (column, category), or None where the group is dropped
CONDENSED: dict[str, tuple[str, str] | None] = {
    raw: (entry["column"], entry["category"]) if entry else None
    for raw, entry in _taxonomy["columns"].items()
}

CATEGORY_ORDER: list[str] = list(_taxonomy["category_order"])
DOMAIN_ORDER: list[str] = list(_taxonomy["domain_order"])

# category -> its columns, in the paper's left-to-right order
GROUP_CATEGORIES: dict[str, list[str]] = {}
for _raw, _mapped in CONDENSED.items():
    if _mapped is None:
        continue
    _column, _category = _mapped
    _bucket = GROUP_CATEGORIES.setdefault(_category, [])
    if _column not in _bucket:
        _bucket.append(_column)
GROUP_CATEGORIES = {c: GROUP_CATEGORIES[c] for c in CATEGORY_ORDER if c in GROUP_CATEGORIES}


def condense_group(raw: str) -> str | None:
    """collapse a benchmark group label to its figure 4 column, or None to drop it."""
    mapped = CONDENSED.get(raw)
    return mapped[0] if mapped else None


def category_of(raw: str) -> str | None:
    mapped = CONDENSED.get(raw)
    return mapped[1] if mapped else None


def domain_order(seen: list[str]) -> list[str]:
    """domains in the paper's order, with anything unrecognised appended."""
    present = set(seen)
    ordered = [d for d in DOMAIN_ORDER if d in present]
    return ordered + sorted(present - set(ordered))
