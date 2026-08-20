"""labelling each need with the groups, subgroups and domains it belongs to."""

from __future__ import annotations

import json
from typing import Any

from blindspot import config
from blindspot.llm import GenConfig, complete_many

_vocab = config.load("groups.yaml")
GROUPS: dict[str, list[str]] = _vocab["groups"]
DOMAINS: list[str] = _vocab["domains"]
ALL_GROUPS: list[str] = [g for members in GROUPS.values() for g in members]

TEMPLATE = """Label this accessibility or inclusivity need.

NEED: "{need}"

Pick every group whose members are directly affected, and the one domain it
belongs to. Use only the labels listed; do not invent new ones.

GROUPS:
{groups}

DOMAINS:
{domains}

Return strict JSON: {{"groups": [...], "subgroups": ["short plain-language description"], "domain": "..."}}"""


def label(needs: list[dict[str, Any]], cfg: GenConfig, workers: int = 8) -> list[dict[str, Any]]:
    """
    attach group, subgroup and domain labels.

    the model picks from a fixed vocabulary rather than free-texting, which is
    what makes labels comparable across runs and lets the group x domain figure
    have cells with anything in them.
    """
    catalogue = "\n".join(
        f"  {category}: {', '.join(members)}" for category, members in GROUPS.items()
    )
    replies = complete_many(
        [
            TEMPLATE.format(need=n["need"], groups=catalogue, domains="\n".join(f"  {d}" for d in DOMAINS))
            for n in needs
        ],
        cfg, max_workers=workers, desc="labelling needs",
    )

    out = []
    for need, reply in zip(needs, replies, strict=True):
        parsed = _parse(reply)
        groups = [g for g in parsed.get("groups", []) if g in ALL_GROUPS]
        domain = parsed.get("domain")
        if not groups or domain not in DOMAINS:
            continue
        out.append({
            **need,
            "groups": groups,
            "subgroups": parsed.get("subgroups", []),
            "domains": [domain],
        })
    return out


def _parse(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
