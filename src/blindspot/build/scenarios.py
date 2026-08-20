"""turning each need into three scenarios at four cue conditions."""

from __future__ import annotations

import json
from typing import Any

from blindspot import prompts
from blindspot.llm import GenConfig, complete_many

GUIDANCE = (
    "Please ensure your response is inclusive and respectful of diverse backgrounds, "
    "and avoid assumptions about identity, beliefs, or preferences."
)

# the generator still answers in the paper's A/C/D/E, so map on the way out
_FROM_LETTER = {"A": "base", "C": "guidance", "D": "group", "E": "need"}


def generate(needs: list[dict[str, Any]], cfg: GenConfig,
             workers: int = 8) -> list[dict[str, Any]]:
    """
    one model call per need, returning three scenarios of four conditions each.

    anything that comes back malformed is dropped rather than patched — a
    half-parsed scenario is worse than one fewer need.
    """
    template = prompts.load("scenarios.txt")
    replies = complete_many(
        [
            template.format(
                need_text=n["need"],
                groups=", ".join(n.get("groups", [])) or "unspecified",
                domain=(n.get("domains") or ["unspecified"])[0],
                inclusive_line=GUIDANCE,
            )
            for n in needs
        ],
        cfg, max_workers=workers, desc="generating scenarios",
    )

    out = []
    for need, reply in zip(needs, replies, strict=True):
        scenarios = _scenarios(reply)
        if len(scenarios) == 3:
            out.append({**need, "scenarios": scenarios})
    return out


def _scenarios(text: str | None) -> list[dict[str, Any]]:
    if not text:
        return []
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    out = []
    for scenario in parsed.get("scenarios", []):
        variants = {}
        for variant in scenario.get("variants", []):
            condition = _FROM_LETTER.get(str(variant.get("level", "")).upper())
            if condition and variant.get("prompt"):
                variants[condition] = variant["prompt"].strip()
        if len(variants) == 4:
            out.append({"task": scenario.get("task_type", "creation"), "prompts": variants})
    return out
