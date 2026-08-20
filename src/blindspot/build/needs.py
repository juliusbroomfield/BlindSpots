"""sourcing candidate needs from reference documents, and filtering them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blindspot import prompts
from blindspot.llm import GenConfig, complete_many

SOURCES = Path("reference_docs")


def _parse(text: str | None) -> list[dict[str, Any]]:
    if not text:
        return []
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [x for x in parsed if isinstance(x, dict)]


def extract(documents: list[Path], cfg: GenConfig, workers: int = 8) -> list[dict[str, Any]]:
    """pull candidate needs out of each reference document."""
    template = prompts.load("needs.txt")
    texts = [d.read_text(encoding="utf-8", errors="replace")[:40_000] for d in documents]
    replies = complete_many(
        [f"{template}\n\nSOURCE: {d.name}\n\n{t}" for d, t in zip(documents, texts, strict=True)],
        cfg, max_workers=workers, desc="extracting needs",
    )

    out = []
    for document, reply in zip(documents, replies, strict=True):
        for item in _parse(reply):
            if item.get("need"):
                out.append({**item, "source": document.name})
    return out


def filter_needs(candidates: list[dict[str, Any]], cfg: GenConfig,
                 workers: int = 8) -> tuple[list[dict], list[dict]]:
    """
    keep needs that are concrete, plausibly omitted, and testable in a prompt.

    returns (kept, rejected) so the rejections stay auditable — that's what
    data/needs.json ships them for.
    """
    template = prompts.load("filter.txt")
    replies = complete_many(
        [template.format(need=c["need"]) for c in candidates],
        cfg, max_workers=workers, desc="filtering needs",
    )

    kept, rejected = [], []
    for candidate, reply in zip(candidates, replies, strict=True):
        verdict = _verdict(reply)
        if verdict.get("approved"):
            kept.append(candidate)
        else:
            rejected.append({**candidate, "reason": verdict.get("reason", "")})
    return kept, rejected


def _verdict(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
