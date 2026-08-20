from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from blindspot import prompts
from blindspot.data import write
from blindspot.llm.batch import BatchRequest
from blindspot.llm.batch import run as run_batch
from blindspot.llm.client import DEFAULT_JUDGE_MODEL, GenConfig, check_credentials


def parse_json(text: str) -> dict[str, Any]:
    """parse the judge's reply, tolerating markdown fences and trailing prose."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def build(records: Sequence[dict[str, Any]]) -> list[BatchRequest]:
    """one judge call per response."""
    out = []
    for i, r in enumerate(records):
        out.append(
            BatchRequest(
                custom_id=f"judge-{i}",
                system=prompts.JUDGE_SYSTEM,
                prompt=prompts.JUDGE.format(
                    need=r.get("need", "Not specified"),
                    group_label=r.get("group_label") or "Not specified",
                    condition=r.get("condition", "Unknown"),
                    prompt=r.get("prompt", ""),
                    response=r.get("response", ""),
                ),
            )
        )
    return out


def apply_verdicts(
    records: list[dict[str, Any]], verdicts: dict[str, str | None]
) -> tuple[int, int]:
    """merge judge output into the records in place. returns (scored, failed)."""
    scored = failed = 0
    for i, record in enumerate(records):
        raw = verdicts.get(f"judge-{i}")
        parsed = parse_json(raw or "")
        if not parsed:
            record.update(
                need_accounted_for=None,
                asks_clarifying_question=None,
                clarifying_targets=[],
                evaluation_reasoning="",
                judge_error="no response" if raw is None else "unparseable JSON",
            )
            failed += 1
            continue

        targets = parsed.get("clarifying_targets") or []
        record.update(
            need_accounted_for=parsed.get("need_accounted_for"),
            asks_clarifying_question=parsed.get("asks_clarifying_question"),
            clarifying_targets=[
                t for t in targets if isinstance(t, str) and t in prompts.CLARIFYING_TARGETS
            ],
            evaluation_reasoning=parsed.get("reasoning", ""),
        )
        record.pop("judge_error", None)
        scored += 1
    return scored, failed


def apply_resolution_flags(records: list[dict[str, Any]]) -> None:
    siblings: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (record.get("need", ""), record.get("task", ""))
        siblings.setdefault(key, {})[record.get("condition", "")] = record

    for record in records:
        key = (record.get("need", ""), record.get("task", ""))
        family = siblings.get(key, {})
        targets = record.get("clarifying_targets") or []

        for target, cue, flag in (
            ("group", "group", "resolved_by_group_cue"),
            ("need", "need", "resolved_by_need_cue"),
        ):
            if target not in targets:
                record[flag] = None
                continue
            sibling = family.get(cue)
            record[flag] = (
                None if sibling is None else sibling.get("need_accounted_for") is True
            )


def run(
    source: str | Path,
    output: str | Path | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 400,
    response_field: str = "response",
    checkpoint: str | Path | None = None,
) -> list[dict[str, Any]]:
    """score every response in `source` and write the labelled copy."""
    check_credentials(model)
    # scored output sits next to its responses, with .scored before the suffix
    source = Path(source)
    output = Path(output) if output else source.with_name(source.stem + ".scored.jsonl")

    with open(source, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{source} should hold a list of response records")

    # multi-stage conditions (MoS) keep their text under a different key.
    for record in records:
        if response_field != "response":
            record["response"] = record.get(response_field, "")
        record.setdefault("response", "")

    print(f"Judging {len(records)} responses with {model}")
    cfg = GenConfig(model=model, max_tokens=max_tokens, temperature=0.0)
    ckpt = checkpoint or f"{Path(output).with_suffix('')}.judge.ckpt.json"
    verdicts = run_batch(build(records), cfg, checkpoint=ckpt)

    scored, failed = apply_verdicts(records, verdicts)
    apply_resolution_flags(records)
    write(output, records)

    accounted = sum(1 for r in records if r.get("need_accounted_for") is True)
    print(f"\nScored {scored}, failed {failed} → {output}")
    if scored:
        print(f"Target need recall: {accounted}/{scored} ({accounted / scored:.1%})")
    return records
