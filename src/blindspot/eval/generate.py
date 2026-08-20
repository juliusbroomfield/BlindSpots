from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from blindspot import mos, personas, prompts
from blindspot.config import REPO_ROOT
from blindspot.data import CONDITIONS, Item, load_items, write
from blindspot.eval import rag
from blindspot.llm.batch import BatchRequest
from blindspot.llm.batch import run as run_batch
from blindspot.llm.client import GenConfig, check_credentials

METHODS = ("none", "persona", "omni", "rag", "mos")


def build(items: Sequence[Item], method: str, contexts: list[str] | None = None
          ) -> list[BatchRequest]:
    """one request per item, with the method's transformation applied."""
    requests: list[BatchRequest] = []
    unmatched = 0

    for i, item in enumerate(items):
        system: str | None = None

        if method == "none":
            text = item.prompt

        elif method == "persona":
            # the benchmark carries a persona per need; fall back to matching
            # the item's group against the bank if it doesn't.
            description = item.persona or personas.persona_for_group(
                item.persona_group or (item.groups[0] if item.groups else "")
            )
            unmatched += not description
            text = f"{description}\n\n{item.prompt}" if description else item.prompt

        elif method == "omni":
            text = f"{personas.OMNI_PERSONA}\n\n{item.prompt}"

        elif method == "rag":
            context = (contexts or [""] * len(items))[i]
            text = f"{item.prompt}\n\n{context}" if context else item.prompt
            system = prompts.RAG_SYSTEM

        else:
            raise ValueError(f"Unknown method {method!r}. Pick from {METHODS}")

        requests.append(BatchRequest(custom_id=item.id, prompt=text, system=system))

    if unmatched:
        print(f"  [warn] {unmatched}/{len(items)} items had no matching persona; "
              f"those went out unmodified")
    return requests


def run(
    model: str,
    output: str | Path | None = None,
    method: str = "none",
    conditions: Sequence[str] = CONDITIONS,
    limit: int | None = None,
    benchmark: str | Path | None = None,
    max_tokens: int = 15000,
    reasoning_effort: str = "medium",
    verbosity: str = "medium",
    api_base: str | None = None,
    rag_top_k: int = 5,
    rag_sources: str | Path | None = None,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """run one method over the benchmark and write scoreable records."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Pick from {METHODS}")
    check_credentials(model)

    items = load_items(benchmark, conditions=conditions, limit=limit)
    print(f"{method} · {model} · {len(items)} prompts ({', '.join(conditions)})")

    cfg = GenConfig(
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        api_base=api_base,
    )
    stem = str(Path(output).with_suffix(""))

    if method == "mos":
        records = mos.run(items, cfg, checkpoint_stem=stem)
        write(output, records)
        print(f"\nWrote {len(records)} records → {output}")
        return records

    contexts = None
    if method == "rag":
        index = rag.build_index(rag_sources or REPO_ROOT / "sources")
        contexts = rag.contexts_for(index, [i.prompt for i in items], rag_top_k)

    replies = run_batch(build(items, method, contexts), cfg,
                        checkpoint=f"{stem}.ckpt.json", max_workers=max_workers)

    records = []
    for i, item in enumerate(items):
        record = item.to_dict()
        record["response"] = replies.get(item.id) or ""
        record["method"] = method
        record["model"] = model
        if method == "rag":
            record["context"] = (contexts or [""])[i]
        records.append(record)

    blank = sum(1 for r in records if not r["response"])
    write(output, records)
    print(f"\nWrote {len(records)} records ({blank} blank) → {output}")
    return records
