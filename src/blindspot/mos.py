"""
mixture of standpoints, over the benchmark.

the method itself lives in the `standpoints` package — `pip install standpoints`
gets you it on its own, no benchmark attached. this is just the adapter: it maps
benchmark items to prompts, hands the stages to blindspot's batch layer so the
1,830-prompt runs go through provider batch endpoints at half price, and maps
the results back into scoreable records.

one implementation, so what the paper ran and what people install are provably
the same code.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from standpoints import MoS, Result

from blindspot.data import Item
from blindspot.llm.batch import BatchRequest
from blindspot.llm.batch import run as run_batch
from blindspot.llm.client import GenConfig

__all__ = ["run"]


def _executor(cfg: GenConfig, checkpoint_stem: str | Path | None):
    """
    hand each stage to the batch layer.

    stages arrive in order, so the counter names their checkpoints — an
    interrupted run picks up the stage it was on rather than starting over.
    """
    stages = iter(["filter", "critique", "merge", "plain"])

    def execute(prompts: Sequence[str], max_tokens: int) -> list[str | None]:
        stage = next(stages, "extra")
        if not prompts:
            return []
        stage_cfg = GenConfig(**{**cfg.__dict__, "max_tokens": min(max_tokens, cfg.max_tokens)})
        checkpoint = f"{checkpoint_stem}.{stage}.ckpt.json" if checkpoint_stem else None
        requests = [BatchRequest(custom_id=str(i), prompt=p) for i, p in enumerate(prompts)]
        replies = run_batch(requests, stage_cfg, checkpoint=checkpoint)
        return [replies.get(str(i)) for i in range(len(prompts))]

    return execute


def _record(item: Item, result: Result) -> dict[str, Any]:
    record = item.to_dict()
    record.update(
        response=result.answer,
        final_response=result.answer,
        activated=result.activated,
        n_consulted=len(result.consulted),
        n_gaps=len(result.gaps),
        gaps=[{"standpoint": g.standpoint, "comment": g.comment} for g in result.gaps],
    )
    return record


def run(items: Sequence[Item], cfg: GenConfig,
        checkpoint_stem: str | Path | None = None) -> list[dict[str, Any]]:
    """run mos over benchmark items and return scoreable records."""
    mos = MoS(model=cfg.model, max_tokens=cfg.max_tokens,
              execute=_executor(cfg, checkpoint_stem))
    results = mos.run_many([item.prompt for item in items])

    merged = sum(1 for r in results if r.gaps)
    mean = sum(len(r.consulted) for r in results) / max(1, len(results))
    print(f"{merged}/{len(results)} answers merged a gap; "
          f"{mean:.1f} standpoints consulted per prompt")
    return [_record(item, result) for item, result in zip(items, results, strict=True)]
