"""
Batch inference behind one function.

    results = run(requests, cfg, checkpoint="run.ckpt.json")   # {custom_id: text}

`cfg.model` picks the backend. Models whose provider has no batch endpoint — a
local vLLM server, say — fall back to concurrent live calls, and the call looks
the same either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from blindspot.llm.batch import anthropic, openai, together
from blindspot.llm.batch.core import (
    BatchRequest,
    clear_checkpoint,
    read_checkpoint,
    write_checkpoint,
)
from blindspot.llm.client import GenConfig, complete_many, provider_of

BACKENDS = {
    "openai": openai.run,
    "anthropic": anthropic.run,
    "together_ai": together.run,
}

__all__ = ["BatchRequest", "backend_for", "run"]


def backend_for(model: str) -> str | None:
    """which backend handles this model, or None if it has no batch endpoint."""
    provider = provider_of(model)
    return provider if provider in BACKENDS else None


def run(
    requests: Sequence[BatchRequest],
    cfg: GenConfig,
    checkpoint: str | Path | None = None,
    allow_fallback: bool = True,
    max_workers: int = 8,
) -> dict[str, str | None]:
    """
    Send `requests` through the batch endpoint for `cfg.model`.

    Returns {custom_id: text}, with None wherever the provider gave us nothing.
    """
    if not requests:
        return {}

    name = backend_for(cfg.model)
    if name is None:
        if not allow_fallback:
            raise RuntimeError(f"No batch endpoint for {cfg.model!r} and fallback is off.")
        print(f"  {cfg.model} has no batch endpoint — running {len(requests)} live calls")
        systems = {r.system for r in requests}
        if len(systems) > 1:
            # mixed system prompts can't share one call signature, so go one by one.
            texts = [complete_many([r.prompt], cfg, r.system, max_workers=1)[0]
                     for r in requests]
        else:
            texts = complete_many([r.prompt for r in requests], cfg,
                                  requests[0].system, max_workers=max_workers)
        return {r.custom_id: t for r, t in zip(requests, texts, strict=True)}

    state = read_checkpoint(checkpoint)
    job_id = state.get("job_id")
    if job_id:
        print(f"  resuming {name} batch {job_id} from {checkpoint}")

    def remember(new_id: str) -> None:
        write_checkpoint(checkpoint, {"job_id": new_id, "backend": name,
                                      "model": cfg.model, "n": len(requests)})

    results = BACKENDS[name](requests, cfg, job_id, remember)

    empty = sum(1 for r in requests if results.get(r.custom_id) is None)
    if empty:
        print(f"  {empty}/{len(requests)} requests came back empty")
    clear_checkpoint(checkpoint)
    return results
