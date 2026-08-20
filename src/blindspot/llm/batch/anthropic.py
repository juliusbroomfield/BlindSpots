"""Anthropic's Message Batches API. requests go inline, not as a file upload."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from blindspot.llm.batch.core import POLL_SECONDS, BatchRequest
from blindspot.llm.client import GenConfig


def run(reqs: Sequence[BatchRequest], cfg: GenConfig, job_id: str | None,
                   on_submit: Callable[[str], None]) -> dict[str, str | None]:
    import anthropic

    client = anthropic.Anthropic()
    model = cfg.model.split("/")[-1]

    if job_id is None:
        payload = []
        for r in reqs:
            params: dict[str, Any] = {
                "model": model,
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "messages": [{"role": "user", "content": r.prompt}],
            }
            if r.system:
                params["system"] = r.system
            payload.append({"custom_id": r.custom_id, "params": params})
        batch = client.messages.batches.create(requests=payload)
        job_id = batch.id
        print(f"  submitted Anthropic batch {job_id} ({len(reqs)} requests)")
        on_submit(job_id)

    while True:
        batch = client.messages.batches.retrieve(job_id)
        counts = batch.request_counts
        print(f"  {job_id}: {batch.processing_status} "
              f"ok={counts.succeeded} err={counts.errored} left={counts.processing}")
        if batch.processing_status == "ended":
            break
        time.sleep(POLL_SECONDS)

    out: dict[str, str | None] = {}
    for result in client.messages.batches.results(job_id):
        if result.result.type != "succeeded":
            out[result.custom_id] = None
            continue
        out[result.custom_id] = "".join(
            b.text for b in result.result.message.content if getattr(b, "type", "") == "text"
        ).strip()
    return out
