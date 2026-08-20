"""openAI's Batch API, over the Responses endpoint."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Any

from blindspot.llm.batch.core import POLL_SECONDS, BatchRequest, text_from_openai_body
from blindspot.llm.client import GenConfig


def run(reqs: Sequence[BatchRequest], cfg: GenConfig, job_id: str | None,
                on_submit: Callable[[str], None]) -> dict[str, str | None]:
    from openai import OpenAI

    # the sdk's own retry honours Retry-After; nothing hand-rolled here
    client = OpenAI(timeout=cfg.timeout, max_retries=cfg.retries)
    model = cfg.model.split("/")[-1]

    if job_id is None:
        lines = []
        for r in reqs:
            body: dict[str, Any] = {
                "model": model,
                "input": r.prompt,
                "max_output_tokens": cfg.max_tokens,
            }
            if cfg.is_reasoning():
                body["reasoning"] = {"effort": cfg.reasoning_effort}
                body["text"] = {"verbosity": cfg.verbosity}
            if r.system:
                body["instructions"] = r.system
            lines.append(
                {"custom_id": r.custom_id, "method": "POST", "url": "/v1/responses", "body": body}
            )
        payload = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines).encode()
        upload = client.files.create(file=("batch_input.jsonl", payload, "application/jsonl"),
                                     purpose="batch")
        batch = client.batches.create(
            input_file_id=upload.id, endpoint="/v1/responses", completion_window="24h"
        )
        job_id = batch.id
        print(f"  submitted OpenAI batch {job_id} ({len(reqs)} requests)")
        on_submit(job_id)

    while True:
        batch = client.batches.retrieve(job_id)
        if batch.status in {"completed", "failed", "expired", "cancelled"}:
            break
        counts = batch.request_counts
        print(f"  {job_id}: {batch.status} "
              f"{getattr(counts, 'completed', 0)}/{getattr(counts, 'total', 0)}")
        time.sleep(POLL_SECONDS)

    if batch.status != "completed":
        raise RuntimeError(f"OpenAI batch {job_id} ended as {batch.status}")

    out: dict[str, str | None] = {}
    if batch.output_file_id:
        for line in client.files.content(batch.output_file_id).text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            out[obj["custom_id"]] = (
                None if obj.get("error") else text_from_openai_body(obj.get("response", {}).get("body", {}))
            )
    if batch.error_file_id:
        for line in client.files.content(batch.error_file_id).text.splitlines():
            if line.strip():
                out[json.loads(line)["custom_id"]] = None
    return out
