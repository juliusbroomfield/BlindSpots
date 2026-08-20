"""
together's batch endpoint, over plain http.

three calls — upload, create, poll — so there's no sdk here. that isn't
minimalism for its own sake: the together package brings a whole terminal
formatting stack (rich, cyclopts, markdown-it) for a cli we never invoke, and
its response objects change between dicts and pydantic models across versions,
which this file used to carry a shim for.

two things to get right, both of which the sdk hid: files upload with
purpose="batch-api" rather than "batch", and batch lines carry only custom_id
and body — no method or url, unlike openai.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

from blindspot.llm.batch.core import POLL_SECONDS, BatchRequest, text_from_openai_body
from blindspot.llm.client import GenConfig

BASE_URL = "https://api.together.xyz/v1"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "EXPIRED"}


def _client(api_key: str) -> httpx.Client:
    # together retries on 429 and 5xx; httpx surfaces them, so let the caller's
    # retry budget handle it rather than sleeping blind here
    return httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=300.0,
        follow_redirects=True,
    )


def _upload(http: httpx.Client, path: Path) -> str:
    with open(path, "rb") as handle:
        response = http.post(
            "/files/upload",
            data={"purpose": "batch-api", "file_name": path.name, "file_type": "jsonl"},
            files={"file": (path.name, handle, "application/jsonl")},
        )
    response.raise_for_status()
    return response.json()["id"]


def _job(payload: dict[str, Any]) -> dict[str, Any]:
    """the job body, which some responses nest under "job" and some don't."""
    return payload.get("job") or payload


def run(reqs: Sequence[BatchRequest], cfg: GenConfig, job_id: str | None,
        on_submit: Callable[[str], None]) -> dict[str, str | None]:
    api_key = os.environ["TOGETHER_API_KEY"]
    model = cfg.model.split("/", 1)[-1] if cfg.model.startswith("together_ai/") else cfg.model

    with _client(api_key) as http:
        if job_id is None:
            lines = []
            for request in reqs:
                messages = ([{"role": "system", "content": request.system}] if request.system else [])
                messages.append({"role": "user", "content": request.prompt})
                lines.append({
                    "custom_id": request.custom_id,
                    "body": {
                        "model": model,
                        "messages": messages,
                        "max_tokens": cfg.max_tokens,
                        "temperature": cfg.temperature,
                    },
                })

            scratch = Path(".blindspot_together_batch.jsonl")
            scratch.write_text(
                "\n".join(json.dumps(line, ensure_ascii=False) for line in lines),
                encoding="utf-8",
            )
            try:
                file_id = _upload(http, scratch)
            finally:
                scratch.unlink(missing_ok=True)

            created = http.post(
                "/batches",
                json={"input_file_id": file_id, "endpoint": "/v1/chat/completions"},
            )
            created.raise_for_status()
            job_id = _job(created.json())["id"]
            print(f"  submitted together batch {job_id} ({len(reqs)} requests)")
            on_submit(job_id)

        import time

        while True:
            response = http.get(f"/batches/{job_id}")
            response.raise_for_status()
            batch = _job(response.json())
            status = str(batch.get("status", "")).upper().split(".")[-1]
            if status in TERMINAL:
                break
            time.sleep(POLL_SECONDS)

        if status != "COMPLETED":
            raise RuntimeError(f"together batch {job_id} ended as {status}")

        out: dict[str, str | None] = {}
        output_id = batch.get("output_file_id")
        if output_id:
            content = http.get(f"/files/{output_id}/content")
            content.raise_for_status()
            for line in content.text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                body = (obj.get("response") or {}).get("body") or obj.get("response") or {}
                out[obj["custom_id"]] = None if obj.get("error") else text_from_openai_body(body)
        return out
