"""
batch inference, one call for every provider.

batch endpoints are half price and have much higher rate limits. that is the
only reason evaluating six models over 7,320 prompts was affordable, so it's
the default path wherever a provider offers one.

liteLLM handles the ordinary chat calls but its Batches API only reaches
openAI, so Anthropic's Message Batches and Together's batch endpoint are
wrapped here. each provider gets its own module because each has its own
quirks — Together's JSONL lines carry no method or url, Anthropic takes
requests inline rather than as an uploaded file, and OpenAI wants the
responses shape.

point `checkpoint=` at a file and an interrupted run picks up the same job
instead of paying for it twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POLL_SECONDS = 30


@dataclass
class BatchRequest:
    custom_id: str
    prompt: str
    system: str | None = None


def text_from_openai_body(body: dict[str, Any]) -> str | None:
    """pull the assistant text out of a Responses (or Chat Completions) body."""
    if body.get("output_text"):
        return str(body["output_text"]).strip()
    chunks: list[str] = []
    for item in body.get("output") or []:
        for part in item.get("content") or []:
            if part.get("type") in ("output_text", "text") and part.get("text"):
                chunks.append(part["text"])
    if chunks:
        return "\n".join(chunks).strip()
    for choice in body.get("choices") or []:
        content = (choice.get("message") or {}).get("content")
        if content:
            chunks.append(content)
    return "\n".join(chunks).strip() if chunks else None


def read_checkpoint(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_checkpoint(path: str | Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def clear_checkpoint(path: str | Path | None) -> None:
    if path and Path(path).exists():
        Path(path).unlink()
