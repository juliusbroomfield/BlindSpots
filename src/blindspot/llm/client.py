from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402

litellm.telemetry = False
litellm.suppress_debug_info = True
litellm.drop_params = True  # quietly drop params a given provider doesn't take

# model families that take reasoning controls instead of temperature
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

DEFAULT_JUDGE_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 4096

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MAX_TOKENS",
    "GenConfig",
    "check_credentials",
    "complete",
    "complete_many",
    "provider_of",
]


@dataclass
class GenConfig:
    """decoding settings, applied per model family as appropriate."""

    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    reasoning_effort: str = "medium"
    verbosity: str = "medium"
    api_base: str | None = None
    retries: int = 5
    timeout: float = 600.0

    def is_reasoning(self) -> bool:
        return self.model.split("/")[-1].startswith(_REASONING_PREFIXES)

    def kwargs(self) -> dict[str, Any]:
        """the provider-appropriate arguments for this config."""
        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "num_retries": self.retries,
        }
        if self.api_base:
            kw["api_base"] = self.api_base
        if self.is_reasoning():
            kw["reasoning_effort"] = self.reasoning_effort
        else:
            kw["temperature"] = self.temperature
        return kw


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def complete(prompt: str, cfg: GenConfig, system: str | None = None) -> str:
    """one completion. litellm and the provider sdk handle the retrying."""
    response = litellm.completion(messages=_messages(prompt, system), **cfg.kwargs())
    return (response.choices[0].message.content or "").strip()


def complete_many(
    prompts: Sequence[str],
    cfg: GenConfig,
    system: str | None = None,
    max_workers: int = 8,
    desc: str = "generating",
) -> list[str | None]:
    """
    run many prompts concurrently, results in input order.
    """
    if not prompts:
        return []

    results: list[str | None] = [None] * len(prompts)
    done = 0

    def one(i: int) -> tuple[int, str | None]:
        try:
            return i, complete(prompts[i], cfg, system)
        except Exception as e:  # noqa: BLE001
            print(f"  [failed] prompt {i}: {type(e).__name__}: {e}")
            return i, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, text in pool.map(one, range(len(prompts))):
            results[i] = text
            done += 1
            if done % 200 == 0 or done == len(prompts):
                print(f"  {desc}: {done}/{len(prompts)}")
    return results


def provider_of(model: str) -> str:
    """the litellm provider id for a model string: openai, anthropic, ..."""
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
        return provider
    except Exception:
        return model.split("/")[0] if "/" in model else "openai"


def check_credentials(model: str) -> None:
    """fail early and clearly when the key for this provider isn't set."""
    needed = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "together_ai": "TOGETHER_API_KEY",
    }.get(provider_of(model))
    if needed and not os.environ.get(needed):
        raise RuntimeError(
            f"{needed} is not set, and model {model!r} needs it.\n"
            f"copy .env.example to .env, fill it in, then: set -a && source .env && set +a"
        )
