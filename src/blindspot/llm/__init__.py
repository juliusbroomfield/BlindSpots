"""model access: one LiteLLM-backed client, plus batch endpoints per provider."""

from blindspot.llm.client import (
    DEFAULT_JUDGE_MODEL,
    GenConfig,
    check_credentials,
    complete,
    complete_many,
    provider_of,
)

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "GenConfig",
    "check_credentials",
    "complete",
    "complete_many",
    "provider_of",
]
