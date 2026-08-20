"""the three stages."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files

from standpoints.bank import META, describe

__all__ = ["Gap", "MoS", "Result", "answer", "respond"]

NO_COMMENT = "[NO_COMMENT]"
NO_GAP = "[NO_GAP]"

# a stage takes prompts and returns one answer each, None where it failed.
# swap it out to route through a batch endpoint or your own client.
Executor = Callable[[Sequence[str], int], list[str | None]]


@cache
def _prompt(name: str) -> str:
    return files(__package__).joinpath(f"prompts/{name}.txt").read_text(encoding="utf-8").rstrip()


@dataclass(frozen=True)
class Gap:
    """something one standpoint says a typical answer would leave out."""

    standpoint: str
    comment: str


@dataclass
class Result:
    prompt: str
    answer: str
    activated: list[str] = field(default_factory=list)
    consulted: list[str] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)

    def __str__(self) -> str:
        return self.answer


def _default_executor(model: str, max_tokens: int, api_base: str | None) -> Executor:
    import litellm

    os.environ.setdefault("LITELLM_LOG", "ERROR")
    litellm.telemetry = False
    litellm.suppress_debug_info = True
    litellm.drop_params = True

    def execute(prompts: Sequence[str], cap: int) -> list[str | None]:
        def one(text: str) -> str | None:
            try:
                kwargs = {"api_base": api_base} if api_base else {}
                reply = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=min(cap, max_tokens),
                    num_retries=5,
                    **kwargs,
                )
                return (reply.choices[0].message.content or "").strip()
            except Exception:
                return None

        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=16) as pool:
            return list(pool.map(one, prompts))

    return execute


@dataclass
class MoS:
    """
    mixture of standpoints.

    three stages, each parallel inside itself, so wall clock is three
    sequential calls however many standpoints activate:

      filter     ask each broad standpoint whether this request would fail it
      critique   ask the ones that activated what a typical answer would miss
      merge      answer the request with those gaps folded in

    `execute` is the hook: it defaults to concurrent litellm calls, and you can
    replace it with anything that maps prompts to answers — a batch endpoint,
    a local server, your own client.
    """

    model: str = "gpt-5-mini"
    max_tokens: int = 4096
    api_base: str | None = None
    execute: Executor | None = None

    def __post_init__(self) -> None:
        if self.execute is None:
            self.execute = _default_executor(self.model, self.max_tokens, self.api_base)

    def run(self, prompt: str) -> Result:
        return self.run_many([prompt])[0]

    def run_many(self, prompts: Sequence[str]) -> list[Result]:
        """
        run every prompt through all three stages together.

        one round trip per stage for the whole batch rather than three per
        prompt, which is why this is the method to reach for.
        """
        prompts = list(prompts)
        if not prompts:
            return []

        results = [Result(prompt=p, answer="") for p in prompts]
        metas = list(META.values())

        # stage 1
        replies = self.execute(
            [
                _prompt("filter").format(persona_description=meta.description, prompt=p)
                for p in prompts
                for meta in metas
            ],
            256,
        )
        for i, result in enumerate(results):
            for j, meta in enumerate(metas):
                if _said_something(replies[i * len(metas) + j], NO_COMMENT):
                    result.activated.append(meta.label)
                    result.consulted.extend(meta.members)
            result.consulted = list(dict.fromkeys(result.consulted))

        # stage 2
        critique_index = [
            (i, label) for i, result in enumerate(results) for label in result.consulted
        ]
        replies = self.execute(
            [
                _prompt("critique").format(persona_description=describe(label), prompt=results[i].prompt)
                for i, label in critique_index
            ],
            256,
        )
        for (i, label), reply in zip(critique_index, replies, strict=True):
            if _said_something(reply, NO_GAP):
                results[i].gaps.append(Gap(label, (reply or "").strip()))

        # stage 3 — anything that activated nothing just gets answered plainly
        merged = [i for i, r in enumerate(results) if r.consulted]
        plain = [i for i, r in enumerate(results) if not r.consulted]

        replies = self.execute(
            [
                _prompt("merge").format(
                    prompt=results[i].prompt,
                    critiques="\n".join(f"[{g.standpoint}]: {g.comment}" for g in results[i].gaps),
                )
                for i in merged
            ],
            self.max_tokens,
        )
        for i, reply in zip(merged, replies, strict=True):
            results[i].answer = reply or ""

        replies = self.execute([results[i].prompt for i in plain], self.max_tokens)
        for i, reply in zip(plain, replies, strict=True):
            results[i].answer = reply or ""

        return results


def _said_something(reply: str | None, sentinel: str) -> bool:
    text = (reply or "").strip().upper()
    return bool(text) and not text.startswith(sentinel)


def respond(prompt: str, model: str = "gpt-5-mini", **kwargs) -> str:
    """run one prompt through mos and return the answer."""
    return MoS(model=model, **kwargs).run(prompt).answer


def answer(prompts: Sequence[str], model: str = "gpt-5-mini", **kwargs) -> list[str]:
    """run a batch. three calls total, not three per prompt."""
    return [r.answer for r in MoS(model=model, **kwargs).run_many(prompts)]
