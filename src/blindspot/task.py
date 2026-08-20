"""
BlindSpot as an Inspect AI task.

inspect eval blindspot/task.py --model openai/gpt-5-mini --batch
"""

from __future__ import annotations

from typing import Any

from blindspot import personas
from blindspot.data import CONDITIONS, Item, load_items
from blindspot.prompts import JUDGE, JUDGE_SYSTEM


def _require_inspect() -> None:
    try:
        import inspect_ai  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "this adapter needs inspect: pip install inspect-ai\n"
            "the native pipeline (`blindspot run`) needs nothing extra."
        ) from e


def _sample(item: Item, method: str) -> Any:
    from inspect_ai.dataset import Sample

    prompt = item.prompt
    if method == "persona":
        description = item.persona or personas.persona_for_group(item.persona_group)
        if description:
            prompt = f"{description}\n\n{prompt}"
    elif method == "omni":
        prompt = f"{personas.OMNI_PERSONA}\n\n{prompt}"

    return Sample(
        id=item.id,
        input=prompt,
        target=item.need,
        metadata={
            "need_id": item.need_id,
            "scenario": item.scenario,
            "condition": item.condition,
            "task": item.task,
            "groups": item.groups,
            "subgroups": item.subgroups,
            "domains": item.domains,
            "group_label": item.group_label,
        },
    )


def need_recall(judge_model: str = "anthropic/claude-haiku-4-5"):
    from inspect_ai.scorer import Score, Target, accuracy, model_graded_qa, scorer, stderr

    grader = model_graded_qa(
        template=JUDGE.replace("{response}", "{answer}")
        .replace("{prompt}", "{question}")
        .replace("{need}", "{criterion}"),
        instructions=JUDGE_SYSTEM,
        model=judge_model,
    )

    @scorer(metrics=[accuracy(), stderr()])
    def score():
        async def run(state, target: Target) -> Score:
            return await grader(state, target)

        return run

    return score()


def blindspot(
    method: str = "none",
    conditions: str = ",".join(CONDITIONS),
    judge_model: str = "anthropic/claude-haiku-4-5",
    limit: int | None = None,
) -> Any:
    _require_inspect()

    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.solver import generate as generate_solver

    wanted = [c.strip().lower() for c in conditions.split(",") if c.strip()]
    items = load_items(conditions=wanted, limit=limit)

    return Task(
        dataset=MemoryDataset([_sample(i, method) for i in items], name="blindspot"),
        solver=generate_solver(),
        scorer=need_recall(judge_model),
    )


try:
    from inspect_ai import task as _task

    blindspot = _task(blindspot)
except ImportError:
    pass
