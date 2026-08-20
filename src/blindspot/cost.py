from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from blindspot import personas, prompts
from blindspot.data import load_items
from blindspot.llm.batch import backend_for

# batch endpoints are half price at every provider the project uses.
BATCH_DISCOUNT = 0.5


def count_tokens(model: str, text: str) -> int:
    import litellm

    try:
        return max(1, litellm.token_counter(model=model, text=text))
    except Exception:
        return max(1, len(text) // 4)


def price_per_million(model: str) -> tuple[float, float]:
    """(input, output) USD per million tokens, or (0, 0) if unknown."""
    import litellm

    try:
        info = litellm.get_model_info(model)
        return (
            (info.get("input_cost_per_token") or 0.0) * 1e6,
            (info.get("output_cost_per_token") or 0.0) * 1e6,
        )
    except Exception:
        return 0.0, 0.0


def _report(model: str, stages: list[tuple[str, int, int, int]], batch: bool) -> dict[str, Any]:
    """stages: (label, n_calls, input_tokens, output_tokens)."""
    in_price, out_price = price_per_million(model)
    discount = BATCH_DISCOUNT if batch else 1.0

    def cost(tin: int, tout: int) -> float:
        return (tin / 1e6 * in_price + tout / 1e6 * out_price) * discount

    print(f"\n{model}  —  ${in_price:.2f}/M in, ${out_price:.2f}/M out"
          + ("  (batch, 50% off)" if batch else "  (live pricing)"))
    if in_price == 0 and out_price == 0:
        print("  [warn] LiteLLM has no price for this model — token counts only")
    print(f"\n  {'Stage':<26}{'Calls':>9}{'Input tok':>13}{'Output tok':>13}{'USD':>10}")
    print("  " + "-" * 71)

    total_in = total_out = total_calls = 0
    for label, calls, tin, tout in stages:
        print(f"  {label:<26}{calls:>9,}{tin:>13,}{tout:>13,}{cost(tin, tout):>10.2f}")
        total_in += tin
        total_out += tout
        total_calls += calls

    print("  " + "-" * 71)
    print(f"  {'Total':<26}{total_calls:>9,}{total_in:>13,}{total_out:>13,}"
          f"{cost(total_in, total_out):>10.2f}\n")
    return {
        "model": model,
        "calls": total_calls,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "usd": cost(total_in, total_out),
        "batch": batch,
    }


def estimate(
    model: str,
    method: str = "none",
    conditions: Sequence[str] = ("base", "guidance", "group", "need"),
    limit: int | None = None,
    expected_output_tokens: int = 900,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """
    estimate a generation run, and optionally the judging pass that follows it.
    """
    items = load_items(conditions=conditions, limit=limit)
    batch = backend_for(model) is not None
    stages: list[tuple[str, int, int, int]] = []

    if method == "mos":
        # stage sizes follow appendix D.5: 8 meta probes each, ~17 personas
        n_meta = len(personas.META_PERSONA_BANK)
        active = 17
        probe_in = sum(
            count_tokens(model, prompts.MOS_FILTER.format(
                persona_description=personas.META_PERSONA_BANK[0][1], prompt=it.prompt))
            for it in items
        ) * n_meta
        crit_in = sum(
            count_tokens(model, prompts.MOS_CRITIQUE.format(
                persona_description="You are blind.", prompt=it.prompt))
            for it in items
        ) * active
        merge_in = sum(
            count_tokens(model, prompts.MOS_MERGE.format(
                prompt=it.prompt, critiques="x " * 20 * active))
            for it in items
        )
        stages = [
            ("Meta-persona filter", len(items) * n_meta, probe_in, len(items) * n_meta * 12),
            ("Persona critiques", len(items) * active, crit_in, len(items) * active * 25),
            ("Merge", len(items), merge_in, len(items) * expected_output_tokens),
        ]
    else:
        prefix = {
            "persona": "You are blind.\n\n",
            "omni": personas.OMNI_PERSONA + "\n\n",
            "rag": "",
        }.get(method, "")
        gen_in = sum(count_tokens(model, prefix + it.prompt) for it in items)
        if method == "rag":
            gen_in += len(items) * 1600 
        stages = [(f"Generation ({method})", len(items), gen_in,
                   len(items) * expected_output_tokens)]

    result = _report(model, stages, batch)

    if judge_model:
        judge_in = sum(
            count_tokens(judge_model, prompts.JUDGE.format(
                need=it.need, group_label=it.group_label, condition=it.condition,
                prompt=it.prompt, response="x " * expected_output_tokens))
            for it in items
        )
        judge = _report(
            judge_model,
            [("Judging", len(items), judge_in, len(items) * 150)],
            backend_for(judge_model) is not None,
        )
        print(f"  Generation + judging: ${result['usd'] + judge['usd']:.2f}\n")
        result["judge"] = judge

    return result
