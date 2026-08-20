"""each method against its own matched baseline."""

from __future__ import annotations

from blindspot import config, data
from blindspot.metrics import need_map, paired_delta_ci

NAME = "mitigations"
TITLE = "Method recall against matched baselines"

PAIRS = [
    ("llama-3.1-8b", ["persona", "omni", "mos", "rag"]),
    ("gpt-5-mini", ["persona", "omni", "mos"]),
]

REQUIRES = [config.parse(model) for model, _ in PAIRS] + [
    config.parse(f"{model}+{method}") for model, methods in PAIRS for method in methods
]


def rows() -> list[dict]:
    """
    marginal means would mislead here — a method run over a subset of the needs
    would look better or worse for reasons that have nothing to do with it. every
    delta is paired over the needs both sides cover.
    """
    out = []
    for model, methods in PAIRS:
        base_run = config.parse(model)
        baseline = need_map(data.load_run(base_run), "base")

        for method in methods:
            run = config.parse(f"{model}+{method}")
            if run.find(required=False) is None:
                continue
            treated = need_map(data.load_run(run), "base")
            shared = sorted(set(baseline) & set(treated))
            if not shared:
                continue

            before = {n: baseline[n] for n in shared}
            after = {n: treated[n] for n in shared}
            delta, lo, hi = paired_delta_ci(before, after)
            out.append({
                "model": base_run.label,
                "method": method,
                "baseline": sum(before.values()) / len(shared),
                "treated": sum(after.values()) / len(shared),
                "delta": delta,
                "ci": (lo, hi),
                "n_needs": len(shared),
            })
    return out


def to_latex(data_rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\begin{tabular}{llccc c}", r"\toprule",
        r"Model & Method & Baseline & Method & $\Delta$ & 95\% CI \\", r"\midrule",
    ]
    current = None
    for row in data_rows:
        first = row["model"] != current
        if first and current is not None:
            lines.append(r"\midrule")
        current = row["model"]
        lo, hi = row["ci"]
        lines.append(
            f"{row['model'] if first else ''} & {row['method']} & "
            f"{row['baseline']:.3f} & {row['treated']:.3f} & "
            f"{row['delta']:+.3f} & [{lo:+.3f}, {hi:+.3f}] \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Base-condition recall under each method, paired against the "
        r"same model's baseline over the needs both cover. CIs are percentile "
        r"bootstrap over the paired per-need deltas.}",
        r"\label{tab:mitigation-results}", r"\end{table}",
    ]
    return "\n".join(lines)


def to_text(data_rows: list[dict]) -> str:
    head = (f"{'Model':<16}{'Method':<12}{'Base':>9}{'Method':>9}"
            f"{'Delta':>10}{'95% CI':>20}{'needs':>8}")
    lines = [head, "-" * len(head)]
    for row in data_rows:
        lo, hi = row["ci"]
        lines.append(
            f"{row['model']:<16}{row['method']:<12}{row['baseline']:>9.1%}"
            f"{row['treated']:>9.1%}{row['delta']:>+10.1%}"
            f"{f'[{lo:+.1%}, {hi:+.1%}]':>20}{row['n_needs']:>8}"
        )
    return "\n".join(lines)


def render() -> tuple[str, str]:
    data_rows = rows()
    return to_latex(data_rows), to_text(data_rows)
