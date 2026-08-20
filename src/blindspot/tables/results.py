"""recall in each condition, per model, with the two gap sizes."""

from __future__ import annotations

from blindspot import config, data
from blindspot.metrics import (
    CONDITION_LABELS,
    CONDITIONS,
    cluster_bootstrap_ci,
    gap_sizes,
    recall,
)

NAME = "results"
TITLE = "Recall by condition, per model, with detection and operationalization gaps"
REQUIRES = config.BASELINES


def rows() -> list[dict]:
    out = []
    for run in REQUIRES:
        records = data.load_run(run)
        gaps = gap_sizes(records)
        row = {"model": run.label, "n": gaps["n"]}
        for cue in CONDITIONS:
            row[cue] = recall(records, cue)
            row[f"{cue}_ci"] = cluster_bootstrap_ci(records, cue)
        row["detection"] = gaps["detection"]
        row["operationalization"] = gaps["operationalization"]
        out.append(row)
    return out


def to_latex(data_rows: list[dict]) -> str:
    header = " & ".join(CONDITION_LABELS[c] for c in CONDITIONS)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\begin{tabular}{lcccc cc}", r"\toprule",
        rf"Model & {header} & Detection & Operationalization \\", r"\midrule",
    ]
    for row in data_rows:
        cells = " & ".join(f"{row[c]:.3f}" for c in CONDITIONS)
        lines.append(
            f"{row['model']} & {cells} & "
            f"{row['detection']:+.3f} & {row['operationalization']:+.3f} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Target need recall by prompt condition. Detection is the "
        r"base-to-group gap, operationalization the group-to-need gap. Both are "
        r"computed on scenarios scored in every condition, so they are paired.}",
        r"\label{tab:main-results}", r"\end{table}",
    ]
    return "\n".join(lines)


def to_text(data_rows: list[dict]) -> str:
    head = f"{'Model':<16}" + "".join(f"{CONDITION_LABELS[c]:>11}" for c in CONDITIONS)
    head += f"{'Detect':>10}{'Operat.':>10}{'n':>8}"
    lines = [head, "-" * len(head)]
    for row in data_rows:
        line = f"{row['model']:<16}" + "".join(f"{row[c]:>11.1%}" for c in CONDITIONS)
        line += f"{row['detection']:>+10.1%}{row['operationalization']:>+10.1%}{row['n']:>8,}"
        lines.append(line)
    return "\n".join(lines)


def render() -> tuple[str, str]:
    data_rows = rows()
    return to_latex(data_rows), to_text(data_rows)
