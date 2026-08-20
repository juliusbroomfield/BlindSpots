"""
the standpoint banks, appendix tables 2 and 3.

generated from the `standpoints` package rather than kept by hand, so the
appendix can't drift from what the model actually sees. that drift isn't
hypothetical: an earlier bank in this repo used prescriptive descriptions the
paper doesn't.
"""

from __future__ import annotations

from blindspot import personas

NAME = "standpoints"
TITLE = "Standpoint and meta-standpoint banks (appendix tables 2 and 3)"
REQUIRES: list = []  # generated from code, needs nothing on disk


def _escape(text: str) -> str:
    for char in ("&", "%", "$", "#", "_"):
        text = text.replace(char, f"\\{char}")
    return text


def to_latex() -> str:
    lines = [
        r"\begin{longtable}{p{0.30\textwidth} p{0.62\textwidth}}",
        r"\caption{Individual standpoints and their descriptions.}",
        r"\label{tab:standpoints} \\",
        r"\toprule Standpoint & Description \\ \midrule \endfirsthead",
        r"\toprule Standpoint & Description \\ \midrule \endhead",
    ]
    for label, description in personas.PERSONA_BANK:
        lines.append(f"{_escape(label)} & {_escape(description)} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}", ""]

    lines += [
        r"\begin{longtable}{p{0.24\textwidth} p{0.68\textwidth}}",
        r"\caption{Meta standpoints used in the MoS filtering step.}",
        r"\label{tab:meta-standpoints} \\",
        r"\toprule Meta standpoint & Description \\ \midrule \endfirsthead",
        r"\toprule Meta standpoint & Description \\ \midrule \endhead",
    ]
    for label, description, _ in personas.META_PERSONA_BANK:
        lines.append(f"{_escape(label)} & {_escape(description)} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


def to_text() -> str:
    lines = [f"{len(personas.PERSONA_BANK)} standpoints across "
             f"{len(personas.META_PERSONA_BANK)} meta standpoints", ""]
    for label, _, members in personas.META_PERSONA_BANK:
        lines.append(f"{label}  ({len(members)})")
        for member in members:
            lines.append(f"    {member:<38} {personas.PERSONA_DESCRIPTIONS[member]}")
        lines.append("")
    return "\n".join(lines)


def render() -> tuple[str, str]:
    return to_latex(), to_text()
