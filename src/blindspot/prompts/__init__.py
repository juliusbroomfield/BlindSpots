"""
every prompt we send, as a file you can read without reading python.

the judge rubric used to exist in four near-identical copies that had drifted
apart. two of them had quietly dropped the group field, so mos responses were
being scored with less context than the baselines they were compared against.
one rubric now, in judge.txt, and nothing else defines one.

they're plain text rather than python strings so you can diff a prompt change
on its own, and so the appendix can be generated from the same file the model
actually sees.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files

__all__ = [
    "CLARIFYING_TARGETS",
    "JUDGE",
    "JUDGE_SYSTEM",
    "PRECISION",
    "PRECISION_SYSTEM",
    "RAG_SYSTEM",
    "load",
]


@cache
def load(name: str) -> str:
    """read a prompt file. cached, since these are read once per run."""
    return files(__package__).joinpath(name).read_text(encoding="utf-8").rstrip("\n")


JUDGE = load("judge.txt")
JUDGE_SYSTEM = load("judge_system.txt")

PRECISION = load("precision.txt")
PRECISION_SYSTEM = load("precision_system.txt")

RAG_SYSTEM = load("rag_system.txt")

# what the judge is allowed to say a clarifying question was about
CLARIFYING_TARGETS = {"group", "need", "other"}

# header on the retrieved context block, so citations are greppable
RAG_CONTEXT_HEADER = "CONTEXT (use when relevant; cite tags like [file.json#c12]):"
