"""
how the benchmark gets built.

four stages, each writing a file the next one reads, all of them going through
`blindspot.llm` so any model litellm can reach will do:

    needs      pull candidate needs out of the reference documents
    filter     keep the ones that are concrete, omittable and testable
    label      tag each with groups, subgroups and a domain
    scenarios  turn each need into 3 scenarios x 4 conditions
    assemble   check for leakage, attach personas, write blindspot.jsonl

    python -m blindspot.build --model anthropic/claude-sonnet-4-5

regenerating gives you a new sample, not a copy of the released benchmark —
scenario generation is a model call. use the shipped data/blindspot.jsonl if you
want numbers comparable to the paper.
"""

from __future__ import annotations

from blindspot.build.assemble import assemble
from blindspot.build.label import label
from blindspot.build.needs import extract, filter_needs
from blindspot.build.scenarios import generate

__all__ = ["assemble", "extract", "filter_needs", "generate", "label"]
