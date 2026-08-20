"""
mixture of standpoints — get a model to account for people the prompt never
mentions.

    from standpoints import respond
    respond("Design a lunch menu for 200 students.", model="gpt-5-mini")

works with any model litellm can reach. three stages, three sequential calls,
about 1.4x the cost of a single generation.

from "Whose Standpoint do LLMs Reflect?" (COLM 2026).
"""

__version__ = "1.0.0"

from standpoints.bank import META, OMNI, STANDPOINTS, Meta, Standpoint, describe, for_group
from standpoints.mos import Gap, MoS, Result, answer, respond

__all__ = [
    "META",
    "OMNI",
    "STANDPOINTS",
    "Gap",
    "MoS",
    "Meta",
    "Result",
    "Standpoint",
    "answer",
    "describe",
    "for_group",
    "respond",
]
