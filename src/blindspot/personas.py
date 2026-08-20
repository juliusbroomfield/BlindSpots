from __future__ import annotations

from standpoints import META, OMNI, STANDPOINTS, for_group

OMNI_PERSONA = OMNI
PERSONA_DESCRIPTIONS: dict[str, str] = {k: v.description for k, v in STANDPOINTS.items()}
PERSONA_NAMES: list[str] = list(PERSONA_DESCRIPTIONS)
PERSONA_BANK: list[tuple[str, str]] = list(PERSONA_DESCRIPTIONS.items())

META_PERSONA_DESCRIPTIONS: dict[str, str] = {k: v.description for k, v in META.items()}
META_PERSONA_MEMBERS: dict[str, list[str]] = {k: list(v.members) for k, v in META.items()}
META_PERSONA_NAMES: list[str] = list(META)
META_PERSONA_BANK: list[tuple[str, str, list[str]]] = [
    (label, META[label].description, list(META[label].members)) for label in META
]

persona_for_group = for_group

__all__ = [
    "META_PERSONA_BANK",
    "META_PERSONA_DESCRIPTIONS",
    "META_PERSONA_MEMBERS",
    "META_PERSONA_NAMES",
    "OMNI_PERSONA",
    "PERSONA_BANK",
    "PERSONA_DESCRIPTIONS",
    "PERSONA_NAMES",
    "persona_for_group",
]
