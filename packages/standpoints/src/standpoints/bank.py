"""the standpoints themselves, loaded from standpoints.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib.resources import files

import yaml

__all__ = ["META", "OMNI", "STANDPOINTS", "Meta", "Standpoint", "describe", "for_group"]


@dataclass(frozen=True)
class Standpoint:
    label: str
    description: str


@dataclass(frozen=True)
class Meta:
    """a broad standpoint that stands in for a set of narrower ones."""

    label: str
    description: str
    members: tuple[str, ...]


@cache
def _load() -> dict:
    text = files(__package__).joinpath("standpoints.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


_bank = _load()

STANDPOINTS: dict[str, Standpoint] = {
    label: Standpoint(label, description)
    for label, description in _bank["personas"].items()
}

META: dict[str, Meta] = {
    label: Meta(label, entry["description"], tuple(entry["members"]))
    for label, entry in _bank["meta"].items()
}

# one intersectional standpoint, for when you don't know who is affected
OMNI: str = _bank["omni"]


def describe(label: str) -> str:
    """the description a model is shown. never the label itself."""
    return STANDPOINTS[label].description


def for_group(name: str) -> str:
    """
    the standpoint description matching a group name, or "".

    exact match first, then containment either way, since group vocabularies
    rarely line up exactly.
    """
    if not name:
        return ""
    if name in STANDPOINTS:
        return STANDPOINTS[name].description

    needle = name.strip().lower()
    for label, standpoint in STANDPOINTS.items():
        hay = label.lower()
        if needle == hay or needle in hay or hay in needle:
            return standpoint.description
    return ""
