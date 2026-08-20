"""
Code for reproducing all of the paper's figures. The registry is in config/figures.yaml.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from blindspot import config

_REGISTRY = config.load("figures.yaml")["figures"]


def load(name: str) -> ModuleType:
    """import one figure module by its short name, e.g. "fig03"."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown figure {name!r}. known: {', '.join(_REGISTRY)}")
    return importlib.import_module(f"blindspot.figures.{_REGISTRY[name]['module']}")


def names() -> list[str]:
    return list(_REGISTRY)


def requires(name: str) -> list[config.Run]:
    """the runs a figure needs, resolved from its yaml spec."""
    return [config.parse(spec) for spec in _REGISTRY[name]["requires"]]


def describe() -> list[tuple[str, str, list[str]]]:
    """(name, title, required runs) for every figure."""
    return [(n, e["title"], list(e["requires"])) for n, e in _REGISTRY.items()]


def render(name: str, out_dir: str | Path | None = None) -> list[Path]:
    entry = _REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"unknown figure {name!r}. known: {', '.join(_REGISTRY)}")

    absent = [r.name for r in requires(name) if r.find(required=False) is None]
    if absent:
        raise config.MissingResults(
            f"{name} needs runs that aren't on disk: {', '.join(sorted(set(absent)))}.\n"
            f"run `blindspot fetch`, or point BLINDSPOT_RESULTS at where they live."
        )
    return load(name).render(Path(out_dir) if out_dir else config.plots_dir())
