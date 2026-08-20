"""
the paper's tables, generated rather than transcribed.

each module exposes NAME, TITLE, REQUIRES and render(), returning (latex, text).
the latex goes to a .tex file and the text twin gets printed, so the number you
eyeball in a terminal and the number that reaches the paper come from the same
code.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from blindspot import config

_MODULES = {
    "results": "results",
    "mitigations": "mitigations",
    "standpoints": "standpoints",
}


def load(name: str) -> ModuleType:
    if name not in _MODULES:
        raise KeyError(f"unknown table {name!r}. known: {', '.join(_MODULES)}")
    return importlib.import_module(f"blindspot.tables.{_MODULES[name]}")


def names() -> list[str]:
    return list(_MODULES)


def describe() -> list[tuple[str, str, list[str]]]:
    out = []
    for name in _MODULES:
        module = load(name)
        out.append((name, module.TITLE, [r.name for r in module.REQUIRES]))
    return out


def render(name: str, out_dir: str | Path | None = None) -> tuple[Path, str]:
    """render one table. returns the .tex path written and the text twin."""
    module = load(name)
    absent = [r.name for r in module.REQUIRES if r.find(required=False) is None]
    if absent:
        raise config.MissingResults(
            f"{name} needs runs that aren't on disk: {', '.join(sorted(set(absent)))}.\n"
            f"run `blindspot fetch`, or point BLINDSPOT_RESULTS at them."
        )

    latex, text = module.render()
    out_dir = Path(out_dir) if out_dir else config.plots_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tex"
    path.write_text(latex, encoding="utf-8")
    return path, text
