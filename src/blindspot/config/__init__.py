"""
where things live, and what things are called.

a run is a model plus a method, and that pair is its name. the filename follows
from it, so nothing has to keep a table of paths:

    Run("gpt-5-mini", "persona")  ->  gpt-5-mini.persona.20260819T1422.jsonl

anything that's data rather than logic sits in a yaml file next to this one: the
persona banks, the group taxonomy, the palette, the figure list, and the runs
behind the paper.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ANNOTATIONS",
    "BASELINES",
    "BENCHMARK",
    "DATA_DIR",
    "MissingResults",
    "NEEDS",
    "PAPER_RUNS",
    "PLOTS_DIR",
    "RESULTS_DIR",
    "Run",
    "load",
    "parse",
    "missing",
    "plots_dir",
    "stamp",
]


@cache
def load(name: str) -> dict[str, Any]:
    return yaml.safe_load(files(__package__).joinpath(name).read_text(encoding="utf-8"))


def _repo_root() -> Path:
    override = os.environ.get("BLINDSPOT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    markers = ("pyproject.toml", "data")
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if all((candidate / marker).exists() for marker in markers):
            return candidate
    return here.parent.parent


REPO_ROOT = _repo_root()

_packaged = Path(__file__).resolve().parent.parent / "data"
DATA_DIR = REPO_ROOT / "data" if (REPO_ROOT / "data").exists() else _packaged

BENCHMARK = DATA_DIR / "blindspot.jsonl"
NEEDS = DATA_DIR / "needs.json"
ANNOTATIONS = DATA_DIR / "annotations.json"

PLOTS_DIR = Path(os.environ.get("BLINDSPOT_PLOTS", REPO_ROOT / "plots"))
RESULTS_DIR = Path(os.environ.get("BLINDSPOT_RESULTS_DIR", REPO_ROOT / "results"))

# searched in order for any run file
_SEARCH = [
    Path(p) for p in os.environ.get("BLINDSPOT_RESULTS", "").split(os.pathsep) if p
] + [RESULTS_DIR, REPO_ROOT / "final_results", REPO_ROOT]

_runs = load("runs.yaml")
MODEL_LABELS: dict[str, str] = _runs["models"]


class MissingResults(FileNotFoundError):
    """a run the caller needs isn't on disk."""


def slug(model: str) -> str:
    """a filename-safe short name for a model id."""
    bare = model.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9.]+", "-", bare.lower()).strip("-")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")


@dataclass(frozen=True)
class Run:
    """one model evaluated under one method."""

    model: str
    method: str = "none"
    text_field: str = "response"

    @property
    def name(self) -> str:
        return self.model if self.method == "none" else f"{self.model}+{self.method}"

    @property
    def label(self) -> str:
        pretty = MODEL_LABELS.get(slug(self.model), self.model)
        return pretty if self.method == "none" else f"{pretty} + {self.method}"

    def filename(self, at: str | None = None) -> str:
        parts = [slug(self.model)]
        if self.method != "none":
            parts.append(self.method)
        parts.append(at or stamp())
        return ".".join(parts) + ".jsonl"

    def new_path(self, directory: Path | None = None) -> Path:
        """where a fresh run of this should be written."""
        target = (directory or RESULTS_DIR) / self.filename()
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def find(self, *, required: bool = True) -> Path | None:
        """
        the newest file for this run, or the paper's archive copy.

        matches `{model}.{method}.*.jsonl` first, so re-running something picks
        up your latest rather than the published one.
        """
        newest: Path | None = None
        for root in _SEARCH:
            if not root.is_dir():
                continue
            for candidate in root.glob("*.jsonl"):
                if _matches(candidate.name, self) and (
                    newest is None or candidate.name > newest.name
                ):
                    newest = candidate
        if newest:
            return newest

        legacy = _PAPER.get((self.model, self.method))
        if legacy:
            for root in _SEARCH:
                candidate = root / legacy
                if candidate.exists():
                    return candidate

        if not required:
            return None
        raise MissingResults(
            f"no results for {self.name}.\n"
            f"looked for {slug(self.model)}"
            f"{'.' + self.method if self.method != 'none' else ''}.*.jsonl in "
            f"{', '.join(str(r) for r in _SEARCH)}\n"
            f"run `blindspot fetch` for the paper's, or generate your own with "
            f"`blindspot run --model {self.model} --method {self.method}`."
        )

    @property
    def path(self) -> Path:
        found = self.find(required=True)
        assert found is not None
        return found


def _matches(filename: str, run: Run) -> bool:
    parts = filename[: -len(".jsonl")].split(".")
    if len(parts) < 2:
        return False
    tail = parts[-1]
    head = ".".join(parts[:-1])
    if not re.fullmatch(r"\d{8}T\d{4}", tail):
        return False
    expected = slug(run.model) if run.method == "none" else f"{slug(run.model)}.{run.method}"
    return head == expected


_PAPER: dict[tuple[str, str], str] = {}
PAPER_RUNS: list[Run] = []
for _entry in _runs["paper"]:
    _run = Run(
        model=_entry["model"],
        method=_entry.get("method", "none"),
        text_field=_entry.get("text_field", "response"),
    )
    _PAPER[(_run.model, _run.method)] = _entry["file"]
    PAPER_RUNS.append(_run)

BASELINES: list[Run] = [r for r in PAPER_RUNS if r.method == "none" and "sft" not in r.model]


def parse(spec: str) -> Run:
    """
    "gpt-5-mini" or "gpt-5-mini+persona" -> a Run.

    the paper's runs carry a text_field for multi-stage methods, so prefer a
    known one over constructing a bare Run.
    """
    model, _, method = spec.partition("+")
    method = method or "none"
    for run in PAPER_RUNS:
        if run.model == model and run.method == method:
            return run
    return Run(model=model, method=method)


def paper_run(model: str, method: str = "none") -> Run:
    """look up one of the paper's runs, so figures can name what they plot."""
    for run in PAPER_RUNS:
        if run.model == model and run.method == method:
            return run
    raise KeyError(f"no paper run for {model!r} + {method!r}")


def missing(runs: list[Run] | None = None) -> list[Run]:
    return [r for r in (runs or PAPER_RUNS) if r.find(required=False) is None]


def plots_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR
