"""
loading the benchmark and the results computed from it.

the benchmark is one JSONL row per prompt. that shape is deliberate: it streams,
it diffs, it maps straight onto a Hugging Face dataset, and it's what every eval
harness expects. the nested need -> scenarios -> variants layout the project
started with had to be flattened by every consumer anyway.

a note on the four conditions. they used to be called A, C, D and E on disk,
which meant you had to keep a lookup table in your head — and there was no B, so
even the letters didn't help. they're now named for what they do: base,
guidance, group, need. result files produced before that change still use the
letters, so `load_judged` translates them on the way in.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from blindspot.config import BENCHMARK, Run

# in the order the cue gets stronger.
CONDITIONS = ("base", "guidance", "group", "need")

CONDITION_LABELS = {
    "base": "Base",
    "guidance": "Guidance",
    "group": "Group",
    "need": "Need",
}

# what the letters meant, for reading older result files.
_LEGACY = {"A": "base", "C": "guidance", "D": "group", "E": "need"}


def normalise_condition(value: str) -> str:
    """accept a condition name or one of the old letters."""
    if not value:
        return ""
    value = value.strip()
    return _LEGACY.get(value.upper(), value.lower())


@dataclass
class Item:
    """one prompt: a need, put into a scenario, disclosed at one cue strength."""

    id: str
    need_id: str
    scenario: int
    condition: str
    task: str
    prompt: str
    need: str
    groups: list[str] = field(default_factory=list)
    subgroups: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    persona: str = ""
    persona_group: str = ""

    @property
    def group_label(self) -> str:
        """
        a readable name for the affected group, for the judge prompt.

        prefer the coarse label, fall back to the fine-grained one. exactly one
        need of 610 has neither.
        """
        if self.subgroups:
            return ", ".join(self.subgroups)
        if self.groups:
            return ", ".join(self.groups)
        return "Not specified"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["group_label"] = self.group_label
        return out


def load_items(
    path: str | Path | None = None,
    conditions: list[str] | tuple[str, ...] = CONDITIONS,
    limit: int | None = None,
) -> list[Item]:
    """read the benchmark, optionally narrowed to some conditions."""
    if path is None and not BENCHMARK.exists():
        from blindspot.fetch import benchmark
        benchmark()
    path = Path(path) if path else BENCHMARK
    wanted = {normalise_condition(c) for c in conditions}
    unknown = wanted - set(CONDITIONS)
    if unknown:
        raise ValueError(
            f"Unknown condition(s): {sorted(unknown)}. Pick from {list(CONDITIONS)}"
        )

    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["condition"] = normalise_condition(row.get("condition", ""))
            if row["condition"] not in wanted:
                continue
            items.append(Item(**{k: v for k, v in row.items() if k in Item.__annotations__}))
            if limit is not None and len(items) >= limit:
                break
    return items


def load_run(run: Run | str | Path) -> list[dict[str, Any]]:
    """
    read a scored run, by Run or by path.

    records come out normalised: `condition` is always a name, and the model's
    text is always under `response` even for multi-stage methods that store it
    somewhere else.
    """
    if isinstance(run, Run):
        path, text_field = run.path, run.text_field
    else:
        path, text_field = Path(run), "response"

    records = _read(path)
    for record in records:
        record["condition"] = normalise_condition(
            record.get("condition") or record.get("level") or ""
        )
        record.pop("level", None)
        if text_field != "response" and "response" not in record:
            record["response"] = record.get(text_field, "")
        record.setdefault("response", "")
    return records


def _read(path: Path) -> list[dict[str, Any]]:
    """JSON array or JSONL, whichever this file happens to be."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError(f"{path} should hold a list of records")
        return records
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write(path: str | Path, records: list[dict[str, Any]]) -> Path:
    """write records as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
