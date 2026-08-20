"""leakage checks, personas, and writing the benchmark out."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from blindspot import personas
from blindspot.data import CONDITIONS, write

# words that give the game away in the base condition
LEAK = re.compile(
    r"\b(inclusiv\w*|accessib\w*|divers\w*|equit\w*|accommodat\w*|"
    r"disabilit\w*|marginali[sz]\w*|underserved)\b",
    re.I,
)


def leaks(prompt: str, need: str, groups: list[str]) -> bool:
    """
    does this base prompt give away what it's testing?

    two ways it can: inclusivity framing, which primes the model regardless of
    the need, or naming the group or a distinctive word from the need outright.
    """
    if LEAK.search(prompt):
        return True

    lowered = prompt.lower()
    if any(g.lower() in lowered for g in groups if len(g) > 4):
        return True

    distinctive = [w for w in re.findall(r"\w{6,}", need.lower())][:6]
    return sum(w in lowered for w in distinctive) >= 2


def assemble(entries: list[dict[str, Any]], output: str | Path,
             seed: int = 42) -> list[dict[str, Any]]:
    """flatten to one row per prompt, dropping scenarios that leak."""
    rng = random.Random(seed)
    rows, dropped = [], 0

    for index, entry in enumerate(entries):
        need_id = f"n{index:04d}"
        groups = entry.get("groups", [])
        group = rng.choice(groups) if groups else ""
        persona = personas.persona_for_group(group)

        for scenario_index, scenario in enumerate(entry.get("scenarios", [])):
            base = scenario["prompts"].get("base", "")
            if leaks(base, entry["need"], groups):
                dropped += 1
                continue

            for condition in CONDITIONS:
                prompt = scenario["prompts"].get(condition)
                if not prompt:
                    continue
                rows.append({
                    "id": f"{need_id}/s{scenario_index}/{condition}",
                    "need_id": need_id,
                    "scenario": scenario_index,
                    "condition": condition,
                    "task": scenario.get("task", "creation"),
                    "prompt": prompt,
                    "need": entry["need"],
                    "groups": groups,
                    "subgroups": entry.get("subgroups", []),
                    "domains": entry.get("domains", []),
                    "sources": entry.get("sources", []),
                    "persona": persona,
                    "persona_group": group,
                })

    write(output, rows)
    print(f"{len(rows)} prompts from {len(entries)} needs "
          f"({dropped} scenarios dropped for leakage) -> {output}")
    return rows
