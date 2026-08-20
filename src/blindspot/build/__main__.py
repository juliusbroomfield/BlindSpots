"""
rebuild the benchmark end to end.

    python -m blindspot.build --model anthropic/claude-sonnet-4-5

the paper used claude-sonnet-4-5, but any model litellm reaches will work.
each stage writes its output so you can inspect it and restart from there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blindspot.build.assemble import assemble
from blindspot.build.label import label
from blindspot.build.needs import SOURCES, extract, filter_needs
from blindspot.build.scenarios import generate
from blindspot.llm import GenConfig, check_credentials


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="blindspot.build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-5",
                    help="any litellm model id (paper: claude-sonnet-4-5)")
    ap.add_argument("--sources", type=Path, default=SOURCES,
                    help="directory of reference documents")
    ap.add_argument("--work", type=Path, default=Path("build_out"),
                    help="where the intermediate files go")
    ap.add_argument("--output", type=Path, default=Path("data/blindspot.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N needs only, for a smoke test")
    args = ap.parse_args(argv)

    check_credentials(args.model)
    cfg = GenConfig(model=args.model, max_tokens=8000)
    args.work.mkdir(parents=True, exist_ok=True)

    documents = sorted(p for p in args.sources.rglob("*") if p.suffix in {".txt", ".md", ".json"})
    if not documents:
        raise SystemExit(f"no reference documents under {args.sources}")

    candidates = extract(documents, cfg, args.workers)
    (args.work / "candidates.json").write_text(json.dumps(candidates, indent=2))

    kept, rejected = filter_needs(candidates, cfg, args.workers)
    (args.work / "needs.json").write_text(
        json.dumps({"needs": kept, "rejected": rejected}, indent=2)
    )
    print(f"{len(kept)} needs kept, {len(rejected)} rejected")

    if args.limit:
        kept = kept[: args.limit]

    labelled = label(kept, cfg, args.workers)
    (args.work / "labelled.json").write_text(json.dumps(labelled, indent=2))
    print(f"{len(labelled)} needs labelled")

    built = generate(labelled, cfg, args.workers)
    (args.work / "scenarios.json").write_text(json.dumps(built, indent=2))

    assemble(built, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
