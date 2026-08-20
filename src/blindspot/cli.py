"""
the command line.

    blindspot check      what's on disk
    blindspot fetch      pull the results archive
    blindspot run        get responses out of a model
    blindspot judge      score them for target need recall
    blindspot validate   check the judge against the human labels
    blindspot precision  count irrelevant needs
    blindspot mos        run mixture of standpoints over set of prompts
    blindspot stats      recompute every number the paper quotes
    blindspot figures    redraw the paper's figures
    blindspot cost       prices a run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from blindspot import config
from blindspot.data import CONDITIONS

MODEL_HELP = (
    "LiteLLM model id — gpt-5-mini, anthropic/claude-haiku-4-5, "
    "together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo, "
    "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"
)


def _conditions(value: str) -> list[str]:
    return [c.strip().lower() for c in value.split(",") if c.strip()]


def check(args) -> int:
    print(f"Repository  {config.REPO_ROOT}\n")
    print("Benchmark")
    ok = True
    for label, path in [
        ("benchmark", config.BENCHMARK),
        ("need bank", config.NEEDS),
        ("human labels", config.ANNOTATIONS),
    ]:
        present = path.exists()
        ok &= present
        size = f"{path.stat().st_size / 1e6:.1f} MB" if present else ""
        print(f"  {'ok     ' if present else 'MISSING'}  {label:<22} {size}")

    print("\nRuns")
    found = absent = 0
    for run in config.PAPER_RUNS:
        path = run.find(required=False)
        if path:
            print(f"  ok       {run.name:<24} {path.stat().st_size / 1e6:>7.1f} MB")
            found += 1
        else:
            print(f"  MISSING  {run.name}")
            absent += 1
    missing = absent

    print(f"\n{found} runs present, {absent} missing.")
    if missing:
        print("Run `blindspot fetch`, or point BLINDSPOT_RESULTS at where they live.")
        print("Figures, tables and stats need these. Running and judging don't.")
    return 0 if ok else 1


def fetch(args) -> int:
    from blindspot.fetch import fetch as do_fetch

    return do_fetch(force=args.force)


def run(args) -> int:
    from blindspot.eval import generate

    generate.run(
        model=args.model,
        output=args.output,
        method=args.method,
        conditions=_conditions(args.conditions),
        limit=args.limit,
        benchmark=args.benchmark,
        max_tokens=args.max_tokens,
        reasoning_effort=args.effort,
        verbosity=args.verbosity,
        api_base=args.api_base,
        rag_top_k=args.top_k,
        rag_sources=args.sources,
        max_workers=args.workers,
    )
    return 0


def judge(args) -> int:
    from blindspot.eval import judge as scorer

    scorer.run(
        source=args.input,
        output=args.output,
        model=args.model,
        max_tokens=args.max_tokens,
        response_field=args.field,
    )
    return 0


def validate(args) -> int:
    from blindspot.eval.validate import judge_agreement

    judge_agreement(args.input, n_boot=args.bootstrap)
    return 0


def precision(args) -> int:
    from blindspot.eval.validate import false_needs

    false_needs(
        runs=[config.parse(spec) for spec in args.runs],
        model=args.model,
        n_sample=args.n,
        seed=args.seed,
        output_path=args.output,
    )
    return 0


def mos(args) -> int:
    from blindspot.data import _read, write
    from blindspot.llm import GenConfig
    from blindspot.mos import answer

    rows = _read(Path(args.input))
    prompts = [r["prompt"] for r in rows if r.get("prompt")]
    if not prompts:
        raise ValueError(f"no rows with a `prompt` field in {args.input}")

    cfg = GenConfig(model=args.model, max_tokens=args.max_tokens, api_base=args.api_base)
    answers = answer(prompts, cfg, checkpoint_stem=str(Path(args.output or "mos").with_suffix("")))

    out = args.output or Path(args.input).with_name(Path(args.input).stem + ".mos.jsonl")
    write(out, [{**row, "response": text} for row, text in zip(rows, answers, strict=True)])
    print(f"{len(answers)} responses -> {out}")
    return 0


def stats(args) -> int:
    from blindspot.stats import report

    report()
    return 0


def cost(args) -> int:
    from blindspot.cost import estimate

    estimate(
        model=args.model,
        method=args.method,
        conditions=_conditions(args.conditions),
        limit=args.limit,
        expected_output_tokens=args.output_tokens,
        judge_model=args.judge_model,
    )
    return 0


def _render(registry, names, out_dir, keep_going, kind):
    """shared driver for figures and tables."""
    known = registry.names()
    unknown = [n for n in names if n not in known]
    if unknown:
        print(f"Unknown {kind}: {', '.join(unknown)}\nAvailable: {', '.join(known)}")
        return 1

    made, skipped, failed = [], [], []
    for name in names:
        try:
            result = registry.render(name, out_dir)
        except config.MissingResults as e:
            print(f"  skip  {name:<22} {str(e).splitlines()[0].split(': ', 1)[-1]}")
            skipped.append(name)
            continue
        except Exception as e:  # noqa: BLE001 — one bad figure shouldn't sink the run
            print(f"  FAIL  {name:<22} {type(e).__name__}: {e}")
            failed.append(name)
            if not keep_going:
                return 1
            continue
        if kind == "table":
            path, text = result
            print(f"  ok    {name:<22} -> {path.name}\n\n{text}\n")
        else:
            print(f"  ok    {name:<22} -> {result[0].stem}.{{png,pdf,svg}}")
        made.append(name)

    print(f"\n{len(made)} generated, {len(skipped)} skipped, {len(failed)} failed.")
    if skipped:
        print("Skipped items need the results archive — run `blindspot fetch`.")
    return 1 if failed else 0


def figures(args) -> int:
    from blindspot import figures as registry

    if args.list:
        for name, title, requires in registry.describe():
            print(f"  {name:<8} {title}")
            print(f"  {'':<8} needs: {', '.join(requires)}")
        return 0
    out_dir = args.out or config.plots_dir()
    print(f"Writing to {out_dir}\n")
    return _render(registry, args.names or registry.names(), out_dir, args.keep_going, "figure")


def tables(args) -> int:
    from blindspot import tables as registry

    if args.list:
        for name, title, _ in registry.describe():
            print(f"  {name:<14} {title}")
        return 0
    out_dir = args.out or config.plots_dir()
    print(f"Writing to {out_dir}\n")
    return _render(registry, args.names or registry.names(), out_dir, args.keep_going, "table")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="blindspot", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    root.add_argument("--version", action="version",
                      version=f"blindspot {__import__('blindspot').__version__}")
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="report what's on disk")
    p.set_defaults(func=check)

    p = sub.add_parser("fetch", help="download the results archive")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.set_defaults(func=fetch)

    p = sub.add_parser("run", help="get responses out of a model")
    p.add_argument("--model", required=True, help=MODEL_HELP)
    p.add_argument("--output", "-o", help="output path (default: results/<model>.<method>.<stamp>.jsonl)")
    p.add_argument("--method", default="none", choices=["none", "persona", "omni", "rag", "mos"],
                   help="mitigation to apply (default: none)")
    p.add_argument("--conditions", default=",".join(CONDITIONS),
                   help=f"cue conditions (default: all four — {', '.join(CONDITIONS)})")
    p.add_argument("--limit", type=int, help="first N prompts only, for a smoke test")
    p.add_argument("--benchmark", help="override the benchmark file")
    p.add_argument("--max-tokens", type=int, default=15000)
    p.add_argument("--effort", default="medium", choices=["minimal", "low", "medium", "high"],
                   help="reasoning effort, GPT-5 family only (paper: medium)")
    p.add_argument("--verbosity", default="medium", choices=["low", "medium", "high"],
                   help="GPT-5 family only (paper: medium)")
    p.add_argument("--api-base", help="base URL for a self-hosted OpenAI-compatible server")
    p.add_argument("--top-k", type=int, default=5, help="retrieved chunks for rag (paper: 2 or 5)")
    p.add_argument("--sources", help="directory of source documents for rag")
    p.add_argument("--workers", type=int, default=8, help="concurrency for live-call fallback")
    p.set_defaults(func=run)

    p = sub.add_parser("judge", help="score responses for target need recall")
    p.add_argument("--input", "-i", required=True, help="responses from `blindspot run`")
    p.add_argument("--output", "-o", help="output path (default: alongside the input, .scored.jsonl)")
    p.add_argument("--model", default="anthropic/claude-haiku-4-5",
                   help="judge model (paper: claude-haiku-4-5)")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--field", default="response",
                   help="field holding the model text (MoS uses final_response)")
    p.set_defaults(func=judge)

    p = sub.add_parser("validate", help="check the judge against the human labels")
    p.add_argument("--input", "-i", help="annotation file (default: the shipped one)")
    p.add_argument("--bootstrap", type=int, default=10_000, help="bootstrap resamples")
    p.set_defaults(func=validate)

    p = sub.add_parser("precision", help="count irrelevant needs each result set surfaces")
    p.add_argument("runs", nargs="*", default=[r.name for r in config.BASELINES],
                   help="runs to compare, as model or model+method "
                        "(default: the six baselines)")
    p.add_argument("--model", default="anthropic/claude-sonnet-4-5", help="scoring model")
    p.add_argument("-n", type=int, default=20, help="prompts to sample (default: 20)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", "-o", default="precision.json")
    p.set_defaults(func=precision)

    p = sub.add_parser("mos", help="run mixture of standpoints over your own prompts")
    p.add_argument("--model", required=True, help=MODEL_HELP)
    p.add_argument("--input", "-i", required=True, help="jsonl with a `prompt` field per row")
    p.add_argument("--output", "-o", help="output path (default: <input>.mos.jsonl)")
    p.add_argument("--max-tokens", type=int, default=15000)
    p.add_argument("--api-base", help="base url for a self-hosted openai-compatible server")
    p.set_defaults(func=mos)

    p = sub.add_parser("stats", help="recompute every number the paper quotes in prose")
    p.set_defaults(func=stats)

    p = sub.add_parser("cost", help="estimate tokens and spend for a run")
    p.add_argument("--model", required=True, help=MODEL_HELP)
    p.add_argument("--method", default="none", choices=["none", "persona", "omni", "rag", "mos"])
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--limit", type=int)
    p.add_argument("--output-tokens", type=int, default=900,
                   help="assumed response length (default: 900)")
    p.add_argument("--judge-model", help="also price the judging pass")
    p.set_defaults(func=cost)

    for name, handler, helptext in [
        ("figures", figures, "redraw the paper's figures"),
        ("tables", tables, "regenerate the paper's tables"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("names", nargs="*", help="names (default: all)")
        p.add_argument("--list", action="store_true", help=f"list available {name}")
        p.add_argument("--out", help="output directory (default: plots/)")
        p.add_argument("--keep-going", action="store_true", help="continue past failures")
        p.set_defaults(func=handler)

    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (config.MissingResults, FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Re-run the same command to pick up from the checkpoint.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
