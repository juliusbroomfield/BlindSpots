from __future__ import annotations

import urllib.request
from pathlib import Path

from blindspot import config

BENCHMARK_REPO = "juliusbroomfield/BlindSpots"
BENCHMARK_FILE = "blindspot.jsonl"

RUNS_REPO = ""


def _download(repo: str, filename: str, target: Path) -> Path:
    """pull one file from the hub, falling back to a plain https fetch."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
        print(f"  huggingface-hub not installed, fetching over https: {url}")
        urllib.request.urlretrieve(url, target)
        return target

    cached = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
    target.write_bytes(Path(cached).read_bytes())
    return target


def benchmark(force: bool = False) -> Path:
    """make sure the benchmark is on disk, downloading it if not."""
    if config.BENCHMARK.exists() and not force:
        return config.BENCHMARK

    print(f"downloading the benchmark from {BENCHMARK_REPO}")
    path = _download(BENCHMARK_REPO, BENCHMARK_FILE, config.BENCHMARK)
    rows = path.read_text(encoding="utf-8").count("\n")
    print(f"  {rows:,} prompts -> {path}")
    return path


def runs() -> int:
    """download the published runs, if there are any yet."""
    absent = config.missing()
    if not absent:
        print(f"all {len(config.PAPER_RUNS)} runs already present.")
        return 0

    if not RUNS_REPO:
        print(f"{len(absent)} runs not on disk: {', '.join(r.name for r in absent)}\n")
        print("the runs aren't published yet — see data/README.md.")
        print("if you have them already, point BLINDSPOT_RESULTS at the directory:")
        print("  export BLINDSPOT_RESULTS=/path/to/results")
        return 1

    from huggingface_hub import snapshot_download

    print(f"downloading runs from {RUNS_REPO}")
    local = Path(snapshot_download(repo_id=RUNS_REPO, repo_type="dataset",
                                   allow_patterns=["runs/*", "sources/*"]))
    for source in local.rglob("*"):
        if source.is_file():
            target = config.RESULTS_DIR / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(source.read_bytes())

    still = config.missing()
    if still:
        print(f"still missing: {', '.join(r.name for r in still)}")
        return 1
    return 0


def fetch(force: bool = False) -> int:
    benchmark(force=force)
    return runs()
