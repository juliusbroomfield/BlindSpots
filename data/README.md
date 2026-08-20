# Data

The benchmark is on the Hub:
**[juliusbroomfield/BlindSpots](https://huggingface.co/datasets/juliusbroomfield/BlindSpots)**

```python
from datasets import load_dataset
ds = load_dataset("juliusbroomfield/BlindSpots")     # 7,320 rows
```

It also ships in this directory, and `blindspot` downloads it automatically if
you installed the package without the checkout — so you never have to fetch it
by hand.

## The benchmark

`blindspot.jsonl` — 7,320 rows, one per prompt. 610 needs, three scenarios
each, four conditions per scenario.

```jsonc
{
  "id": "n0042/s1/base",
  "need_id": "n0042",
  "scenario": 1,
  "condition": "base",              // base | guidance | group | need
  "task": "creation",               // creation | advice
  "prompt": "I'm designing a menu for my new café …",
  "need": "Clear ingredient labeling for all top 9 food allergens …",
  "groups": ["Peanut allergy", "Tree nut allergy", "…"],
  "subgroups": ["People with food allergies"],
  "domains": ["Food Systems"],
  "sources": ["food.json", "food-allergy.json"],
  "persona": "You have a severe peanut allergy.",
  "persona_group": "Peanut allergy"
}
```

One row per prompt rather than a nested tree, because every consumer flattened
it anyway — and because that's the shape Hugging Face, Inspect and the eval
harnesses all expect. The persona used to live in a second parallel file; it's
a column now.

The four conditions used to be called A, C, D and E on disk. There was no B, so
even the letters didn't help you remember the order. `blindspot.data` still
reads the letters, so older result files keep working.

`groups` is fine-grained, `subgroups` is coarser. The judge prefers `subgroups`
and falls back to `groups`. Exactly one need of 610 has neither.

## The rest

`needs.json` is the wider bank the benchmark was drawn from: 777 candidates,
the 52 rejected during review, and the generation statistics — kept so the
filtering is auditable rather than something you have to take on trust.

`annotations.json` holds the labels behind the judge validation. 150 responses
sampled stratified by task, condition and judge verdict; 100 annotated before
agreement stabilised. `blindspot validate` recomputes 92% agreement and
κ = 0.84 from it.

## Model responses

The runs — every model's responses plus their judge labels — come to roughly
642 MB across 15 runs. Derived artifacts rather than source, and too big for
git, so they're published separately from the benchmark.

> **TODO before release:** upload the runs and set `RUNS_REPO` in
> `src/blindspot/fetch.py`. The same Hub dataset works: put them under a
> `runs/` prefix and `blindspot fetch` will pick them up.

```bash
blindspot fetch     # benchmark now, runs once they're published
blindspot check     # what's on disk
```

Already have them somewhere? Skip the download:

```bash
export BLINDSPOT_RESULTS=/path/to/final_results:/path/to/mitigations
```

Every result file has a short name in `src/blindspot/config.py` — `gpt-5`,
`rag`, `mos-llama` — and that registry is the only place anything looks a path
up. Rename or move a file and you change it there, nowhere else.

## License

CC BY 4.0.
