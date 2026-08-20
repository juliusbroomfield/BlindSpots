#!/usr/bin/env python3
"""
prepare fine-tuning data from GPT-5 evaluation results.

splits by domain:
  - Training (8 domains): 90% train / 10% val
  - Held-out (2 domains): Healthcare, Events & Gatherings

output files (JSONL, messages format):
  finetune_data/train.jsonl
  finetune_data/val.jsonl
  finetune_data/held_out.jsonl
  finetune_data/split_info.json
"""

import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from blindspot import paths  # noqa: E402

SEED = 42
random.seed(SEED)

INPUT_FILE = paths.results_path("gpt-5")
OUT_DIR = os.environ.get("BLINDSPOT_FINETUNE_DATA", "finetune_data")

HELD_OUT_DOMAINS = {
    "Healthcare Interactions and Settings",
    "Events and Gatherings",
}

VAL_SPLIT = 0.10


def make_messages(prompt: str, response: str) -> dict:
    """format as Llama chat template: user = prompt, assistant = GPT-5 response."""
    return {
        "messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(INPUT_FILE) as f:
        data = json.load(f)

    # base condition only — the model gets no cue, which is the point
    base_examples = [
        d for d in data
        if d.get("condition") in ("base", "A") and d.get("prompt") and d.get("response")
    ]
    print(f"Base-condition examples with responses: {len(base_examples)}")

    # group by primary domain
    by_domain: dict[str, list] = defaultdict(list)
    for d in base_examples:
        domain = d["domains"][0] if d.get("domains") else "Unknown"
        by_domain[domain].append(d)

    print("\nDomain counts:")
    for dom, items in sorted(by_domain.items(), key=lambda x: -len(x[1])):
        tag = " [HELD OUT]" if dom in HELD_OUT_DOMAINS else ""
        print(f"  {dom}: {len(items)}{tag}")

    train_examples, val_examples, held_out_examples = [], [], []

    for domain, items in by_domain.items():
        random.shuffle(items)
        examples = [make_messages(d["prompt"], d["response"]) for d in items]

        if domain in HELD_OUT_DOMAINS:
            held_out_examples.extend(examples)
        else:
            split = max(1, int(len(examples) * VAL_SPLIT))
            val_examples.extend(examples[:split])
            train_examples.extend(examples[split:])

    # shuffle final splits
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    random.shuffle(held_out_examples)

    def write_jsonl(path, records):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    write_jsonl(os.path.join(OUT_DIR, "train.jsonl"),    train_examples)
    write_jsonl(os.path.join(OUT_DIR, "val.jsonl"),      val_examples)
    write_jsonl(os.path.join(OUT_DIR, "held_out.jsonl"), held_out_examples)

    info = {
        "seed":              SEED,
        "val_split":         VAL_SPLIT,
        "held_out_domains":  sorted(HELD_OUT_DOMAINS),
        "train_count":       len(train_examples),
        "val_count":         len(val_examples),
        "held_out_count":    len(held_out_examples),
        "total":             len(train_examples) + len(val_examples) + len(held_out_examples),
    }
    with open(os.path.join(OUT_DIR, "split_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\nSplit complete:")
    print(f"  Train:    {len(train_examples)}")
    print(f"  Val:      {len(val_examples)}")
    print(f"  Held-out: {len(held_out_examples)}")
    print(f"\nOutput → {OUT_DIR}/")


if __name__ == "__main__":
    main()
