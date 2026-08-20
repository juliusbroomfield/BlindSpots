# BlindSpots

The BlindSpots dataset can be found on the HuggingFace Hub:
**[juliusbroomfield/BlindSpots](https://huggingface.co/datasets/juliusbroomfield/BlindSpots)**

```python
from datasets import load_dataset
ds = load_dataset("juliusbroomfield/BlindSpots")
```

We also include it in this directory, and `blindspot` downloads it automatically if
you installed the package without the checkout.

## Dataset

`blindspot.jsonl` includes **7,320** rows, one for each prompt condition; this includes **1,830** individual prompts generated from **610** needs:

```jsonc
{
  "id": "n0042/s1/base",
  "need_id": "n0042",
  "scenario": 1,
  "condition": "base", // base | guidance | group | need
  "task": "creation", // creation | advice
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

## Other Data

`needs.json` is the wider bank the benchmark was drawn from.

`annotations.json` holds the labels behind the human validation of the judge.
## License

CC BY 4.0.
