# Whose Standpoint do LLMs Reflect? Surfacing and Mitigating Epistemic Blindspots

Code and data for *Whose Standpoint do LLMs Reflect? Surfacing and Mitigating
Epistemic Blindspots* (COLM 2026).

The dataset alone can be found at https://huggingface.co/datasets/juliusbroomfield/BlindSpots.

## ⚙️ Setup

```bash
git clone https://github.com/juliusbroomfield/BlindSpots.git
cd BlindSpots

poetry install --with dev -E analysis
poetry shell

cp .env.example .env # fill in whichever providers you'll use
set -a && source .env && set +a

blindspot check
```

## 🚀 Usage

```bash
blindspot run --model gpt-5-mini
blindspot run --model anthropic/claude-sonnet-4-5
blindspot run --model together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo
blindspot run --model hosted_vllm/meta-llama/Llama-3.1-8B-Instruct \
              --api-base http://localhost:8000/v1
```

A run is a model plus a method

```
results/gpt-5-mini.20260819T1422.jsonl
results/gpt-5-mini.persona.20260819T1422.jsonl
```

Judging follows the same idea

```bash
blindspot judge -i results/gpt-5-mini.20260819T1422.jsonl
# -> results/gpt-5-mini.20260819T1422.scored.jsonl
```

Mitigations are a flag on the same command

```bash
blindspot run --model gpt-5-mini --method persona
blindspot run --model gpt-5-mini --method omni
blindspot run --model gpt-5-mini --method mos --conditions base
blindspot run --model together_ai/... --method rag --top-k 5
```

## Mixture of Standpoints (MoS)

We release MoS separately as a standalone package. There are two dependencies: (1) LiteLLM and (2) PyYAML, and source code can be found in
[`packages/standpoints/`](packages/standpoints/). 

We note that all MoS code in the repo requires importing the package
```bash
pip install standpoints
```

```python
from standpoints import respond

respond("Design a lunch menu for 200 students.", model="gpt-5-mini")
```

Inside this repo there's also a wrapper for running it over a file of your own
prompts, which routes through the batch endpoints

```bash
blindspot mos --model gpt-5-mini --input my_prompts.jsonl
```

## Inspect?

There's an [Inspect AI](https://inspect.aisi.org.uk) task if you'd rather not
adopt anything from here

```bash
pip install inspect-ai
inspect eval src/blindspot/task.py --model openai/gpt-5-mini --batch
```

## Reproducing the paper

```bash
blindspot fetch      # the benchmark
blindspot figures    # figures 3-12 into plots/
blindspot tables     # results, mitigations, standpoint appendix
blindspot stats      # every number reported
```

The benchmark is on the Hub at
[juliusbroomfield/BlindSpots](https://huggingface.co/datasets/juliusbroomfield/BlindSpots),
so you can skip this repo entirely if all you want is the data

```python
from datasets import load_dataset
ds = load_dataset("juliusbroomfield/BlindSpots")
```
## 🤝 Citation

```bibtex
@inproceedings{broomfield2026standpoint,
  title     = {Whose Standpoint do LLMs Reflect? Surfacing and Mitigating Epistemic Blindspots},
  author    = {Broomfield, Julius and Sharma, Kartik},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```
