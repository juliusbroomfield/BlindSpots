# standpoints

Ask a model to design a 120-person office and you get desks, meeting rooms, a
kitchen, restrooms. You rarely get a prayer room, a wudu station, a lactation
room, or step-free routes — not because the model doesn't know what those are,
but because nothing in the request made the people who need them visible.

Mixture of Standpoints is an inference-time method for that. It asks a set of
standpoints whether your request would fail them, collects the specific things
they'd need, and folds those into the answer.

```bash
pip install standpoints
```

```python
from standpoints import respond

respond("Design a lunch menu for 200 students.", model="gpt-5-mini")
```

Any model [LiteLLM](https://docs.litellm.ai/docs/providers) can reach works —
`gpt-5-mini`, `anthropic/claude-sonnet-4-5`, `together_ai/…`,
`hosted_vllm/…` against a local server.

## A batch

Three stages run in parallel across the whole batch, so it's three round trips
whatever the size — much better than looping `respond`.

```python
from standpoints import answer

answers = answer(my_prompts, model="gpt-5-mini")
```

## Seeing what fired

```python
from standpoints import MoS

result = MoS(model="gpt-5-mini").run("Plan a 3-day company offsite for 40 people.")

result.answer          # the merged response
result.activated       # ['Religious minority', 'Allergy, intolerance, or sensitivity', ...]
result.consulted       # the individual standpoints those cover
result.gaps            # [Gap(standpoint='Muslim', comment='prayer space and halal options...')]
```

## How it works

Three stages, each parallel inside itself, so latency is three sequential calls
however many standpoints activate:

1. **filter** — each of 8 broad standpoints is asked whether a typical answer to
   this request would actually fail them. Most say `[NO_COMMENT]` and drop out.
2. **critique** — the ones that activated name the single thing they'd need that
   a typical answer omits, or say `[NO_GAP]`.
3. **merge** — the request is answered with those gaps folded in as prose, not
   as a bolted-on checklist.

Filtering first is what keeps this affordable: you're not paying for 60
standpoints on every request. It comes to roughly 1.4× a single generation.

The standpoints are identity-only — "You are blind.", not "You are blind. You
notice when information is conveyed visually." The second version is a
checklist, and a model that surfaces the need after being handed the checklist
tells you nothing about what the standpoint made visible on its own.

## Bringing your own client

`execute` maps prompts to answers. Replace it to route through a batch
endpoint, a shared client, your own rate limiting:

```python
def execute(prompts, max_tokens):
    return my_batch_api(prompts, max_tokens)   # list[str | None], same order

MoS(model="gpt-5-mini", execute=execute).run_many(prompts)
```

That's the hook the benchmark this came from uses, so its 1,830-prompt runs go
through provider batch endpoints at half price.

## Where this comes from

*Whose Standpoint do LLMs Reflect? Surfacing and Mitigating Epistemic
Blindspots* (COLM 2026). MoS reaches 84% recall on GPT-5-mini in that paper's
evaluation, up from 59% unaided — matching what naming the affected group
directly achieves, without needing to know the group.

The benchmark, and the code that produced those numbers, is at
[juliusbroomfield/BlindSpots](https://github.com/juliusbroomfield/BlindSpots).

```bibtex
@inproceedings{broomfield2026standpoint,
  title     = {Whose Standpoint do {LLMs} Reflect? Surfacing and Mitigating Epistemic Blindspots},
  author    = {Broomfield, Julius and Sharma, Kartik},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

MIT licensed.
