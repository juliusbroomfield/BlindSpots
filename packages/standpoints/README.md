# standpoints

When a user prompts a LLM to produce an output that affects an external
group, the model must implicitly act on behalf of people whose preferences
the user may not fully know. Thus, it is important to assess where the
models fail to surface such materially relevant but unstated preferences

We introduce Mixture of Standpoints (MoS), an adaptive inference-time method
that critiques and revises a response using persona conditioned models and merges
their critiques into a single response. 

```bash
pip install standpoints
```

```python
from standpoints import respond

respond("Design a lunch menu for 200 students.", model="gpt-5-mini")
```

## How it works

There are three stages, each parallel inside itself, so latency is three sequential calls
however many standpoints activate:

1. **filter** — each of 8 broad standpoints is asked whether an answer to
   this request would actually fail them.
2. **critique** — the ones that activated name the single thing they'd need that
   the answer omits.
3. **merge** — the request is answered with those gaps folded in, in a critique and revise fashion.

### Seeing what fired

```python
from standpoints import MoS

result = MoS(model="gpt-5-mini").run("Plan a 3-day company offsite for 40 people.")

result.answer          # the merged response
result.activated       # ['Religious minority', 'Allergy, intolerance, or sensitivity', ...]
result.consulted       # the individual standpoints those cover
result.gaps            # [Gap(standpoint='Muslim', comment='prayer space and halal options...')]
```

### Bringing your own client

```python
def execute(prompts, max_tokens):
    return my_batch_api(prompts, max_tokens)   # list[str | None], same order

MoS(model="gpt-5-mini", execute=execute).run_many(prompts)
```

## Citation

```bibtex
@inproceedings{broomfield2026standpoint,
  title     = {Whose Standpoint do {LLMs} Reflect? Surfacing and Mitigating Epistemic Blindspots},
  author    = {Broomfield, Julius and Sharma, Kartik},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```
