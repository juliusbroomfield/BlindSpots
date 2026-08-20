from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blindspot import config, data
from blindspot.metrics import (
    cluster_bootstrap_ci,
    gap_sizes,
    item_map,
    mcnemar,
    need_map,
    pairwise_correlation,
    recall,
)


@dataclass
class Claim:
    section: str
    text: str
    value: str
    paper: str = ""

    def render(self) -> str:
        line = f"  {self.text:<52} {self.value:>22}"
        if self.paper:
            line += f"   paper: {self.paper}"
        return line


def _have(*specs: str) -> bool:
    return all(config.parse(spec).find(required=False) is not None for spec in specs)


def _load(spec: str):
    return data.load_run(config.parse(spec))


def collect() -> list[Claim]:
    claims: list[Claim] = []
    baselines = [r for r in config.BASELINES if r.find(required=False)]
    if not baselines:
        raise config.MissingResults(
            "No baseline results on disk. Run `blindspot fetch`, or set BLINDSPOT_RESULTS."
        )

    records = {r.label: data.load_run(r) for r in baselines}

    # section 3: how well models surface latent needs
    base_recall = {k: recall(v, "base") for k, v in records.items()}
    lo_key = min(base_recall, key=base_recall.get)
    hi_key = max(base_recall, key=base_recall.get)
    claims.append(Claim(
        "3", "Base recall range across models",
        f"{base_recall[lo_key]:.1%} – {base_recall[hi_key]:.1%}", "47.8% – 83.5%"))
    claims.append(Claim(
        "3", "Mean base recall",
        f"{np.mean(list(base_recall.values())):.1%}", "65%"))
    claims.append(Claim(
        "3", "  lowest / highest",
        f"{lo_key} / {hi_key}"))

    gaps = {k: gap_sizes(v) for k, v in records.items()}
    detection = [g["detection"] for g in gaps.values()]
    operationalization = [g["operationalization"] for g in gaps.values()]
    claims.append(Claim(
        "3", "Detection gap, mean / median",
        f"{np.mean(detection):.1%} / {np.median(detection):.1%}", "18.6% / 20.4%"))
    claims.append(Claim(
        "3", "Operationalization gap, mean / median",
        f"{np.mean(operationalization):.1%} / {np.median(operationalization):.1%}",
        "15.5% / 11.2%"))

    # appendix E.4: do models miss the same needs?
    vectors = {label: need_map(v, "base") for label, v in records.items()}
    if len(vectors) > 1:
        labels, matrix = pairwise_correlation(vectors)
        off = [matrix[i][j] for i in range(len(labels)) for j in range(len(labels)) if i < j]
        claims.append(Claim(
            "E.4", "Cross-model per-need correlation, mean",
            f"{np.mean(off):.2f}"))
        least = min(range(len(labels)),
                    key=lambda i: np.nanmean([matrix[i][j] for j in range(len(labels)) if j != i]))
        claims.append(Claim("E.4", "  least correlated model", labels[least]))

    # section 4.1: persona conditioning
    for model_label, model in [("GPT-5-mini", "gpt-5-mini"), ("Llama 3.1 8B", "llama-3.1-8b")]:
        baseline_key, persona_key = model, f"{model}+persona"
        if not _have(baseline_key, persona_key):
            continue
        baseline = _load(baseline_key)
        persona = _load(persona_key)
        group_cue = item_map(baseline, "group")
        persona_hits = item_map(persona, "base")
        test = mcnemar(persona_hits, group_cue)
        claims.append(Claim(
            "4.1", f"{model_label}: persona vs. group recall",
            f"{recall(persona, 'base'):.1%} vs {recall(baseline, 'group'):.1%}"))
        claims.append(Claim(
            "4.1", f"{model_label}: McNemar persona-only / group-only",
            f"{test['a_only']} / {test['b_only']}  p={test['p']:.1e}",
            "231 / 55, p<0.0001" if "mini" in model_label else "266 / 243, p=0.33"))

    # section 4.1: OmniPersona
    for model_label, model in [("GPT-5-mini", "gpt-5-mini"), ("Llama 3.1 8B", "llama-3.1-8b")]:
        baseline_key, omni_key = model, f"{model}+omni"
        if not _have(baseline_key, omni_key):
            continue
        guidance = recall(_load(baseline_key), "guidance")
        omni = recall(_load(omni_key), "base")
        claims.append(Claim(
            "4.1", f"{model_label}: OmniPersona vs. guidance",
            f"{omni:.1%} vs {guidance:.1%} ({(omni - guidance) * 100:+.1f} pts)",
            "+10.4 pts" if "mini" in model_label else "56.7% -> 60.5%"))

    # section 4.2: retrieval
    if _have("llama-3.1-8b", "llama-3.1-8b+rag"):
        llama = _load("llama-3.1-8b")
        rag = _load("llama-3.1-8b+rag")
        lo, hi = cluster_bootstrap_ci(rag, "base")
        claims.append(Claim(
            "4.2", "RAG base recall (Llama 3.1 8B)",
            f"{recall(rag, 'base'):.1%}  [{lo:.1%}, {hi:.1%}]", "47.8% -> 66.3%"))
        claims.append(Claim(
            "4.2", "RAG + group vs. need condition",
            f"{recall(rag, 'group'):.1%} vs {recall(llama, 'need'):.1%}", "77.2% vs 99.1%"))

        base = item_map(llama, "base")
        group = item_map(llama, "group")
        rag_hits = item_map(rag, "base")
        group_fails = [k for k in set(base) & set(group) & set(rag_hits)
                       if not base[k] and not group[k]]
        if group_fails:
            rescued = sum(1 for k in group_fails if rag_hits[k]) / len(group_fails)
            claims.append(Claim(
                "4.2", "RAG rescue rate where group cue fails",
                f"{rescued:.0%}  (n={len(group_fails)})", "65%"))

    # section 4.3: distillation
    if _have("llama-3.1-8b", "llama-3.1-8b-sft", "gpt-5"):
        sft = _load("llama-3.1-8b-sft")
        llama = _load("llama-3.1-8b")
        sft_needs = set(need_map(sft, "base"))
        held_out = [r for r in llama if r.get("need") in sft_needs]
        claims.append(Claim(
            "4.3", "SFT held-out recall (base -> distilled)",
            f"{recall(held_out, 'base'):.1%} -> {recall(sft, 'base'):.1%}", "41% -> 61.9%"))
        sft_gaps = gap_sizes(sft)
        base_gaps = gap_sizes(held_out)
        claims.append(Claim(
            "4.3", "  operationalization gap (base -> distilled)",
            f"{base_gaps['operationalization']:.0%} -> {sft_gaps['operationalization']:.0%}",
            "29 -> 10 pts"))
        claims.append(Claim(
            "4.3", "  detection gap (base -> distilled)",
            f"{base_gaps['detection']:.0%} -> {sft_gaps['detection']:.0%}",
            "essentially unchanged"))

        from scipy import stats as sps
        teacher = need_map(_load("gpt-5"), "base")
        student = need_map(sft, "base")
        parent = need_map(llama, "base")
        shared = sorted(set(teacher) & set(student) & set(parent))
        if len(shared) > 3:
            claims.append(Claim(
                "4.3", "  per-group correlation, distilled vs. base Llama",
                f"{sps.spearmanr([student[n] for n in shared], [parent[n] for n in shared]).statistic:.2f}",
                "0.81"))
            claims.append(Claim(
                "4.3", "  per-group correlation, distilled vs. GPT-5",
                f"{sps.spearmanr([student[n] for n in shared], [teacher[n] for n in shared]).statistic:.2f}",
                "0.07"))

    return claims


def report() -> list[Claim]:
    claims = collect()
    current = None
    for claim in claims:
        if claim.section != current:
            current = claim.section
            print(f"\nSection {current}")
        print(claim.render())
    print()
    return claims
