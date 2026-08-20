"""statistics shared by every figure, table and reported number."""

from blindspot.metrics.agreement import (
    bootstrap_kappa_ci,
    cohens_kappa,
    icc,
    kappa_ci,
    pairwise_correlation,
)
from blindspot.metrics.gaps import Decomposition, decompose, gap_sizes
from blindspot.metrics.recall import (
    CONDITION_LABELS,
    CONDITIONS,
    base_prompts,
    cluster_bootstrap_ci,
    contingency,
    item_key,
    item_map,
    keyed,
    mcnemar,
    need_map,
    paired_delta_ci,
    recall,
    recall_by_condition,
    scored,
)

__all__ = [
    "CONDITIONS", "CONDITION_LABELS",
    "item_key", "item_map", "keyed", "base_prompts", "need_map", "scored",
    "recall", "recall_by_condition", "cluster_bootstrap_ci", "paired_delta_ci",
    "contingency", "mcnemar",
    "Decomposition", "decompose", "gap_sizes",
    "cohens_kappa", "kappa_ci", "bootstrap_kappa_ci", "icc", "pairwise_correlation",
]
