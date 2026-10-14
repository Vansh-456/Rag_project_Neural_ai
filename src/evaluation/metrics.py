"""
Retrieval and system evaluation metrics with hand-computable verification.
Includes Recall@K, Precision@K, MRR, NDCG@K (graded), and bootstrap confidence intervals.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np


def compute_recall_at_k(retrieved_ids: list[str], ground_truth: dict[str, int], k: int = 5) -> float:
    """Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|."""
    relevant_ids = {doc_id for doc_id, grade in ground_truth.items() if grade > 0}
    if not relevant_ids:
        return 1.0  # Vacuously true if nothing was relevant

    top_k_ids = set(retrieved_ids[:k])
    hits = len(top_k_ids.intersection(relevant_ids))
    return hits / len(relevant_ids)


def compute_precision_at_k(retrieved_ids: list[str], ground_truth: dict[str, int], k: int = 5) -> float:
    """Precision@K = |Retrieved@K ∩ Relevant| / K."""
    if k == 0:
        return 0.0
    relevant_ids = {doc_id for doc_id, grade in ground_truth.items() if grade > 0}
    top_k_ids = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k_ids if doc_id in relevant_ids)
    return hits / k


def compute_mrr(retrieved_ids: list[str], ground_truth: dict[str, int]) -> float:
    """MRR = 1 / rank of first relevant item (or 0 if none retrieved)."""
    relevant_ids = {doc_id for doc_id, grade in ground_truth.items() if grade > 0}
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: list[str], ground_truth: dict[str, int], k: int = 5) -> float:
    """
    NDCG@K with graded relevance:
    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    IDCG@K = DCG@K of ideal descending sort
    """
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        rel = ground_truth.get(doc_id, 0)
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)

    # Ideal DCG
    ideal_rels = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2.0 ** rel - 1.0) / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_hit_rate_at_k(retrieved_ids: list[str], ground_truth: dict[str, int], k: int = 5) -> float:
    """Hit-Rate@K = 1.0 if at least one relevant document is in top-K, else 0.0."""
    relevant_ids = {doc_id for doc_id, grade in ground_truth.items() if grade > 0}
    for doc_id in retrieved_ids[:k]:
        if doc_id in relevant_ids:
            return 1.0
    return 0.0


def bootstrap_confidence_interval(
    values: list[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> tuple[float, float, float]:
    """
    Computes (mean, lower_ci, upper_ci) using empirical bootstrap resampling.
    """
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0]

    np.random.seed(seed)
    arr = np.array(values)
    means = []
    n = len(arr)

    for _ in range(n_resamples):
        sample = np.random.choice(arr, size=n, replace=True)
        means.append(float(np.mean(sample)))

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1.0 - alpha) * 100))
    mean_val = float(np.mean(arr))

    return round(mean_val, 4), round(lower, 4), round(upper, 4)
