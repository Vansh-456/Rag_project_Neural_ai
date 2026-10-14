"""
Unit tests for retrieval and ranking evaluation metrics against hand-computed values.
"""

import pytest
from src.evaluation.metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_hit_rate_at_k,
    bootstrap_confidence_interval,
)


def test_recall_at_k_hand_computed():
    ground_truth = {"doc_a": 2, "doc_b": 1, "doc_c": 0}
    # Relevant documents are doc_a and doc_b (total = 2)

    # Top-2 retrieves doc_a and doc_c -> hits = 1 -> recall = 1/2 = 0.5
    assert compute_recall_at_k(["doc_a", "doc_c"], ground_truth, k=2) == 0.5

    # Top-2 retrieves doc_a and doc_b -> hits = 2 -> recall = 2/2 = 1.0
    assert compute_recall_at_k(["doc_a", "doc_b"], ground_truth, k=2) == 1.0

    # Top-2 retrieves doc_c and doc_d -> hits = 0 -> recall = 0.0
    assert compute_recall_at_k(["doc_c", "doc_d"], ground_truth, k=2) == 0.0


def test_precision_at_k_hand_computed():
    ground_truth = {"doc_a": 1, "doc_b": 1}

    # Top-2 retrieves doc_a, doc_c -> hits = 1 -> precision = 1/2 = 0.5
    assert compute_precision_at_k(["doc_a", "doc_c"], ground_truth, k=2) == 0.5

    # Top-4 retrieves doc_a, doc_b, doc_x, doc_y -> hits = 2 -> precision = 2/4 = 0.5
    assert compute_precision_at_k(["doc_a", "doc_b", "doc_x", "doc_y"], ground_truth, k=4) == 0.5


def test_mrr_hand_computed():
    ground_truth = {"doc_gold": 1}

    # First position -> 1/1 = 1.0
    assert compute_mrr(["doc_gold", "doc_other"], ground_truth) == 1.0

    # Second position -> 1/2 = 0.5
    assert compute_mrr(["doc_other", "doc_gold"], ground_truth) == 0.5

    # Third position -> 1/3 = 0.3333...
    assert pytest.approx(compute_mrr(["d1", "d2", "doc_gold"], ground_truth), 0.01) == 0.3333

    # None found -> 0.0
    assert compute_mrr(["d1", "d2"], ground_truth) == 0.0


def test_ndcg_at_k_hand_computed():
    # Graded relevance: doc_best: 2, doc_good: 1
    ground_truth = {"doc_best": 2, "doc_good": 1}

    # Perfect ranking: [doc_best, doc_good]
    perfect_ndcg = compute_ndcg_at_k(["doc_best", "doc_good"], ground_truth, k=2)
    assert pytest.approx(perfect_ndcg, 0.001) == 1.0

    # Inverted ranking: [doc_good, doc_best]
    # DCG = (2^1 - 1)/log2(2) + (2^2 - 1)/log2(3) = 1/1 + 3/1.58496 = 1 + 1.8928 = 2.8928
    # IDCG = (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3) = 3/1 + 1/1.58496 = 3 + 0.6309 = 3.6309
    # NDCG = 2.8928 / 3.6309 = 0.7967
    inverted_ndcg = compute_ndcg_at_k(["doc_good", "doc_best"], ground_truth, k=2)
    assert 0.78 < inverted_ndcg < 0.82


def test_hit_rate():
    gt = {"doc_a": 1}
    assert compute_hit_rate_at_k(["doc_b", "doc_a"], gt, k=2) == 1.0
    assert compute_hit_rate_at_k(["doc_b", "doc_c"], gt, k=2) == 0.0


def test_bootstrap_confidence_interval():
    data = [0.8, 0.85, 0.9, 0.82, 0.88, 0.84]
    mean_val, low, high = bootstrap_confidence_interval(data, n_resamples=500)
    assert low <= mean_val <= high
    assert 0.75 < low < 0.95
