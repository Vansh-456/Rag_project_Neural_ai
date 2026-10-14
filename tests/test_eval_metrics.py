"""Metrics against hand-computed examples; bootstrap utilities."""

import math

import pytest

from agvrag.evaluation.metrics import Label, compute_metrics, path_metrics, satisfied_labels
from agvrag.evaluation.stats import bootstrap_ci, paired_bootstrap

LABELS = [Label("d1", ("alpha",), 2), Label("d2", ("beta",), 1)]


def test_satisfied_labels_anchor_and_doc():
    assert satisfied_labels("d1", "The  ALPHA method", LABELS) == [0]
    assert satisfied_labels("d2", "alpha", LABELS) == []  # wrong document
    assert satisfied_labels("d1", "nothing", [Label("d1", (), 1)]) == [0]  # doc-level label


def test_hand_computed_metrics():
    # ranks: 1 irrelevant, 2 satisfies label0 (grade 2), 3 satisfies label0 again, 4 label1 (grade 1)
    matches = [[], [0], [0], [1]]
    m = compute_metrics(matches, LABELS, [1, 3, 5])
    assert m["mrr"] == pytest.approx(0.5)
    assert m["hit@1"] == 0 and m["hit@3"] == 1
    assert m["recall@3"] == pytest.approx(0.5) and m["recall@5"] == pytest.approx(1.0)
    assert m["precision@3"] == pytest.approx(2 / 3)
    assert m["precision@5"] == pytest.approx(3 / 5)
    dcg = 2 / math.log2(3) + 1 / math.log2(5)  # rank 3 repeats label0: no gain
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert m["ndcg@5"] == pytest.approx(dcg / idcg)
    assert m["ndcg@1"] == 0.0


def test_perfect_ranking_scores_one():
    m = compute_metrics([[0], [1]], LABELS, [2])
    assert m["ndcg@2"] == pytest.approx(1.0) and m["recall@2"] == 1.0 and m["mrr"] == 1.0


def test_path_metrics():
    m = path_metrics([{"a", "b"}, {"b", "c"}], [{"X"}, {"Y"}], ["a", "c"], ["X", "Z"])
    assert m["path_entity_recall"] == 1.0
    assert m["path_relation_recall"] == 0.5
    assert m["path_hit"] == 0.0  # no single path covers both a and c


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    mean, lo, hi = bootstrap_ci(vals, 500, seed=1)
    assert lo <= mean <= hi
    assert bootstrap_ci(vals, 500, seed=1) == (mean, lo, hi)


def test_paired_bootstrap_detects_clear_difference():
    a = [0.1] * 30
    b = [0.9] * 30
    r = paired_bootstrap(a, b)
    assert r["diff"] == pytest.approx(0.8) and r["p_value"] < 0.01
    same = paired_bootstrap([0.5, 0.6, 0.4], [0.6, 0.4, 0.5])
    assert same["p_value"] > 0.2
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])
