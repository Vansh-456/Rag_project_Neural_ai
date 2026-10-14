"""Retrieval metrics over *labels* (chunking-independent relevance).

Ground truth is a list of labelled passages per query. A label is ``(doc_id, anchors,
grade)``; a chunk *satisfies* a label when it belongs to ``doc_id`` and contains one of the
label's anchor strings. Because labels are defined on text spans rather than chunk ids, the
same ground truth evaluates every chunker fairly (see ``docs/labeling_guideline.md``).

Given the ranked chunk list r_1..r_n and labels L (|L| = R):

  hit@K       = 1 if any of r_1..r_K satisfies any label
  Recall@K    = |{l in L : some r_i (i<=K) satisfies l}| / R
  Precision@K = |{i <= K : r_i satisfies some label}| / K
  MRR         = 1 / (rank of first chunk satisfying any label), 0 if none
  NDCG@K      = DCG@K / IDCG@K with
                DCG@K  = sum_{i<=K} gain_i / log2(i + 1),
                gain_i = grade of the best *newly* satisfied label at rank i (each label
                         counts once — redundant chunks for the same passage earn nothing)
                IDCG@K = DCG of the label grades sorted descending, truncated to K.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    doc_id: str
    anchors: tuple[str, ...]
    grade: int


def satisfied_labels(chunk_doc: str, chunk_text: str, labels: Sequence[Label]) -> list[int]:
    """Indices of labels this chunk satisfies (anchor match is case/space-insensitive)."""
    norm = " ".join(chunk_text.lower().split())
    out = []
    for i, lab in enumerate(labels):
        if lab.doc_id != chunk_doc or lab.grade <= 0:
            continue
        if not lab.anchors or any(" ".join(a.lower().split()) in norm for a in lab.anchors):
            out.append(i)
    return out


def compute_metrics(
    matches: Sequence[list[int]], labels: Sequence[Label], ks: Sequence[int]
) -> dict[str, float]:
    """``matches[i]`` = label indices satisfied by the chunk at rank i+1."""
    grades = [lab.grade for lab in labels if lab.grade > 0]
    R = len(grades)
    out: dict[str, float] = {}
    first = next((i for i, m in enumerate(matches) if m), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    for k in ks:
        top = matches[:k]
        covered: set[int] = set()
        for m in top:
            covered.update(m)
        out[f"hit@{k}"] = 1.0 if any(top) else 0.0
        out[f"recall@{k}"] = len(covered) / R if R else 0.0
        out[f"precision@{k}"] = sum(1 for m in top if m) / k
        seen: set[int] = set()
        dcg = 0.0
        for i, m in enumerate(top):
            new = [j for j in m if j not in seen]
            if new:
                gain = max(labels[j].grade for j in new)
                seen.update(new)
                dcg += gain / math.log2(i + 2)
        ideal = sorted(grades, reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return out


def path_metrics(
    retrieved_nodes: Sequence[set[str]],
    retrieved_relations: Sequence[set[str]],
    expected_entities: Sequence[str],
    expected_relations: Sequence[str],
) -> dict[str, float]:
    """Path-level metrics for graph queries.

    entity_recall   = fraction of expected entities that appear on any retrieved path
    relation_recall = fraction of expected relation types traversed by any retrieved path
    path_hit        = 1 if a single path covers all expected entities
    """
    all_nodes = set().union(*retrieved_nodes) if retrieved_nodes else set()
    all_rels = set().union(*retrieved_relations) if retrieved_relations else set()
    ee, er = set(expected_entities), set(expected_relations)
    return {
        "path_entity_recall": len(ee & all_nodes) / len(ee) if ee else 0.0,
        "path_relation_recall": len(er & all_rels) / len(er) if er else 0.0,
        "path_hit": 1.0 if ee and any(ee <= nodes for nodes in retrieved_nodes) else 0.0,
    }
