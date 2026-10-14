"""Labelling helper (CLI). Pools candidates from dense + BM25 for a query, shows them, and
records graded labels with anchors into ``ground_truth.json``.

Workflow per candidate: type ``0``/``1``/``2`` for the grade, then an anchor — a short,
distinctive phrase copied from the passage (<= 8 words) that identifies *this* passage
independently of chunk boundaries. ``s`` skips, ``q`` saves and quits.
"""

from __future__ import annotations

import sys

from agvrag.config import Settings
from agvrag.pipeline.components import ComponentRegistry
from agvrag.retrieval.fusion import rrf_fuse
from agvrag.retrieval.retrievers import DenseRetriever, SparseRetriever
from agvrag.schemas import QueryAnalysis
from agvrag.utils.io import read_json, write_json


def candidate_pool(settings: Settings, query: str, pool: int, registry: ComponentRegistry) -> list:
    idx = registry.index(settings)
    dense = DenseRetriever(registry.embedder(settings), idx.vector_store)
    sparse = SparseRetriever(idx.bm25)
    qa = QueryAnalysis(query, "unknown", [], 0.0, 0, "")
    return rrf_fuse(
        {"dense": dense.retrieve(query, qa, pool), "bm25": sparse.retrieve(query, qa, pool)},
        60,
        None,
        pool,
    )


def interactive_label(settings: Settings, query_id: str, pool: int = 15) -> int:
    ev = settings.evaluation
    queries = {q["id"]: q for q in read_json(settings.resolve(ev.queries_file))}
    if query_id not in queries:
        sys.stderr.write(f"unknown query id {query_id}\n")
        return 2
    gt_path = settings.resolve(ev.ground_truth_file)
    gt = read_json(gt_path)
    q = queries[query_id]
    labels = gt.setdefault(query_id, [])
    out = sys.stdout
    out.write(f"\n{query_id} [{q['type']}] {q['text']}\nexisting labels: {labels}\n\n")
    for ev_ in candidate_pool(settings, q["text"], pool, ComponentRegistry()):
        p = ev_.provenance
        out.write(
            f"--- #{ev_.rank} {ev_.source_doc_ids[0]} | {p.get('doc_title', '')[:60]} | {p.get('section')}\n"
        )
        out.write(ev_.content[:900] + "\n")
        grade = input("grade [0/1/2, s=skip, q=quit]: ").strip().lower()
        if grade == "q":
            break
        if grade not in {"1", "2"}:
            continue
        anchor = input("anchor phrase (copied from passage): ").strip()
        if anchor and anchor.lower() not in " ".join(ev_.content.lower().split()):
            out.write("  ! anchor not found verbatim in passage; label skipped\n")
            continue
        labels.append(
            {
                "doc_id": ev_.source_doc_ids[0],
                "anchors": [anchor] if anchor else [],
                "grade": int(grade),
            }
        )
    write_json(gt_path, gt)
    out.write(f"saved {len(labels)} labels for {query_id}\n")
    return 0
