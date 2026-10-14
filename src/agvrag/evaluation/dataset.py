"""Evaluation dataset: queries, label-based ground truth, expected graph paths, splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agvrag.chunking.chunkers import chunk_embed_text
from agvrag.config import Settings
from agvrag.evaluation.metrics import Label, satisfied_labels
from agvrag.schemas import Chunk
from agvrag.utils.io import read_json

QUERY_TYPES = ("definition", "relationship", "comparison", "multi_hop", "experimental")


@dataclass
class EvalQuery:
    id: str
    text: str
    type: str
    split: str
    answerable: bool = True
    tags: list[str] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    expected_entities: list[str] = field(default_factory=list)
    expected_relations: list[str] = field(default_factory=list)
    reference_answer: str = ""


def load_eval_set(settings: Settings, split: str | None = None) -> list[EvalQuery]:
    ev = settings.evaluation
    queries = read_json(settings.resolve(ev.queries_file))
    gt = read_json(settings.resolve(ev.ground_truth_file))
    paths_file = settings.resolve(ev.expected_paths_file)
    paths: dict[str, Any] = read_json(paths_file) if Path(paths_file).exists() else {}
    split = split or ev.split
    out: list[EvalQuery] = []
    for q in queries:
        if q["type"] not in QUERY_TYPES:
            raise ValueError(f"query {q['id']} has unknown type {q['type']}")
        if split != "all" and q["split"] != split:
            continue
        labels = [
            Label(lab["doc_id"], tuple(lab.get("anchors", [])), int(lab.get("grade", 1)))
            for lab in gt.get(q["id"], [])
        ]
        p = paths.get(q["id"], {})
        out.append(
            EvalQuery(
                id=q["id"],
                text=q["text"],
                type=q["type"],
                split=q["split"],
                answerable=bool(q.get("answerable", True)),
                tags=list(q.get("tags", [])),
                labels=labels,
                expected_entities=list(p.get("entities", [])),
                expected_relations=list(p.get("relations", [])),
                reference_answer=q.get("reference_answer", ""),
            )
        )
    return out


def label_coverage(queries: list[EvalQuery], chunks: dict[str, Chunk]) -> dict[str, Any]:
    """For each label, how many index chunks satisfy it. Zero = broken anchor for this chunker
    (or the anchor is split across a chunk boundary) — must be fixed before trusting numbers."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks.values():
        by_doc.setdefault(c.doc_id, []).append(c)
    unmapped: list[dict[str, Any]] = []
    counts: list[int] = []
    for q in queries:
        for i, lab in enumerate(q.labels):
            n = sum(
                1
                for c in by_doc.get(lab.doc_id, [])
                if satisfied_labels(c.doc_id, chunk_embed_text(c), [lab])
            )
            counts.append(n)
            if n == 0:
                unmapped.append(
                    {"query": q.id, "label": i, "doc_id": lab.doc_id, "anchors": list(lab.anchors)}
                )
    return {
        "n_labels": len(counts),
        "n_unmapped": len(unmapped),
        "unmapped": unmapped,
        "mean_chunks_per_label": round(sum(counts) / max(1, len(counts)), 2),
    }
