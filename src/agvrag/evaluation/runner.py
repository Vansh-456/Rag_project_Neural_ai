"""Experiment runner + results registry.

``run_experiment`` evaluates one config on one split and writes::

    results/<exp_id>/
      config.yaml      exact resolved config
      git_commit.txt
      metrics.json     aggregate + per query type, each with bootstrap CIs
      per_query.csv    query_id, type, strategy, metrics, latency, costs
      traces.jsonl     PipelineTrace per query (optional)
      failures.json    categorized failures

``exp_id = <phase>_<name>_<timestamp>``. Retrieval is evaluated independently of generation:
retrieval metrics always come from the ranked chunk list, never from the answer.
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from agvrag.chunking.chunkers import chunk_embed_text
from agvrag.config import Settings, dump_settings
from agvrag.evaluation.dataset import EvalQuery, label_coverage, load_eval_set
from agvrag.evaluation.metrics import compute_metrics, path_metrics, satisfied_labels
from agvrag.evaluation.stats import bootstrap_ci
from agvrag.pipeline.budget import utility
from agvrag.pipeline.components import ComponentRegistry
from agvrag.pipeline.pipeline import RAGPipeline
from agvrag.retrieval.bm25 import BM25Tokenizer
from agvrag.schemas import EvidenceType, PipelineTrace
from agvrag.utils.io import ensure_dir, read_json, write_json, write_jsonl
from agvrag.utils.logging import get_logger

log = get_logger(__name__)
_TOK = BM25Tokenizer(stem=True, remove_stopwords=True)


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
        sha = out.stdout.strip() or "uncommitted"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return f"{sha}{' (dirty)' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def evaluate_trace(
    q: EvalQuery, trace: PipelineTrace, pipe: RAGPipeline, ks: list[int], lam: float
) -> dict[str, Any]:
    ranked = trace.ranked_chunk_ids()
    chunks = pipe.index.chunks
    matches = [
        satisfied_labels(chunks[c].doc_id, chunk_embed_text(chunks[c]), q.labels)
        if c in chunks
        else []
        for c in ranked
    ]
    row: dict[str, Any] = {
        "query_id": q.id,
        "type": q.type,
        "split": q.split,
        "answerable": q.answerable,
        "strategy": trace.routing.strategy,
        "intent": trace.query_analysis.intent,
        "n_linked": len(trace.query_analysis.linked_entities),
    }
    if q.answerable and q.labels:
        row.update(compute_metrics(matches, q.labels, ks))
        # Lenient, document-level view: did the right *paper* appear in the top k?
        gt_docs = {lab.doc_id for lab in q.labels}
        for k in ks:
            top_docs = {chunks[c].doc_id for c in ranked[:k] if c in chunks}
            row[f"doc_recall@{k}"] = len(gt_docs & top_docs) / len(gt_docs)
    paths = [e for e in trace.evidence_pool if e.type != EvidenceType.TEXT_CHUNK]
    if q.expected_entities:
        nodes = [set(e.provenance.get("nodes", [])) for e in paths]
        rels = [{x["type"] for x in e.provenance.get("edges", [])} for e in paths]
        row.update(path_metrics(nodes, rels, q.expected_entities, q.expected_relations))
    c = trace.costs
    row["latency_ms"] = round(c.get("latency_ms.total", 0.0), 2)
    row["retrieval_ms"] = round(c.get("latency_ms.retrieval", 0.0), 2)
    row["rerank_ms"] = round(c.get("latency_ms.rerank", 0.0), 2)
    row["cost_units"] = c.get("budget.cost_units", 0.0)
    row["retrieval_calls"] = c.get("budget.retrieval_calls", 0.0)
    row["graph_hops"] = c.get("budget.graph_hops", 0.0)
    row["llm_tokens"] = c.get("llm_input_tokens", 0.0) + c.get("llm_output_tokens", 0.0)
    row["llm_cost_usd"] = c.get("llm_cost_usd", 0.0)
    pk = f"ndcg@{max(k for k in ks if k <= 10)}"
    if pk in row:
        row["utility"] = round(utility(row[pk], row["cost_units"], lam), 6)
    if trace.answer is not None:
        abstained = trace.answer.strip().startswith("INSUFFICIENT_EVIDENCE")
        row["abstained"] = float(abstained)
        row["abstention_correct"] = float(abstained != q.answerable)
        v = trace.verification or {}
        cv = v.get("citation_validation", {})
        row["citation_valid"] = float(bool(cv.get("valid", False)))
        row["citation_coverage"] = cv.get("citation_coverage", 0.0)
        if "support_rate" in v:
            row["support_rate"] = v["support_rate"]
            row["verify_iterations"] = v.get("iterations", 0)
        cited = {cid for cit in trace.citations for cid in cit.get("chunk_ids", [])}
        if q.labels and cited:
            ok = sum(
                1
                for cid in cited
                if cid in chunks
                and satisfied_labels(chunks[cid].doc_id, chunk_embed_text(chunks[cid]), q.labels)
            )
            row["citation_precision"] = round(ok / len(cited), 4)
    return row


def categorize_failure(
    q: EvalQuery, row: dict[str, Any], trace: PipelineTrace, pipe: RAGPipeline, k: int
) -> dict[str, Any] | None:
    if not q.answerable:
        if row.get("abstained") == 0.0:
            return {"query_id": q.id, "category": "answered_unanswerable", "query": q.text}
        return None
    if not q.labels or row.get(f"hit@{k}", 1.0) > 0:
        return None
    chunks = pipe.index.chunks
    top = trace.ranked_chunk_ids()[:k]
    top_docs = {chunks[c].doc_id for c in top if c in chunks}
    gt_docs = {lab.doc_id for lab in q.labels}
    cats: list[str] = []
    if (
        trace.routing.strategy in {"graph", "multi_hop", "graph_vector"}
        and not trace.query_analysis.linked_entities
    ):
        cats.append("entity_not_linked")
    if not top:
        cats.append("empty_result")
    elif gt_docs & top_docs:
        cats.append("right_doc_wrong_passage")
    else:
        cats.append("relevant_doc_not_retrieved")
    # vocabulary mismatch: query terms vs the best relevant chunk in the corpus
    q_terms = set(_TOK.tokenize(q.text))
    best = 0.0
    for c in chunks.values():
        if c.doc_id in gt_docs and satisfied_labels(c.doc_id, chunk_embed_text(c), q.labels):
            terms = set(_TOK.tokenize(chunk_embed_text(c)))
            best = max(best, len(q_terms & terms) / max(1, len(q_terms)))
    if best < 0.4:
        cats.append("vocabulary_mismatch")
    if re.search(r"\b[A-Z][A-Za-z]*[A-Z]+[A-Za-z0-9-]*\b", q.text) and "acronym" in q.tags:
        cats.append("acronym_or_name_variant")
    if q.type == "multi_hop":
        cats.append("multi_hop_chain")
    return {
        "query_id": q.id,
        "type": q.type,
        "query": q.text,
        "strategy": trace.routing.strategy,
        "categories": cats,
        "top_docs": sorted(top_docs)[:5],
        "expected_docs": sorted(gt_docs),
        "best_relevant_term_overlap": round(best, 3),
    }


def aggregate(
    rows: list[dict[str, Any]], metric_keys: list[str], n_boot: int, conf: float, seed: int
) -> dict[str, Any]:
    def agg(rs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(rs)}
        for m in metric_keys:
            vals = [r[m] for r in rs if m in r and r[m] is not None]
            if not vals:
                continue
            mean, lo, hi = bootstrap_ci(vals, n_boot, conf, seed)
            out[m] = {"mean": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        lat = [r["latency_ms"] for r in rs if "latency_ms" in r]
        if lat:
            out["latency_ms_p50"] = round(median(lat), 2)
            out["latency_ms_p95"] = round(float(np.percentile(lat, 95)), 2)
        for c in ("cost_units", "retrieval_calls", "graph_hops", "llm_tokens", "llm_cost_usd"):
            vals = [r[c] for r in rs if c in r]
            if vals:
                out[f"mean_{c}"] = round(fmean(vals), 4)
        return out

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    strategies: dict[str, int] = defaultdict(int)
    for r in rows:
        strategies[r["strategy"]] += 1
    return {
        "overall": agg(rows),
        "per_type": {t: agg(v) for t, v in sorted(by_type.items())},
        "strategy_counts": dict(strategies),
    }


METRIC_KEYS = [
    "mrr",
    "hit@1",
    "hit@3",
    "hit@5",
    "hit@10",
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "precision@1",
    "precision@3",
    "precision@5",
    "precision@10",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    "doc_recall@1",
    "doc_recall@5",
    "doc_recall@10",
    "path_entity_recall",
    "path_relation_recall",
    "path_hit",
    "utility",
    "abstained",
    "abstention_correct",
    "citation_valid",
    "citation_coverage",
    "citation_precision",
    "support_rate",
]


def run_experiment(
    settings: Settings,
    phase: str = "eval",
    split: str | None = None,
    registry: ComponentRegistry | None = None,
    save_traces: bool | None = None,
    strategy_override: str | None = None,
    tag: str = "",
    queries: list[EvalQuery] | None = None,
) -> dict[str, Any]:
    registry = registry or ComponentRegistry()
    ev = settings.evaluation
    split = split or ev.split
    queries = queries if queries is not None else load_eval_set(settings, split)
    pipe = RAGPipeline(settings, registry)
    coverage = label_coverage(queries, pipe.index.chunks)
    name = (
        settings.pipeline.name
        + (f"-{strategy_override}" if strategy_override else "")
        + (f"-{tag}" if tag else "")
    )
    exp_id = f"{phase}_{name}_{split}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')[:-3]}"
    out_dir = ensure_dir(settings.path("results_dir") / exp_id)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for i, q in enumerate(queries, start=1):
        trace = pipe.run(q.text, strategy_override=strategy_override)
        row = evaluate_trace(q, trace, pipe, ev.k_values, settings.budget.lambda_cost)
        rows.append(row)
        f = categorize_failure(q, row, trace, pipe, ev.primary_k)
        if f:
            failures.append(f)
        if save_traces if save_traces is not None else ev.save_traces:
            traces.append({"query_id": q.id, **trace.to_dict()})
        if i % 10 == 0:
            log.info("eval progress", extra={"exp": exp_id, "done": i, "total": len(queries)})
    metrics = aggregate(
        [r for r in rows if r["answerable"]],
        METRIC_KEYS,
        ev.bootstrap_samples,
        ev.confidence,
        settings.seed,
    )
    unans = [r for r in rows if not r["answerable"]]
    if unans:
        metrics["unanswerable"] = aggregate(
            unans,
            ["abstained", "abstention_correct"],
            ev.bootstrap_samples,
            ev.confidence,
            settings.seed,
        )["overall"]
    metrics.update(
        {
            "exp_id": exp_id,
            "pipeline": settings.pipeline.name,
            "split": split,
            "strategy_override": strategy_override,
            "n_queries": len(queries),
            "label_coverage": {k: v for k, v in coverage.items() if k != "unmapped"},
            "index": settings.index_name,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    (out_dir / "config.yaml").write_text(dump_settings(settings), encoding="utf-8")
    (out_dir / "git_commit.txt").write_text(
        git_commit(Path(settings.root)) + "\n", encoding="utf-8"
    )
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "failures.json",
        {
            "label_coverage_unmapped": coverage["unmapped"],
            "failures": failures,
            "category_counts": _count_categories(failures),
        },
    )
    _write_csv(out_dir / "per_query.csv", rows)
    if traces:
        write_jsonl(out_dir / "traces.jsonl", traces)
    log.info("experiment done", extra={"exp_id": exp_id})
    return {"exp_id": exp_id, "dir": str(out_dir), "metrics": metrics, "rows": rows}


def _count_categories(failures: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for f in failures:
        for c in f.get("categories", [f.get("category", "other")]):
            counts[c] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    path.write_text(buf.getvalue(), encoding="utf-8")


def list_experiments(results_dir: Path) -> list[dict[str, Any]]:
    out = []
    for d in sorted(results_dir.glob("*/metrics.json"), reverse=True):
        m = read_json(d)
        o = m.get("overall", {})
        out.append(
            {
                "exp_id": m.get("exp_id", d.parent.name),
                "pipeline": m.get("pipeline"),
                "split": m.get("split"),
                "n_queries": m.get("n_queries"),
                "ndcg@5": (o.get("ndcg@5") or {}).get("mean"),
                "recall@5": (o.get("recall@5") or {}).get("mean"),
                "mrr": (o.get("mrr") or {}).get("mean"),
                "latency_ms_p50": o.get("latency_ms_p50"),
                "created_at": m.get("created_at"),
            }
        )
    return out


def read_per_query(results_dir: Path, exp_id: str) -> list[dict[str, str]]:
    with (results_dir / exp_id / "per_query.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
