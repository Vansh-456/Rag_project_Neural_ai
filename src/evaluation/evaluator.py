"""
Evaluation harness and benchmark runner.
Executes test queries, records per-query metrics, runs bootstrap CIs,
and classifies failure modes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import numpy as np
from src.schemas import PipelineTrace
from src.evaluation.metrics import (
    bootstrap_confidence_interval,
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from src.pipeline.rag_pipeline import UnifiedRAGPipeline


class BenchmarkEvaluator:
    """Runs automated evaluation and outputs ablation summaries."""

    def __init__(self, pipeline: UnifiedRAGPipeline, results_dir: str = "results"):
        self.pipeline = pipeline
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_queries(
        self,
        queries_path: str = "data/evaluation/queries.json",
        ground_truth_path: str = "data/evaluation/ground_truth.json",
        pipeline_mode: str = "adaptive",
        top_k: int = 5,
        experiment_id: str = "eval_run"
    ) -> dict[str, Any]:
        """Runs evaluation over dataset and saves artifacts."""
        with open(queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)

        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        per_query_records: list[dict[str, Any]] = []
        recalls, precisions, mrrs, ndcgs, hit_rates, latencies = [], [], [], [], [], []

        # Per query type buckets
        by_type: dict[str, dict[str, list[float]]] = {}
        failures: list[dict[str, Any]] = []

        for q_item in queries:
            qid = q_item["id"]
            q_text = q_item["text"]
            q_type = q_item.get("type", "definition")
            gt = ground_truth.get(qid, {})

            # Execute pipeline
            trace: PipelineTrace = self.pipeline.run(query=q_text, pipeline_mode=pipeline_mode, top_k=top_k)

            # Map retrieved evidence to doc IDs
            retrieved_doc_ids = []
            for ev in trace.retrieved:
                retrieved_doc_ids.extend(ev.source_doc_ids)

            # Calculate metrics
            rec = compute_recall_at_k(retrieved_doc_ids, gt, k=top_k)
            prec = compute_precision_at_k(retrieved_doc_ids, gt, k=top_k)
            mrr = compute_mrr(retrieved_doc_ids, gt)
            ndcg = compute_ndcg_at_k(retrieved_doc_ids, gt, k=top_k)
            hit = compute_hit_rate_at_k(retrieved_doc_ids, gt, k=top_k)
            lat = trace.costs.get("total_latency_ms", 0.0)

            recalls.append(rec)
            precisions.append(prec)
            mrrs.append(mrr)
            ndcgs.append(ndcg)
            hit_rates.append(hit)
            latencies.append(lat)

            if q_type not in by_type:
                by_type[q_type] = {"recalls": [], "mrrs": [], "ndcgs": []}
            by_type[q_type]["recalls"].append(rec)
            by_type[q_type]["mrrs"].append(mrr)
            by_type[q_type]["ndcgs"].append(ndcg)

            # Record failure case if Recall is 0
            if rec == 0.0:
                failure_cat = "VOCABULARY_MISMATCH"
                if q_type == "multi_hop":
                    failure_cat = "MULTI_HOP_CHAIN_BROKEN"
                elif q_type == "relationship":
                    failure_cat = "ENTITY_UNRESOLVED"

                failures.append({
                    "query_id": qid,
                    "query": q_text,
                    "type": q_type,
                    "category": failure_cat,
                    "strategy_used": trace.routing.strategy,
                    "retrieved_count": len(trace.retrieved)
                })

            per_query_records.append({
                "query_id": qid,
                "query_type": q_type,
                "strategy": trace.routing.strategy,
                "recall@k": rec,
                "precision@k": prec,
                "mrr": mrr,
                "ndcg@k": ndcg,
                "hit_rate@k": hit,
                "latency_ms": lat
            })

        # Compute aggregate bootstrap CIs
        r_mean, r_low, r_high = bootstrap_confidence_interval(recalls)
        p_mean, p_low, p_high = bootstrap_confidence_interval(precisions)
        m_mean, m_low, m_high = bootstrap_confidence_interval(mrrs)
        n_mean, n_low, n_high = bootstrap_confidence_interval(ndcgs)

        p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

        # Type breakdown summary
        type_breakdown = {}
        for t_name, t_metrics in by_type.items():
            type_breakdown[t_name] = {
                "count": len(t_metrics["recalls"]),
                "mean_recall": round(float(np.mean(t_metrics["recalls"])), 4),
                "mean_mrr": round(float(np.mean(t_metrics["mrrs"])), 4),
                "mean_ndcg": round(float(np.mean(t_metrics["ndcgs"])), 4),
            }

        summary = {
            "experiment_id": experiment_id,
            "pipeline_mode": pipeline_mode,
            "query_count": len(queries),
            "metrics": {
                "recall@k": {"mean": r_mean, "ci_95": [r_low, r_high]},
                "precision@k": {"mean": p_mean, "ci_95": [p_low, p_high]},
                "mrr": {"mean": m_mean, "ci_95": [m_low, m_high]},
                "ndcg@k": {"mean": n_mean, "ci_95": [n_low, n_high]},
                "hit_rate@k": round(float(np.mean(hit_rates)), 4) if hit_rates else 0.0,
                "latency_ms": {"p50": round(p50_lat, 2), "p95": round(p95_lat, 2)}
            },
            "per_query_type": type_breakdown,
            "total_failures": len(failures)
        }

        # Write output directory
        exp_dir = self.results_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with open(exp_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

        if per_query_records:
            keys = per_query_records[0].keys()
            with open(exp_dir / "per_query.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(per_query_records)

        return summary
