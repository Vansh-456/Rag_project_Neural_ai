"""
Comprehensive evaluation benchmark script.
Evaluates all systems (Vector, BM25, Hybrid, Hybrid+Rerank, Graph, Multi-Hop, Adaptive, Adaptive+Verified)
over the 50-query test set, computes bootstrap confidence intervals, and outputs the ablation report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.api.app import init_system
from src.evaluation.evaluator import BenchmarkEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation over RAG pipelines.")
    parser.add_argument("--pipeline", type=str, default="all", help="Pipeline to evaluate (or 'all').")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K cutoff for retrieval metrics.")
    args = parser.parse_args()

    pipeline = init_system()
    evaluator = BenchmarkEvaluator(pipeline)

    systems = [
        "vector",
        "bm25",
        "hybrid",
        "hybrid_rerank",
        "graph",
        "multi_hop",
        "adaptive",
        "adaptive_verified"
    ] if args.pipeline == "all" else [args.pipeline]

    print("\n" + "=" * 80)
    print("🚀 RUNNING COMPREHENSIVE BENCHMARK EVALUATION (50 QUERIES, 5 TAXONOMY TYPES)")
    print("=" * 80)

    ablation_summary = []

    for sys_name in systems:
        print(f"\nEvaluating system: [{sys_name.upper()}] ...")
        res = evaluator.evaluate_queries(
            pipeline_mode=sys_name,
            top_k=args.top_k,
            experiment_id=f"benchmark_{sys_name}"
        )
        m = res["metrics"]
        rec = m["recall@k"]["mean"]
        mrr = m["mrr"]["mean"]
        ndcg = m["ndcg@k"]["mean"]
        p95 = m["latency_ms"]["p95"]

        ablation_summary.append({
            "system": sys_name,
            "recall@5": rec,
            "recall_ci": m["recall@k"]["ci_95"],
            "mrr": mrr,
            "ndcg@5": ndcg,
            "hit_rate": m["hit_rate@k"],
            "p95_latency_ms": p95,
            "per_query_type": res["per_query_type"]
        })
        print(f"  -> Recall@{args.top_k}: {rec:.4f} (95% CI: {m['recall@k']['ci_95']}) | MRR: {mrr:.4f} | NDCG@{args.top_k}: {ndcg:.4f} | p95 Latency: {p95:.1f}ms")

    # Print markdown ablation table
    print("\n" + "=" * 80)
    print("📊 ABLATION BENCHMARK RESULTS TABLE")
    print("=" * 80)
    print("| System | Recall@5 | 95% CI | MRR | NDCG@5 | Hit-Rate | p95 Latency (ms) |")
    print("|---|---|---|---|---|---|---|")
    for r in ablation_summary:
        print(f"| {r['system']} | {r['recall@5']:.4f} | [{r['recall_ci'][0]:.3f}, {r['recall_ci'][1]:.3f}] | {r['mrr']:.4f} | {r['ndcg@5']:.4f} | {r['hit_rate']:.4f} | {r['p95_latency_ms']:.1f} |")
    print("=" * 80 + "\n")

    # Save summary file
    results_path = Path("results/ablation_benchmark_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)
    print(f"Summary saved to {results_path}")


if __name__ == "__main__":
    main()
