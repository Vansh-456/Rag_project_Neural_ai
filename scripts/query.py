"""
Interactive command-line query tool.
Runs a query through any pipeline mode, displays router decisions, retrieved evidence,
grounded answer, citations, and verification status.
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.api.app import init_system


def main():
    parser = argparse.ArgumentParser(description="Query the Adaptive Graph + Vector RAG system.")
    parser.add_argument("query", type=str, help="Question or search query.")
    parser.add_argument("--pipeline", type=str, default="adaptive", choices=["vector", "bm25", "hybrid", "graph", "multi_hop", "adaptive", "adaptive_verified"], help="Pipeline execution strategy.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of evidence items to retrieve.")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"[*] EXECUTING QUERY: '{args.query}' (Pipeline: {args.pipeline.upper()})")
    print("=" * 70)

    pipeline = init_system()
    trace = pipeline.run(query=args.query, pipeline_mode=args.pipeline, top_k=args.top_k)

    print("\n[1] QUERY UNDERSTANDING & ROUTING")
    print(f"  - Intent: {trace.query_analysis.intent}")
    print(f"  - Entities: {trace.query_analysis.entities or 'None'}")
    print(f"  - Complexity: {trace.query_analysis.complexity}")
    print(f"  - Strategy Chosen: {trace.routing.strategy.upper()}")
    print(f"  - Router Rationale: {trace.routing.rationale}")

    print(f"\n[2] RETRIEVED EVIDENCE ({len(trace.reranked)} items)")
    for ev in trace.reranked:
        print(f"  [{ev.evidence_id}] (Retriever: {ev.retriever.upper()} | Score: {ev.raw_score} | Rerank: {ev.rerank_score or 'N/A'})")
        print(f"     Title: {ev.provenance.get('doc_title', 'Unknown')} | Page: {ev.provenance.get('page_start', 1)}")
        snippet = ev.content[:160].replace("\n", " ")
        print(f"     Preview: {snippet}...")

    print("\n[3] GROUNDED ANSWER")
    print(f"  {trace.answer}")

    print("\n[4] CITATIONS")
    if trace.citations:
        for c in trace.citations:
            valid_str = "VALID" if c.get("is_valid") else "INVALID"
            print(f"  - [{valid_str}] {c.get('evidence_id')} -> {c.get('doc_title', '')} (Chunk: {c.get('target_chunk_id', '')})")
    else:
        print("  - No citations emitted.")

    if trace.verification:
        print("\n[5] CLAIM VERIFICATION & QUALITY")
        print(f"  - Answer Sufficient: {trace.verification.get('is_sufficient')}")
        print(f"  - Supported Ratio: {trace.verification.get('supported_ratio') * 100:.1f}%")
        print(f"  - Contradictions Detected: {trace.verification.get('contradiction_found')}")
        for cl in trace.verification.get("claims", []):
            print(f"    - [{cl.get('status').upper()}] \"{cl.get('claim')}\" (Confidence: {cl.get('confidence')})")

    print("\n[6] SYSTEM PROFILE & COST")
    timings = trace.costs.get("timings_ms", {})
    print(f"  - Total Latency: {trace.costs.get('total_latency_ms')} ms")
    for stage, t in timings.items():
        print(f"      - {stage}: {t} ms")
    print(f"  - Retrieval Calls: {trace.costs.get('retrieval_calls_used')}")
    print(f"  - Graph Hops: {trace.costs.get('graph_hops_used')}")
    print(f"  - Estimated Cost: ${trace.costs.get('estimated_cost_usd', 0.0):.6f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
