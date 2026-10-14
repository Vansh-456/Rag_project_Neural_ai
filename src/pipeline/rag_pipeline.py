"""
Unified composable RAG pipeline.
Executes end-to-end query processing, routing, retrieval, ranking, context construction,
generation, claim verification, and bounded self-correction.
"""

from __future__ import annotations

import time
from typing import Any
from src.schemas import (
    Evidence,
    PipelineTrace,
    QueryAnalysis,
    RetrievalBudget,
    RetrievalStrategy,
    RoutingDecision,
)
from src.config import SystemSettings
from src.embeddings.embedder import BaseEmbedder
from src.vector_store.store import InMemoryVectorStore, QdrantVectorStore
from src.retrieval.sparse import BM25Index
from src.graph.store import NetworkXGraphStore, Neo4jGraphStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.multi_hop import MultiHopRetriever
from src.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from src.routing.router import AdaptiveRouter, QueryAnalyzer
from src.ranking.reranker import CrossEncoderReranker
from src.context.builder import ContextBuilder
from src.generation.client import BaseLLMClient, parse_and_validate_citations
from src.verification.verifier import ClaimVerifier
from src.utils.metrics import timer


class UnifiedRAGPipeline:
    """End-to-end composable RAG pipeline supporting all experimental configurations."""

    def __init__(
        self,
        settings: SystemSettings,
        embedder: BaseEmbedder,
        vector_store: InMemoryVectorStore | QdrantVectorStore,
        sparse_index: BM25Index,
        graph_store: NetworkXGraphStore | Neo4jGraphStore,
        llm_client: BaseLLMClient,
        reranker: CrossEncoderReranker | None = None,
        router: AdaptiveRouter | None = None,
        context_builder: ContextBuilder | None = None,
        verifier: ClaimVerifier | None = None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.graph_store = graph_store
        self.llm_client = llm_client

        # Sub-components
        self.dense_retriever = DenseRetriever(embedder, vector_store)
        self.graph_retriever = GraphRetriever(graph_store)
        self.multi_hop_retriever = MultiHopRetriever(self.dense_retriever, self.graph_retriever)
        self.reranker = reranker or CrossEncoderReranker(model_name=settings.reranking.model_name)
        self.router = router or AdaptiveRouter()
        self.context_builder = context_builder or ContextBuilder(max_tokens=settings.budget.max_token_budget)
        self.verifier = verifier or ClaimVerifier(
            min_supported_ratio=settings.verification.min_supported_claims_ratio,
            contradiction_threshold=settings.verification.contradiction_threshold
        )

    def _execute_retrieval(
        self,
        strategy: str,
        query: str,
        top_k: int,
        budget: RetrievalBudget
    ) -> list[Evidence]:
        """Dispatches retrieval based on chosen strategy."""
        if strategy == RetrievalStrategy.VECTOR.value:
            return self.dense_retriever.retrieve(query, top_k=top_k, budget=budget)

        elif strategy == RetrievalStrategy.BM25.value:
            if not budget.can_call_retrieval():
                return []
            chunks_with_scores = self.sparse_index.search(query, top_k=top_k)
            evidence = []
            for rank, (chunk, score) in enumerate(chunks_with_scores, start=1):
                from src.schemas import EvidenceType
                evidence.append(Evidence(
                    evidence_id=f"ev_bm25_{chunk.chunk_id}",
                    type=EvidenceType.TEXT_CHUNK,
                    content=chunk.text,
                    source_doc_ids=[chunk.doc_id],
                    source_chunk_ids=[chunk.chunk_id],
                    retriever="bm25",
                    raw_score=round(score, 4),
                    rank=rank,
                    provenance={"doc_title": chunk.metadata.get("doc_title", ""), "token_count": chunk.token_count}
                ))
            budget.record_call(candidates=len(evidence))
            return evidence

        elif strategy == RetrievalStrategy.GRAPH.value:
            return self.graph_retriever.retrieve(query, top_k=top_k, budget=budget)

        elif strategy == RetrievalStrategy.MULTI_HOP.value:
            return self.multi_hop_retriever.retrieve(query, top_k=top_k, budget=budget)

        elif strategy == RetrievalStrategy.HYBRID.value:
            dense_ev = self.dense_retriever.retrieve(query, top_k=self.settings.retrieval.dense_top_k, budget=budget)
            sparse_chunks = self.sparse_index.search(query, top_k=self.settings.retrieval.sparse_top_k)
            sparse_ev = []
            for rank, (chunk, score) in enumerate(sparse_chunks, start=1):
                from src.schemas import EvidenceType
                sparse_ev.append(Evidence(
                    evidence_id=f"ev_bm25_{chunk.chunk_id}",
                    type=EvidenceType.TEXT_CHUNK,
                    content=chunk.text,
                    source_doc_ids=[chunk.doc_id],
                    source_chunk_ids=[chunk.chunk_id],
                    retriever="bm25",
                    raw_score=round(score, 4),
                    rank=rank,
                    provenance={"doc_title": chunk.metadata.get("doc_title", ""), "token_count": chunk.token_count}
                ))
            budget.record_call(candidates=len(sparse_ev))

            if self.settings.retrieval.fusion_method == "rrf":
                return reciprocal_rank_fusion([dense_ev, sparse_ev], k=self.settings.retrieval.rrf_k, top_k=top_k)
            else:
                return weighted_score_fusion([
                    (dense_ev, self.settings.retrieval.dense_weight),
                    (sparse_ev, self.settings.retrieval.sparse_weight)
                ], top_k=top_k)

        # Fallback to hybrid
        return self._execute_retrieval(RetrievalStrategy.HYBRID.value, query, top_k, budget)

    def run(
        self,
        query: str,
        pipeline_mode: str | None = None,
        top_k: int = 5,
        budget: RetrievalBudget | None = None
    ) -> PipelineTrace:
        """Executes full RAG workflow with profiling and trace logging."""
        budget = budget or RetrievalBudget(
            max_retrieval_calls=self.settings.budget.max_retrieval_calls,
            max_graph_hops=self.settings.budget.max_graph_hops,
            max_candidate_pool=self.settings.budget.max_candidate_pool,
            max_latency_ms=self.settings.budget.max_latency_ms,
            max_token_budget=self.settings.budget.max_token_budget,
            cost_per_1k_prompt_tokens=self.settings.budget.cost_per_1k_prompt_tokens,
            cost_per_1k_completion_tokens=self.settings.budget.cost_per_1k_completion_tokens
        )

        stage_timings: dict[str, float] = {}

        # 1. Query Analysis & Routing
        with timer() as t:
            analysis = self.router.analyzer.analyze(query)
            if pipeline_mode and pipeline_mode not in ("adaptive", "adaptive_verified"):
                routing = RoutingDecision(
                    strategy=pipeline_mode,
                    params={"override": True},
                    rationale=f"Explicitly forced pipeline strategy: {pipeline_mode}"
                )
            else:
                routing = self.router.route(analysis)
        stage_timings["routing_ms"] = t["elapsed_ms"]

        # 2. Retrieval Stage
        with timer() as t:
            retrieved = self._execute_retrieval(routing.strategy, query, top_k=top_k, budget=budget)
        stage_timings["retrieval_ms"] = t["elapsed_ms"]

        # 3. Reranking Stage
        with timer() as t:
            should_rerank = self.settings.reranking.enabled and (pipeline_mode in ("hybrid_rerank", "adaptive", "adaptive_verified") or routing.params.get("rerank", False))
            if should_rerank and retrieved:
                reranked = self.reranker.rerank(query, retrieved, top_k=top_k)
            else:
                reranked = list(retrieved)
        stage_timings["reranking_ms"] = t["elapsed_ms"]

        # 4. Context Construction
        with timer() as t:
            context, packed_evidence = self.context_builder.build_context(
                evidence=reranked,
                query=query,
                detected_entities=analysis.entities,
                max_tokens=self.settings.budget.max_token_budget
            )
        stage_timings["context_ms"] = t["elapsed_ms"]

        # 5. Generation Stage
        with timer() as t:
            answer, token_counts = self.llm_client.generate(prompt=context)
            budget.record_tokens(token_counts.get("prompt_tokens", 0), token_counts.get("completion_tokens", 0))
        stage_timings["generation_ms"] = t["elapsed_ms"]

        # 6. Citation Extraction & Validation
        citations, all_valid = parse_and_validate_citations(answer, packed_evidence)

        # 7. Verification & Bounded Self-Correction
        verification_dict = None
        should_verify = self.settings.verification.enabled or (pipeline_mode == "adaptive_verified")

        if should_verify:
            with timer() as t:
                v_res = self.verifier.verify(answer, packed_evidence, query)
                verification_dict = {
                    "is_sufficient": v_res.is_sufficient,
                    "supported_ratio": v_res.supported_ratio,
                    "contradiction_found": v_res.contradiction_found,
                    "retry_recommended": v_res.retry_recommended,
                    "claims": [
                        {
                            "claim": c.claim,
                            "status": c.status.value,
                            "cited_evidence_ids": c.cited_evidence_ids,
                            "reasoning": c.reasoning,
                            "confidence": c.confidence
                        }
                        for c in v_res.claims
                    ]
                }

                # Trigger retry if recommended and budget permits
                if v_res.retry_recommended and budget.can_call_retrieval():
                    retry_query = v_res.reformulated_query or f"{query} details"
                    extra_evidence = self._execute_retrieval(RetrievalStrategy.HYBRID.value, retry_query, top_k=top_k, budget=budget)
                    if extra_evidence:
                        combined = reranked + extra_evidence
                        context, packed_evidence = self.context_builder.build_context(combined, query, analysis.entities)
                        answer, tokens_retry = self.llm_client.generate(prompt=context)
                        budget.record_tokens(tokens_retry.get("prompt_tokens", 0), tokens_retry.get("completion_tokens", 0))
                        citations, _ = parse_and_validate_citations(answer, packed_evidence)
                        v_res_2 = self.verifier.verify(answer, packed_evidence, query)
                        verification_dict["is_sufficient"] = v_res_2.is_sufficient
                        verification_dict["retried"] = True
                        verification_dict["retry_query"] = retry_query

            stage_timings["verification_ms"] = t["elapsed_ms"]

        total_latency = sum(stage_timings.values())
        budget.elapsed_ms = total_latency

        costs = {
            "timings_ms": stage_timings,
            "total_latency_ms": round(total_latency, 2),
            "retrieval_calls_used": budget.calls_used,
            "graph_hops_used": budget.hops_used,
            "prompt_tokens": budget.prompt_tokens_used,
            "completion_tokens": budget.completion_tokens_used,
            "estimated_cost_usd": budget.total_cost_usd()
        }

        return PipelineTrace(
            query_analysis=analysis,
            routing=routing,
            retrieved=retrieved,
            reranked=reranked,
            context=context,
            answer=answer,
            citations=citations,
            verification=verification_dict,
            costs=costs
        )
