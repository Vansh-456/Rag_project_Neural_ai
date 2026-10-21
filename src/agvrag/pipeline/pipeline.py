"""The configurable RAG pipeline.

    query -> analyze -> route -> retrieve -> [rerank] -> context -> [generate] -> [verify]
                                   ^                                                |
                                   +---- reformulate / escalate (bounded) <---------+

Every pipeline variant (vector-only ... adaptive+verification) is this class with a
different ``Settings``. Each run returns a ``PipelineTrace`` holding every intermediate list,
per-stage latencies, token/cost accounting, and each retry iteration.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agvrag.config import Settings
from agvrag.context.builder import StructuredContextBuilder
from agvrag.generation.generator import ExtractiveGenerator, GenerationResult, LLMGenerator
from agvrag.interfaces import BuiltContext
from agvrag.pipeline.budget import BudgetExceeded, BudgetTracker
from agvrag.pipeline.components import ComponentRegistry
from agvrag.retrieval.fusion import rrf_fuse
from agvrag.retrieval.retrievers import (
    DenseRetriever,
    GraphRetriever,
    GraphVectorRetriever,
    HybridRetriever,
    MultiHopRetriever,
    SparseRetriever,
)
from agvrag.routing.router import FixedRouter, RuleQueryAnalyzer, RuleRouter
from agvrag.schemas import Evidence, EvidenceType, PipelineTrace, QueryAnalysis, RoutingDecision
from agvrag.utils.ids import new_trace_id
from agvrag.utils.logging import get_logger
from agvrag.utils.timing import StageTimer

log = get_logger(__name__)

GRAPH_STRATEGIES = {"graph", "graph_vector", "multi_hop"}


@dataclass
class RetrievalOutcome:
    strategy: str
    retrieved: list[Evidence]
    lists: dict[str, list[Evidence]]
    final: list[Evidence]
    reranked: bool


class RAGPipeline:
    def __init__(self, settings: Settings, registry: ComponentRegistry | None = None) -> None:
        self.s = settings
        self.registry = registry or ComponentRegistry()
        self.name = settings.pipeline.name
        p = settings.pipeline
        self.uses_graph = p.router == "adaptive" or p.strategy in GRAPH_STRATEGIES or p.use_budget
        self.index = self.registry.index(settings)
        self.embedder = self.registry.embedder(settings)
        self.graph = self.registry.graph(settings) if self.uses_graph else None
        self.linker = self.registry.linker(settings) if self.uses_graph else None
        self.gazetteer = self.registry.gazetteer(settings) if self.uses_graph else None
        self.analyzer = RuleQueryAnalyzer(settings.router, self.linker)
        self.router = (
            RuleRouter(settings.router, settings.retrieval)
            if p.router == "adaptive"
            else FixedRouter(p.strategy, settings.retrieval)
        )
        self.needs_reranker = p.rerank or p.router == "adaptive"
        self.reranker = (
            self.registry.reranker(settings)
            if (self.needs_reranker and settings.reranker.enabled)
            else None
        )
        self.context_builder = StructuredContextBuilder(settings.context)
        self._generator: Any = None
        if p.generate or p.verify:
            self._generator = self.generator  # fail fast on misconfigured providers
        self._verifier: Any = None
        if p.verify:
            self._verifier = self.verifier

    @property
    def verifier(self) -> Any:
        """Built on first use, so any pipeline can verify on request (``run(verify=True)``)."""
        if self._verifier is None:
            from agvrag.verification.verifier import ClaimVerifier

            v = self.s.verification
            v_llm = self.registry.llm(self.s) if v.method == "llm" else None
            self._verifier = ClaimVerifier(v, v_llm)
        return self._verifier

    @property
    def generator(self) -> Any:
        """Built on first use, so retrieval-only pipelines can still answer on request."""
        if self._generator is None:
            g = self.s.generation
            llm = self.registry.llm(self.s)
            self._generator = (
                LLMGenerator(llm, g.prompt_version, g.max_tokens, g.temperature)
                if llm is not None
                else ExtractiveGenerator()
            )
        return self._generator

    # ------------------------------------------------------------------ retrieval
    def _retriever(self, strategy: str, budget: BudgetTracker) -> Any:
        cost = budget.charge
        dense = DenseRetriever(self.embedder, self.index.vector_store, cost)
        sparse = SparseRetriever(self.index.bm25, cost)
        if strategy == "vector":
            return dense
        if strategy == "bm25":
            return sparse
        if strategy == "hybrid":
            return HybridRetriever(dense, sparse, self.s.retrieval)
        if self.graph is None or self.linker is None:
            raise ValueError(
                f"strategy '{strategy}' needs the graph; pipeline was built without it"
            )
        graph = GraphRetriever(self.graph, self.linker, self.index.chunks, self.s.retrieval, cost)
        if strategy == "graph":
            return graph
        if strategy == "graph_vector":
            return GraphVectorRetriever(dense, sparse, graph, self.gazetteer, self.s.retrieval)
        if strategy == "multi_hop":
            return MultiHopRetriever(dense, graph, self.s.retrieval)
        raise ValueError(f"unknown strategy {strategy}")

    def _retrieve(
        self,
        query: str,
        analysis: QueryAnalysis,
        routing: RoutingDecision,
        budget: BudgetTracker,
        timer: StageTimer,
        strategy: str | None = None,
        k_override: int | None = None,
    ) -> RetrievalOutcome:
        strategy = strategy or routing.strategy
        k = k_override or int(routing.params.get("k", self.s.retrieval.top_k))
        do_rerank = self.reranker is not None and (
            self.s.pipeline.rerank or bool(routing.params.get("rerank"))
        )
        pool = (
            int(routing.params.get("candidate_pool", self.s.reranker.candidate_pool))
            if do_rerank
            else k
        )
        retriever = self._retriever(strategy, budget)
        with timer.stage("retrieval"):
            retrieved = retriever.retrieve(query, analysis, pool)
        lists = dict(getattr(retriever, "last_lists", {}) or {strategy: retrieved})
        if (
            do_rerank
            and self.reranker is not None
            and budget.can_afford("rerank_candidate", len(retrieved))
        ):
            budget.charge("rerank_candidate", float(len(retrieved)))
            with timer.stage("rerank"):
                final = self.reranker.rerank(query, retrieved, k)
            return RetrievalOutcome(strategy, retrieved, lists, final, True)
        return RetrievalOutcome(strategy, retrieved, lists, _top_k_chunks(retrieved, k), False)

    @staticmethod
    def _confidence(outcome: RetrievalOutcome) -> float:
        """Retrieval confidence used by the budget policy: best rerank score (sigmoid) or the
        best dense cosine in the outcome's lists."""
        import math

        if outcome.reranked and outcome.final and outcome.final[0].rerank_score is not None:
            return 1 / (1 + math.exp(-outcome.final[0].rerank_score))
        best = 0.0
        for name, evs in outcome.lists.items():
            if name.startswith("dense") and evs:
                best = max(best, max(e.raw_score for e in evs))
        if outcome.strategy == "graph" and outcome.final:
            best = max(best, min(1.0, outcome.final[0].raw_score))
        return best

    # ------------------------------------------------------------------ main entry
    def run(
        self,
        query: str,
        generate: bool | None = None,
        verify: bool | None = None,
        strategy_override: str | None = None,
    ) -> PipelineTrace:
        p = self.s.pipeline
        generate = p.generate if generate is None else generate
        verify = p.verify if verify is None else verify
        timer = StageTimer()
        budget = BudgetTracker(self.s.budget, enforce=p.use_budget)
        with timer.stage("analysis"):
            analysis = self.analyzer.analyze(query)
        with timer.stage("routing"):
            routing = self.router.route(analysis)
            if strategy_override:
                # pure strategy (used by the oracle): no router-requested reranking
                routing = RoutingDecision(
                    strategy_override,
                    {**routing.params, "rerank": False},
                    f"override -> {strategy_override}",
                )
            if routing.strategy in GRAPH_STRATEGIES and self.graph is None:
                routing = RoutingDecision(
                    "hybrid", routing.params, routing.rationale + " (no graph: hybrid)"
                )
        iterations: list[dict[str, Any]] = []

        if p.use_budget and p.router == "adaptive" and not strategy_override:
            outcome = self._budgeted_retrieve(query, analysis, routing, budget, timer, iterations)
        else:
            outcome = self._retrieve(query, analysis, routing, budget, timer)

        ctx: BuiltContext | None = None
        gen: GenerationResult | None = None
        verification: dict[str, Any] | None = None
        if generate or verify:
            with timer.stage("context"):
                ctx = self.context_builder.build(query, analysis, outcome.final)
            with timer.stage("generation"):
                gen = self.generator.generate(query, analysis, ctx)
            self._charge_llm(budget, gen)
            if verify:
                verification, outcome, ctx, gen = self._verify_loop(
                    query, analysis, routing, outcome, ctx, gen, budget, timer, iterations
                )

        costs = timer.as_costs()
        costs.update({f"budget.{k}": v for k, v in budget.summary().items()})
        costs["candidates_retrieved"] = float(len(outcome.retrieved))
        if gen is not None:
            costs.update({k: float(v) for k, v in gen.usage.items()})
        if ctx is not None:
            costs["context_tokens"] = float(ctx.token_count)
        return PipelineTrace(
            trace_id=new_trace_id(),
            pipeline=self.name,
            query_analysis=analysis,
            routing=RoutingDecision(outcome.strategy, routing.params, routing.rationale),
            retrieved=outcome.retrieved,
            reranked=outcome.final if outcome.reranked else [],
            context=ctx.text if ctx else "",
            answer=gen.answer if gen else None,
            citations=gen.citations if gen else [],
            verification=(
                {**(verification or {}), "citation_validation": gen.validation}
                if gen
                else verification
            ),
            costs=costs,
            retrieval_lists=outcome.lists,
            evidence_pool=outcome.final,
            iterations=iterations,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _charge_llm(self, budget: BudgetTracker, gen: GenerationResult) -> None:
        tokens = gen.usage.get("llm_input_tokens", 0) + gen.usage.get("llm_output_tokens", 0)
        if tokens:
            try:
                budget.charge("llm_tokens", float(tokens))
            except BudgetExceeded as exc:
                log.warning("token budget exceeded after generation", extra={"limit": exc.limit})

    def _budgeted_retrieve(
        self,
        query: str,
        analysis: QueryAnalysis,
        routing: RoutingDecision,
        budget: BudgetTracker,
        timer: StageTimer,
        iterations: list[dict[str, Any]],
    ) -> RetrievalOutcome:
        """Cost-aware policy: start at the cheapest ladder step allowed by the router decision,
        escalate while confidence < threshold and budget remains; keep the best outcome."""
        ladder = self.s.budget.escalation_ladder
        start = ladder.index(routing.strategy) if routing.strategy in ladder else 0
        # Never start *above* what the router asked for, but allow starting cheaper for
        # simple queries.
        start = 0 if analysis.intent in {"definition", "experimental"} else start
        best: RetrievalOutcome | None = None
        for step in ladder[start:]:
            if step in GRAPH_STRATEGIES and self.graph is None:
                continue
            try:
                outcome = self._retrieve(query, analysis, routing, budget, timer, strategy=step)
            except BudgetExceeded as exc:
                iterations.append({"step": step, "stopped": f"budget: {exc.limit}"})
                break
            conf = self._confidence(outcome)
            iterations.append(
                {
                    "step": step,
                    "confidence": round(conf, 4),
                    "cost_units": round(budget.cost_units, 4),
                }
            )
            if best is None or conf > self._confidence(best):
                best = outcome
            if conf >= self.s.budget.min_confidence:
                break
        if best is None:
            best = self._retrieve(
                query, analysis, routing, BudgetTracker(self.s.budget, enforce=False), timer
            )
        return best

    def _verify_loop(
        self,
        query: str,
        analysis: QueryAnalysis,
        routing: RoutingDecision,
        outcome: RetrievalOutcome,
        ctx: BuiltContext,
        gen: GenerationResult,
        budget: BudgetTracker,
        timer: StageTimer,
        iterations: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], RetrievalOutcome, BuiltContext, GenerationResult]:
        ladder = self.s.budget.escalation_ladder
        with timer.stage("verification"):
            verification = self.verifier.verify(gen.answer, gen.citations, ctx)
        iterations.append(
            {
                "iteration": 0,
                "strategy": outcome.strategy,
                "support_rate": verification["support_rate"],
                "sufficient": verification["sufficient"],
            }
        )
        k = int(routing.params.get("k", self.s.retrieval.top_k))
        for it in range(1, self.s.verification.max_iterations + 1):
            if verification["sufficient"]:
                break
            nxt = _next_strategy(outcome.strategy, ladder, self.graph is not None)
            if nxt is None and it > 1:
                break
            strategy = nxt or outcome.strategy
            unsupported = [c["claim"] for c in verification["claims"] if c["label"] != "supported"]
            reform = _reformulate(query, analysis, unsupported)
            k = int(k * 1.5)
            try:
                new = self._retrieve(
                    reform, analysis, routing, budget, timer, strategy=strategy, k_override=k
                )
            except BudgetExceeded as exc:
                iterations.append({"iteration": it, "stopped": f"budget: {exc.limit}"})
                break
            merged = rrf_fuse(
                {"previous": outcome.final, "retry": new.final}, self.s.retrieval.rrf_k, None, k
            )
            outcome = RetrievalOutcome(
                strategy,
                outcome.retrieved + new.retrieved,
                {**outcome.lists, **{f"retry{it}:{n}": v for n, v in new.lists.items()}},
                merged,
                outcome.reranked or new.reranked,
            )
            with timer.stage("context"):
                ctx = self.context_builder.build(query, analysis, outcome.final)
            with timer.stage("generation"):
                gen = self.generator.generate(query, analysis, ctx)
            self._charge_llm(budget, gen)
            with timer.stage("verification"):
                verification = self.verifier.verify(gen.answer, gen.citations, ctx)
            iterations.append(
                {
                    "iteration": it,
                    "strategy": strategy,
                    "reformulated_query": reform,
                    "k": k,
                    "support_rate": verification["support_rate"],
                    "sufficient": verification["sufficient"],
                }
            )
        verification["iterations"] = len(iterations) - 1
        return verification, outcome, ctx, gen

    # ------------------------------------------------------------------ streaming (API)
    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        """Yield stage events: analysis, routing, retrieval, then answer tokens, then done."""
        timer = StageTimer()
        budget = BudgetTracker(self.s.budget, enforce=self.s.pipeline.use_budget)
        analysis = self.analyzer.analyze(query)
        yield {"event": "analysis", "data": analysis.to_dict()}
        routing = self.router.route(analysis)
        if routing.strategy in GRAPH_STRATEGIES and self.graph is None:
            routing = RoutingDecision("hybrid", routing.params, routing.rationale + " (no graph)")
        yield {"event": "routing", "data": routing.to_dict()}
        outcome = self._retrieve(query, analysis, routing, budget, timer)
        yield {"event": "evidence", "data": [e.to_dict() for e in outcome.final]}
        ctx = self.context_builder.build(query, analysis, outcome.final)
        gen = self.generator
        answer = ""
        for piece in gen.stream(query, analysis, ctx):
            answer += piece
            yield {"event": "token", "data": piece}
        from agvrag.generation.generator import validate_citations

        citations, validation = validate_citations(answer, ctx)
        yield {
            "event": "done",
            "data": {"citations": citations, "validation": validation, "costs": timer.as_costs()},
        }


def _top_k_chunks(evidence: list[Evidence], k: int) -> list[Evidence]:
    """Keep the first k text chunks; graph paths travel along without counting toward k."""
    out: list[Evidence] = []
    n = 0
    for ev in evidence:
        if ev.type == EvidenceType.TEXT_CHUNK:
            if n >= k:
                continue
            n += 1
        out.append(ev)
    return out


def _next_strategy(current: str, ladder: list[str], has_graph: bool) -> str | None:
    if current not in ladder:
        return ladder[1] if len(ladder) > 1 else None
    for s in ladder[ladder.index(current) + 1 :]:
        if s in GRAPH_STRATEGIES and not has_graph:
            continue
        return s
    return None


def _reformulate(query: str, analysis: QueryAnalysis, unsupported_claims: list[str]) -> str:
    """Query expansion for retries: add linked-entity names/aliases and salient terms from
    claims the verifier could not support (they name what evidence is missing)."""
    extra: list[str] = []
    for e in analysis.linked_entities:
        if e["name"].lower() not in query.lower():
            extra.append(e["name"])
    for c in unsupported_claims[:2]:
        words = [w for w in c.split() if len(w) > 5 and w.lower() not in query.lower()]
        extra.extend(words[:4])
    return f"{query} {' '.join(dict.fromkeys(extra))}".strip()
