"""Query understanding + adaptive routing (rule-based, every rule tunable via YAML).

``RuleQueryAnalyzer`` produces a ``QueryAnalysis``: intent (definition | relationship |
comparison | multi_hop | experimental), linked graph entities, relation cues, a complexity
score in [0, 1] and an estimated hop count.

``RuleRouter`` maps the analysis to a ``RoutingDecision`` (§8 of the spec):
  definition   & <= 1 entity            -> vector
  relationship & >= 1 entity            -> graph   (falls back to hybrid with no entity)
  comparison   & >= 2 entities          -> graph_vector (hybrid + graph)
  multi_hop    (relation cues, >=2 hops) -> multi_hop (graph -> vector)
  experimental                          -> vector + rerank, larger candidate pool
  otherwise                             -> fallback (hybrid)

The intent -> strategy table (``router.intent_strategy``) and which intents get reranked
(``router.rerank_intents``) are YAML-configurable; ``configs/experiments/adaptive_tuned.yaml``
holds the table re-fit on the dev split (see DECISIONS.md ADR-015).

``FixedRouter`` always returns one strategy (used by non-adaptive pipelines).
Alternative routers (LLM, learned) implement the same ``route`` method.
"""

from __future__ import annotations

import re
from typing import Any

from agvrag.config import RetrievalConfig, RouterConfig
from agvrag.graph.linking import EntityLinker
from agvrag.graph.traversal import relation_cues_for
from agvrag.schemas import QueryAnalysis, RoutingDecision


def _has_cue(q: str, cues: list[str]) -> list[str]:
    return [c for c in cues if c in q]


class RuleQueryAnalyzer:
    def __init__(self, cfg: RouterConfig, linker: EntityLinker | None = None) -> None:
        self.cfg = cfg
        self.linker = linker

    def analyze(self, query: str) -> QueryAnalysis:
        q = f" {query.lower().strip()} "
        linked = self.linker.link(query) if self.linker else []
        n_ent = len(linked)
        rel_cues = relation_cues_for(query)
        hits = {
            "definition": _has_cue(q, self.cfg.definition_cues),
            "relationship": _has_cue(q, self.cfg.relationship_cues),
            "comparison": _has_cue(q, self.cfg.comparison_cues),
            "multi_hop": _has_cue(q, self.cfg.multi_hop_cues),
            "experimental": _has_cue(q, self.cfg.experimental_cues),
        }
        # "which ... that ..." relative-clause chains are the strongest multi-hop signal
        chain = bool(
            re.search(r"\b(which|what)\b.+\b(that|whose|which)\b.+\b(by|on|from|to|over)\b", q)
        )
        reasons: list[str] = []
        if (hits["multi_hop"] or chain) and (len(rel_cues) >= 2 or chain):
            intent = "multi_hop"
            reasons.append(
                f"multi-hop cues {hits['multi_hop'] or ['relative-clause chain']} "
                f"with relation cues {rel_cues}"
            )
        elif hits["comparison"]:
            intent = "comparison"
            reasons.append(f"comparison cues {hits['comparison']}")
        elif hits["relationship"]:
            intent = "relationship"
            reasons.append(f"relationship cues {hits['relationship']}")
        elif hits["experimental"]:
            intent = "experimental"
            reasons.append(f"experimental cues {hits['experimental']}")
        elif hits["definition"]:
            intent = "definition"
            reasons.append(f"definition cues {hits['definition']}")
        elif hits["multi_hop"]:
            intent = "multi_hop"
            reasons.append(f"multi-hop cues {hits['multi_hop']}")
        else:
            intent = "definition" if n_ent == 1 and len(q.split()) <= 6 else "unknown"
            reasons.append("no explicit cue")
        hops = {
            "definition": 0,
            "experimental": 0,
            "relationship": 1,
            "comparison": 1,
            "multi_hop": 2,
            "unknown": 1,
        }[intent]
        if intent in {"relationship", "multi_hop"} and len(rel_cues) >= 2:
            hops = max(hops, 2)
        words = len(q.split())
        complexity = min(
            1.0,
            round(
                0.15 * n_ent
                + 0.1 * len(rel_cues)
                + 0.01 * words
                + (0.3 if intent == "multi_hop" else 0.0)
                + (0.15 if intent == "comparison" else 0.0),
                3,
            ),
        )
        reasons.append(f"{n_ent} linked entities: {[e['name'] for e in linked]}")
        return QueryAnalysis(
            query=query,
            intent=intent,
            entities=[e["name"] for e in linked],
            complexity=complexity,
            hops_estimate=hops,
            rationale="; ".join(reasons),
            linked_entities=linked,
            relation_cues=rel_cues,
        )


class FixedRouter:
    def __init__(self, strategy: str, rcfg: RetrievalConfig) -> None:
        self.strategy = strategy
        self.rcfg = rcfg

    def route(self, analysis: QueryAnalysis) -> RoutingDecision:
        return RoutingDecision(
            strategy=self.strategy,
            params={
                "k": self.rcfg.top_k,
                "candidate_pool": self.rcfg.candidate_pool,
                "hops": self.rcfg.graph.max_hops,
            },
            rationale=f"fixed pipeline: always '{self.strategy}'",
        )


GRAPH_REQUIRING = {"graph", "multi_hop"}


class RuleRouter:
    def __init__(self, cfg: RouterConfig, rcfg: RetrievalConfig) -> None:
        self.cfg = cfg
        self.rcfg = rcfg

    def route(self, analysis: QueryAnalysis) -> RoutingDecision:
        n = len(analysis.linked_entities)
        intent = analysis.intent
        params: dict[str, Any] = {
            "k": self.rcfg.top_k,
            "candidate_pool": self.rcfg.candidate_pool,
            "hops": max(1, analysis.hops_estimate),
            "rerank": False,
        }
        table = self.cfg.intent_strategy
        if intent == "definition" and n <= 1:
            strategy, why = table["definition"], "definition question about <=1 entity"
        elif intent == "relationship" and n >= 1:
            strategy, why = table["relationship"], "relationship question with a linked entity"
        elif intent == "relationship":
            strategy, why = (
                self.cfg.fallback_strategy,
                "relationship cue but no linked entity: fall back",
            )
        elif intent == "comparison" and n >= 2:
            strategy, why = table["comparison"], "comparison of >=2 linked entities"
        elif intent == "comparison":
            strategy, why = "hybrid", "comparison with <2 linked entities: lexical+dense"
        elif intent == "multi_hop" and n >= 1:
            strategy, why = table["multi_hop"], "multi-hop relation chain"
            params["hops"] = max(2, analysis.hops_estimate)
        elif intent == "experimental":
            strategy, why = table["experimental"], "experimental evidence: larger candidate pool"
            params["candidate_pool"] = self.cfg.experimental_candidate_pool
        else:
            strategy, why = self.cfg.fallback_strategy, "no rule matched: fallback"
        if intent in self.cfg.rerank_intents:
            params["rerank"] = True
            why += " + rerank"
        if strategy in GRAPH_REQUIRING and n == 0:
            strategy, why = self.cfg.fallback_strategy, why + " (no linked entity: fallback)"
        return RoutingDecision(
            strategy=strategy,
            params=params,
            rationale=f"intent={intent}, entities={n} -> {strategy}: {why}",
        )
