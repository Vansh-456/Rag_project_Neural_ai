"""Retrieval budget: explicit limits, per-action cost model, and the utility objective.

The pipeline optimizes (per query) the trade-off from the spec:

    max_pi  E[ Q(A, E) - lambda * C(R) ]

where C(R) is the sum of per-action costs (``action_costs``) charged through ``charge``.
Hard limits raise ``BudgetExceeded`` *before* the action runs (``can_afford`` lets callers
check without raising), so limits are provably enforced — see ``tests/test_budget.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agvrag.config import BudgetConfig


class BudgetExceeded(RuntimeError):
    def __init__(self, limit: str, used: float, cap: float) -> None:
        super().__init__(f"budget exceeded: {limit} ({used} > {cap})")
        self.limit = limit


_LIMIT_OF = {
    "dense": "retrieval_calls",
    "bm25": "retrieval_calls",
    "graph_hop": "graph_hops",
    "rerank_candidate": "candidates",
}


@dataclass
class BudgetTracker:
    cfg: BudgetConfig
    enforce: bool = True
    retrieval_calls: int = 0
    graph_hops: int = 0
    candidates: int = 0
    tokens: int = 0
    cost_units: float = 0.0
    log: list[tuple[str, float]] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def _caps(self) -> dict[str, float]:
        return {
            "retrieval_calls": self.cfg.max_retrieval_calls,
            "graph_hops": self.cfg.max_graph_hops,
            "candidates": self.cfg.max_candidates,
            "tokens": self.cfg.max_tokens,
        }

    def _used(self, limit: str) -> float:
        return float(getattr(self, limit))

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def can_afford(self, action: str, amount: float = 1.0) -> bool:
        limit = _LIMIT_OF.get(action, "tokens" if action == "llm_tokens" else "")
        if self.elapsed_ms() > self.cfg.max_latency_ms:
            return False
        if not limit:
            return True
        return self._used(limit) + amount <= self._caps()[limit]

    def charge(self, action: str, amount: float = 1.0) -> None:
        """Record an action. Raises BudgetExceeded (when enforcing) if a cap would be exceeded."""
        limit = _LIMIT_OF.get(action, "tokens" if action == "llm_tokens" else "")
        if self.enforce:
            if self.elapsed_ms() > self.cfg.max_latency_ms:
                raise BudgetExceeded("latency_ms", self.elapsed_ms(), self.cfg.max_latency_ms)
            if limit and self._used(limit) + amount > self._caps()[limit]:
                raise BudgetExceeded(limit, self._used(limit) + amount, self._caps()[limit])
        if limit:
            setattr(self, limit, type(getattr(self, limit))(self._used(limit) + amount))
        if action == "llm_tokens":
            self.cost_units += self.cfg.action_costs.get("llm_1k_tokens", 0.0) * amount / 1000
        else:
            self.cost_units += self.cfg.action_costs.get(action, 0.0) * amount
        self.log.append((action, amount))

    def summary(self) -> dict[str, float]:
        return {
            "retrieval_calls": float(self.retrieval_calls),
            "graph_hops": float(self.graph_hops),
            "rerank_candidates": float(self.candidates),
            "llm_tokens": float(self.tokens),
            "cost_units": round(self.cost_units, 4),
        }


def utility(quality: float, cost_units: float, lam: float) -> float:
    """Q - lambda * C."""
    return quality - lam * cost_units
