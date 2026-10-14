"""Latency instrumentation."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    """Accumulates wall-clock milliseconds per named stage (re-entrant stages add up)."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start) * 1000

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def as_costs(self) -> dict[str, float]:
        out = {f"latency_ms.{k}": round(v, 3) for k, v in self.stages.items()}
        out["latency_ms.total"] = round(self.elapsed_ms(), 3)
        return out
