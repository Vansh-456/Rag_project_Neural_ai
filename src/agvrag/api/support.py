"""API infrastructure: trace persistence, LRU result cache, rate limiting, request metrics."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any


class TraceStore:
    """SQLite-backed trace persistence (one row per query execution)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, created_at TEXT, pipeline TEXT, "
            "query TEXT, strategy TEXT, latency_ms REAL, body TEXT)"
        )
        self._lock = threading.Lock()

    def save(self, trace: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    trace["trace_id"],
                    trace["created_at"],
                    trace["pipeline"],
                    trace["query_analysis"]["query"],
                    trace["routing"]["strategy"],
                    trace["costs"].get("latency_ms.total", 0.0),
                    json.dumps(trace, default=str),
                ),
            )
            self._db.commit()

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT body FROM traces WHERE id = ?", (trace_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, created_at, pipeline, query, strategy, latency_ms FROM traces "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "trace_id": r[0],
                "created_at": r[1],
                "pipeline": r[2],
                "query": r[3],
                "strategy": r[4],
                "latency_ms": r[5],
            }
            for r in rows
        ]


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._d: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any | None:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t else 0.0


class RateLimiter:
    """Sliding-window limiter: at most ``limit`` requests per ``window`` seconds per key."""

    def __init__(self, limit: int, window: float = 60.0) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, self.window - (now - q[0])
            q.append(now)
            return True, 0.0


class RequestMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts: dict[str, int] = defaultdict(int)
        self.errors: dict[str, int] = defaultdict(int)
        self.latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=1000))

    def record(self, route: str, ms: float, status: int) -> None:
        with self._lock:
            self.counts[route] += 1
            self.latencies[route].append(ms)
            if status >= 500:
                self.errors[route] += 1

    def snapshot(self) -> dict[str, Any]:
        import numpy as np

        with self._lock:
            return {
                r: {
                    "count": self.counts[r],
                    "errors": self.errors[r],
                    "p50_ms": round(float(np.percentile(list(v), 50)), 2) if v else None,
                    "p95_ms": round(float(np.percentile(list(v), 95)), 2) if v else None,
                }
                for r, v in self.latencies.items()
            }
