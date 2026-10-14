"""
Timing, token counting, and cost calculation utilities.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager for measuring elapsed execution time in milliseconds."""
    result: dict[str, float] = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - start) * 1000.0, 2)


def estimate_tokens(text: str) -> int:
    """Rough token estimation (approx 4 chars per token for English text)."""
    if not text:
        return 0
    words = text.split()
    # Approx 1.3 tokens per whitespace-delimited word
    return max(1, int(len(words) * 1.3))
