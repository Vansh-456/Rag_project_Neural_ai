"""Provider-agnostic LLM layer: shared helpers, fake + Ollama clients, response cache, factory.

Provider implementations live in their own modules (``llm_anthropic``, ``llm_openai``) so
each SDK is imported only when that provider is selected.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx

from agvrag.config import GenerationConfig, Secrets
from agvrag.interfaces import LLMClient, LLMResponse
from agvrag.utils.logging import get_logger

log = get_logger(__name__)

# USD per 1M tokens (input, output). Used for cost accounting only; verify against the
# provider's pricing page before quoting costs. Unknown models report cost 0.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.00, 50.00),
}

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1:8b",
    "fake": "fake-llm",
    "extractive": "extractive",
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICING.get(model)
    if price is None:
        return 0.0
    return round(input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1], 6)


def with_retries(
    fn: Callable[[], LLMResponse],
    retries: int,
    retry_on: tuple[type[BaseException], ...],
    base_delay: float = 1.0,
) -> LLMResponse:
    """Exponential backoff for transport-level failures (SDKs add their own retries too)."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == retries:
                raise
            delay = min(30.0, base_delay * 2**attempt)
            log.warning(
                "llm call failed; retrying", extra={"error": str(exc)[:200], "delay": delay}
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


class FakeLLMClient:
    """Deterministic test double. ``responder(system, prompt) -> text``."""

    def __init__(
        self, responder: Callable[[str, str], str] | None = None, model: str = "fake-llm"
    ) -> None:
        self.model = model
        self.responder = responder or (lambda s, p: "INSUFFICIENT_EVIDENCE")
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append((system, prompt))
        text = self.responder(system, prompt)
        return LLMResponse(
            text=text, model=self.model, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4
        )

    def stream(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> Iterator[str]:
        text = self.complete(system, prompt, max_tokens, temperature).text
        for i in range(0, len(text), 16):
            yield text[i : i + 16]


class OllamaClient:
    """Local models via Ollama's HTTP API (``/api/chat``)."""

    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434",
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.max_retries = max_retries
        self.http = httpx.Client(timeout=timeout)

    def _payload(
        self,
        system: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        return body

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        def call() -> LLMResponse:
            t0 = time.perf_counter()
            r = self.http.post(
                f"{self.url}/api/chat",
                json=self._payload(system, prompt, max_tokens, temperature, False, json_mode),
            )
            r.raise_for_status()
            d = r.json()
            return LLMResponse(
                text=d["message"]["content"],
                model=self.model,
                input_tokens=int(d.get("prompt_eval_count", 0)),
                output_tokens=int(d.get("eval_count", 0)),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return with_retries(call, self.max_retries, (httpx.TransportError,))

    def stream(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> Iterator[str]:
        with self.http.stream(
            "POST",
            f"{self.url}/api/chat",
            json=self._payload(system, prompt, max_tokens, temperature, True),
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                piece = d.get("message", {}).get("content", "")
                if piece:
                    yield piece


class CachedLLMClient:
    """SQLite response cache keyed by (model, system, prompt, max_tokens, temperature, json).

    Deterministic replays make experiments reproducible and repeated evaluation cheap.
    Streaming bypasses the cache.
    """

    def __init__(self, inner: LLMClient, path: Path) -> None:
        self.inner = inner
        self.model = inner.model
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        key = hashlib.sha256(
            json.dumps([self.model, system, prompt, max_tokens, temperature, json_mode]).encode()
        ).hexdigest()
        with self._lock:
            row = self._db.execute("SELECT v FROM llm WHERE k = ?", (key,)).fetchone()
        if row:
            self.hits += 1
            d = json.loads(row[0])
            return LLMResponse(**{**d, "cached": True, "cost_usd": 0.0, "latency_ms": 0.0})
        self.misses += 1
        resp = self.inner.complete(system, prompt, max_tokens, temperature, json_mode)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO llm (k, v) VALUES (?, ?)", (key, json.dumps(resp.__dict__))
            )
            self._db.commit()
        return resp

    def stream(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> Iterator[str]:
        return self.inner.stream(system, prompt, max_tokens, temperature)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def build_llm(
    cfg: GenerationConfig,
    secrets: Secrets | None = None,
    cache_dir: Path | None = None,
    effort: str | None = None,
) -> LLMClient | None:
    """Return an LLM client for ``cfg.provider`` or ``None`` for the extractive generator."""
    secrets = secrets or Secrets()
    model = cfg.model or DEFAULT_MODELS[cfg.provider]
    client: LLMClient
    if cfg.provider == "extractive":
        return None
    if cfg.provider == "fake":
        client = FakeLLMClient(model=model)
    elif cfg.provider == "anthropic":
        from agvrag.generation.llm_anthropic import AnthropicClient

        client = AnthropicClient(
            model=model,
            api_key=secrets.anthropic_api_key,
            timeout=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
            effort=effort or cfg.effort,
            fallbacks=cfg.anthropic_fallbacks,
        )
    elif cfg.provider == "openai":
        from agvrag.generation.llm_openai import OpenAIClient

        client = OpenAIClient(
            model=model,
            api_key=secrets.openai_api_key,
            timeout=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
        )
    elif cfg.provider == "ollama":
        client = OllamaClient(
            model=model,
            url=cfg.ollama_url,
            timeout=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
        )
    else:
        raise ValueError(f"unknown provider {cfg.provider}")
    if cfg.cache_responses and cache_dir is not None and cfg.provider != "fake":
        return CachedLLMClient(client, cache_dir / "llm_cache.sqlite")
    return client
