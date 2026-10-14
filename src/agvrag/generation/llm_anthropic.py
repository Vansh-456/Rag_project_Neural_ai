"""Claude via the official ``anthropic`` SDK.

Notes for current models (Claude Opus 5.5 / Sonnet 5.5 / Haiku 5.5):
* Sampling parameters (``temperature``) are rejected, and thinking cannot be disabled on
  Opus 5.5 — ``output_config.effort`` is the depth/cost control. ``temperature`` from the
  ``LLMClient`` interface is therefore ignored here.
* Server-side refusal fallback (``fallbacks="default"`` + beta header) is opted into by
  default for models that support it on the Claude API; set
  ``generation.anthropic_fallbacks: false`` to disable (e.g. on Bedrock/Vertex/Foundry).
* ``stop_reason == "refusal"`` is surfaced as an empty answer with a marker rather than
  silently returning partial content.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import anthropic

from agvrag.generation.llm import estimate_cost
from agvrag.interfaces import LLMResponse
from agvrag.utils.logging import get_logger

log = get_logger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"}
REFUSAL_MARKER = "[REFUSED_BY_MODEL]"


class AnthropicClient:
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        effort: str = "medium",
        fallbacks: bool = True,
    ) -> None:
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY / auth token / `ant` profile.
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.effort = effort
        self.use_fallbacks = fallbacks and model in _FALLBACK_MODELS

    def _params(self, system: str, prompt: str, max_tokens: int, json_mode: bool) -> dict[str, Any]:
        if json_mode:
            system = f"{system}\n\nRespond with a single JSON object and nothing else."
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.effort},
        }
        if self.use_fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _messages(self) -> Any:
        return self.client.beta.messages if self.use_fallbacks else self.client.messages

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        params = self._params(system, prompt, max_tokens, json_mode)
        resp = self._messages().create(**params)
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            log.warning("model refusal", extra={"category": getattr(details, "category", None)})
            text = REFUSAL_MARKER
        else:
            text = "".join(b.text for b in resp.content if b.type == "text")
            if resp.stop_reason == "max_tokens":
                log.warning("llm output truncated at max_tokens", extra={"max_tokens": max_tokens})
        served_by = getattr(resp, "model", self.model) or self.model
        usage = resp.usage
        return LLMResponse(
            text=text,
            model=served_by,
            input_tokens=int(usage.input_tokens),
            output_tokens=int(usage.output_tokens),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=estimate_cost(served_by, int(usage.input_tokens), int(usage.output_tokens)),
        )

    def stream(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> Iterator[str]:
        params = self._params(system, prompt, max_tokens, json_mode=False)
        with self._messages().stream(**params) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
            if final.stop_reason == "refusal":
                yield f"\n{REFUSAL_MARKER}"
