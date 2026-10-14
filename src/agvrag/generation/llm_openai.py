"""OpenAI chat-completions client (optional provider, ``openai`` extra)."""

from __future__ import annotations

import time
from collections.abc import Iterator

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from agvrag.interfaces import LLMResponse


class OpenAIClient:
    def __init__(
        self, model: str, api_key: str | None = None, timeout: float = 60.0, max_retries: int = 3
    ) -> None:
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if json_mode:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
        else:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature, messages=messages
            )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def stream(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
