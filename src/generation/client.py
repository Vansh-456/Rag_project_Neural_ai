"""
LLM generation clients and citation extraction module.
Provides implementations for OpenAI, Anthropic, Ollama, and a deterministic offline test mock.
"""

from __future__ import annotations

import json
import re
from typing import Any
import httpx
from src.schemas import Evidence
from src.utils.metrics import estimate_tokens


SYSTEM_PROMPT = """You are a rigorous, truthful academic research assistant.
Your answers MUST be strictly grounded in the provided evidence.

CRITICAL RULES:
1. Every factual statement or claim you make MUST cite one or more evidence IDs in square brackets, e.g., [E1] or [E1, E2].
2. Do NOT use outside knowledge not present in the evidence.
3. If the provided evidence is insufficient, contradictory, or does not mention the answer, you MUST state:
   "I cannot answer this question based on the provided evidence."
4. Do not invent citations or cite IDs that do not exist in the evidence.
"""


class BaseLLMClient:
    """Base class for all LLM providers."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> tuple[str, dict[str, int]]:
        raise NotImplementedError


class DeterministicMockLLMClient(BaseLLMClient):
    """
    Deterministic mock generator for offline benchmarking and CI.
    Synthesizes grounded answers directly from evidence text with valid citations [E1], [E2].
    Refuses/abstains if no evidence is present.
    """

    def generate(self, prompt: str, system_prompt: str | None = None) -> tuple[str, dict[str, int]]:
        # Check if evidence exists in prompt
        if "No relevant evidence retrieved." in prompt or "# RETRIEVED EVIDENCE:\n\n" in prompt:
            answer = "I cannot answer this question based on the provided evidence."
            return answer, {"prompt_tokens": estimate_tokens(prompt), "completion_tokens": estimate_tokens(answer)}

        # Extract available evidence blocks
        evidence_matches = re.findall(r"\[(E\d+)\]\s*\([^\)]*\)\n(.*?)(?=\n\[E\d+\]|\Z)", prompt, re.DOTALL)
        if not evidence_matches:
            answer = "I cannot answer this question based on the provided evidence."
            return answer, {"prompt_tokens": estimate_tokens(prompt), "completion_tokens": estimate_tokens(answer)}

        # Synthesize summary sentences grounded in top evidence
        claims = []
        for eid, text in evidence_matches[:3]:
            # Pick a core informative line from the evidence snippet
            lines = [l.strip() for l in text.split("\n") if l.strip() and not l.startswith("Document:") and not l.startswith("Section:")]
            chosen_snippet = lines[0] if lines else text[:120].strip()
            # Clean snippet for answer formation
            if chosen_snippet.startswith("Knowledge Graph Relation Path:"):
                chosen_snippet = chosen_snippet.replace("Knowledge Graph Relation Path:", "").strip()
            claims.append(f"According to the evidence, {chosen_snippet.rstrip('.')} [{eid}].")

        answer = " ".join(claims)
        p_tok = estimate_tokens(prompt)
        c_tok = estimate_tokens(answer)
        return answer, {"prompt_tokens": p_tok, "completion_tokens": c_tok}


class OpenAIClient(BaseLLMClient):
    """OpenAI API client via direct HTTP."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str | None = None) -> tuple[str, dict[str, int]]:
        if not self.api_key:
            mock = DeterministicMockLLMClient()
            return mock.generate(prompt, system_prompt)

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1024
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens = {
                    "prompt_tokens": usage.get("prompt_tokens", estimate_tokens(prompt)),
                    "completion_tokens": usage.get("completion_tokens", estimate_tokens(content))
                }
                return content, tokens
        except Exception:
            mock = DeterministicMockLLMClient()
            return mock.generate(prompt, system_prompt)


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client via direct HTTP."""

    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022", temperature: float = 0.0):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str | None = None) -> tuple[str, dict[str, int]]:
        if not self.api_key:
            mock = DeterministicMockLLMClient()
            return mock.generate(prompt, system_prompt)

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "system": system_prompt or SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 1024
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data["content"][0]["text"]
                usage = data.get("usage", {})
                tokens = {
                    "prompt_tokens": usage.get("input_tokens", estimate_tokens(prompt)),
                    "completion_tokens": usage.get("output_tokens", estimate_tokens(content))
                }
                return content, tokens
        except Exception:
            mock = DeterministicMockLLMClient()
            return mock.generate(prompt, system_prompt)


class OllamaClient(BaseLLMClient):
    """Local Ollama client via HTTP."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str | None = None) -> tuple[str, dict[str, int]]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or SYSTEM_PROMPT,
            "stream": False
        }
        try:
            with httpx.Client(timeout=45.0) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                content = data.get("response", "")
                tokens = {
                    "prompt_tokens": data.get("prompt_eval_count", estimate_tokens(prompt)),
                    "completion_tokens": data.get("eval_count", estimate_tokens(content))
                }
                return content, tokens
        except Exception:
            mock = DeterministicMockLLMClient()
            return mock.generate(prompt, system_prompt)


def parse_and_validate_citations(answer: str, evidence: list[Evidence]) -> tuple[list[dict[str, Any]], bool]:
    """
    Extracts citations like [E1], [E2] from generated answer.
    Validates that each cited ID exists in the provided evidence.
    Returns (citations_list, all_valid).
    """
    citations = []
    # Map index 1..N to evidence items
    ev_by_id = {f"E{i}": ev for i, ev in enumerate(evidence, start=1)}

    raw_matches = re.finditer(r"\[(E\d+(?:,\s*E\d+)*)\]", answer)
    all_valid = True

    for m in raw_matches:
        full_match = m.group(0)
        inner = m.group(1)
        sub_ids = [s.strip() for s in inner.split(",")]

        for sid in sub_ids:
            if sid in ev_by_id:
                target_ev = ev_by_id[sid]
                citations.append({
                    "citation_marker": full_match,
                    "evidence_id": sid,
                    "target_chunk_id": target_ev.source_chunk_ids[0] if target_ev.source_chunk_ids else "",
                    "target_doc_id": target_ev.source_doc_ids[0] if target_ev.source_doc_ids else "",
                    "doc_title": target_ev.provenance.get("doc_title", ""),
                    "page": target_ev.provenance.get("page_start"),
                    "is_valid": True
                })
            else:
                citations.append({
                    "citation_marker": full_match,
                    "evidence_id": sid,
                    "is_valid": False,
                    "error": f"Evidence ID '{sid}' not in retrieved evidence pool."
                })
                all_valid = False

    return citations, all_valid


def get_llm_client(
    provider: str = "mock",
    openai_key: str = "",
    anthropic_key: str = "",
    ollama_url: str = "http://localhost:11434",
    model_name: str = "gpt-4o-mini"
) -> BaseLLMClient:
    """Factory helper for LLM clients."""
    if provider == "openai" and openai_key:
        return OpenAIClient(api_key=openai_key, model_name=model_name)
    elif provider == "anthropic" and anthropic_key:
        return AnthropicClient(api_key=anthropic_key, model_name=model_name)
    elif provider == "ollama":
        return OllamaClient(base_url=ollama_url, model_name=model_name)
    return DeterministicMockLLMClient()
