"""Grounded answer generation with mandatory citations.

* ``LLMGenerator``        — prompt (versioned) that forces ``[E#]``/``[G#]`` citations, forbids
                            outside knowledge and requires abstention when evidence is
                            insufficient. Supports streaming.
* ``ExtractiveGenerator`` — no-LLM baseline: selects the evidence sentences that best cover
                            the query terms and cites them. Zero cost, fully faithful by
                            construction (it can only quote), useful as a floor and offline.

``parse_citations`` / ``validate_citations`` check that every cited id exists in the context
and that every claim-bearing sentence carries at least one citation.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from agvrag.interfaces import BuiltContext, LLMClient
from agvrag.retrieval.bm25 import BM25Tokenizer
from agvrag.schemas import EvidenceType, QueryAnalysis
from agvrag.utils.text import split_sentences

ABSTAIN = "INSUFFICIENT_EVIDENCE"

PROMPTS: dict[str, str] = {
    "grounded_v1": (
        "You answer questions about research papers using ONLY the evidence provided in the "
        "context. Rules:\n"
        "1. Every sentence that states a fact must end with one or more citations in square "
        "brackets using the evidence ids exactly as given, e.g. [E1] or [G2][E3].\n"
        "2. Do not use any knowledge that is not in the evidence. Do not invent ids.\n"
        "3. If the evidence does not contain enough information to answer, reply with exactly "
        f"'{ABSTAIN}' followed by one sentence explaining what is missing.\n"
        "4. Graph evidence [G#] lists relations extracted from the papers; prefer text evidence "
        "[E#] when both support a claim, and cite both when helpful.\n"
        "5. Be concise: at most 6 sentences."
    ),
}

_CITE_RE = re.compile(r"\[((?:[EG]\d+)(?:\s*[,;]\s*[EG]\d+)*)\]")


@dataclass
class GenerationResult:
    answer: str
    citations: list[dict[str, Any]]
    validation: dict[str, Any]
    abstained: bool
    usage: dict[str, float] = field(default_factory=dict)
    prompt_version: str = ""


def parse_citations(answer: str) -> list[tuple[int, str, list[str]]]:
    """Return (sentence_index, sentence, [labels]) for each sentence of the answer."""
    out: list[tuple[int, str, list[str]]] = []
    for i, sent in enumerate(split_sentences(answer)):
        labels: list[str] = []
        for m in _CITE_RE.finditer(sent):
            labels.extend(x.strip() for x in re.split(r"[,;]", m.group(1)))
        out.append((i, sent, list(dict.fromkeys(labels))))
    return out


def validate_citations(
    answer: str, ctx: BuiltContext
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = parse_citations(answer)
    citations: list[dict[str, Any]] = []
    invalid: list[str] = []
    uncited: list[str] = []
    for idx, sent, labels in parsed:
        bare = _CITE_RE.sub("", sent).strip()
        is_claim = len(bare.split()) >= 4 and ABSTAIN not in sent
        if is_claim and not labels:
            uncited.append(bare)
        for lab in labels:
            ev = ctx.evidence_ids.get(lab)
            if ev is None:
                invalid.append(lab)
                continue
            citations.append(
                {
                    "label": lab,
                    "sentence_index": idx,
                    "sentence": bare,
                    "evidence_id": ev.evidence_id,
                    "type": ev.type.value,
                    "chunk_ids": ev.source_chunk_ids,
                    "doc_ids": ev.source_doc_ids,
                    "page_start": ev.provenance.get("page_start"),
                    "doc_title": ev.provenance.get("doc_title"),
                }
            )
    n_claims = sum(
        1 for _, s, _ in parsed if len(_CITE_RE.sub("", s).split()) >= 4 and ABSTAIN not in s
    )
    validation = {
        "n_sentences": len(parsed),
        "n_claim_sentences": n_claims,
        "invalid_ids": sorted(set(invalid)),
        "uncited_claims": uncited,
        "citation_coverage": round(1 - len(uncited) / n_claims, 4) if n_claims else 1.0,
        "valid": not invalid and not uncited,
    }
    return citations, validation


def build_prompt(query: str, ctx: BuiltContext) -> str:
    return f"{ctx.text}\n\nAnswer the QUESTION using only the evidence above, with citations."


class LLMGenerator:
    def __init__(
        self,
        llm: LLMClient,
        prompt_version: str = "grounded_v1",
        max_tokens: int = 700,
        temperature: float = 0.0,
    ) -> None:
        if prompt_version not in PROMPTS:
            raise ValueError(f"unknown prompt version {prompt_version}")
        self.llm = llm
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str, analysis: QueryAnalysis, ctx: BuiltContext) -> GenerationResult:
        if not ctx.evidence_ids:
            return GenerationResult(
                f"{ABSTAIN} No evidence was retrieved.",
                [],
                {
                    "valid": True,
                    "n_claim_sentences": 0,
                    "citation_coverage": 1.0,
                    "invalid_ids": [],
                    "uncited_claims": [],
                },
                True,
                prompt_version=self.prompt_version,
            )
        resp = self.llm.complete(
            PROMPTS[self.prompt_version],
            build_prompt(query, ctx),
            self.max_tokens,
            self.temperature,
        )
        answer = resp.text.strip()
        citations, validation = validate_citations(answer, ctx)
        return GenerationResult(
            answer=answer,
            citations=citations,
            validation=validation,
            abstained=answer.startswith(ABSTAIN) or "[REFUSED_BY_MODEL]" in answer,
            usage={
                "llm_input_tokens": resp.input_tokens,
                "llm_output_tokens": resp.output_tokens,
                "llm_cost_usd": resp.cost_usd,
                "llm_latency_ms": resp.latency_ms,
                "llm_cached": float(resp.cached),
            },
            prompt_version=self.prompt_version,
        )

    def stream(self, query: str, analysis: QueryAnalysis, ctx: BuiltContext) -> Iterator[str]:
        if not ctx.evidence_ids:
            yield f"{ABSTAIN} No evidence was retrieved."
            return
        yield from self.llm.stream(
            PROMPTS[self.prompt_version],
            build_prompt(query, ctx),
            self.max_tokens,
            self.temperature,
        )


class ExtractiveGenerator:
    """Quote-and-cite baseline. Abstains when no sentence covers enough query terms."""

    prompt_version = "extractive_v1"

    def __init__(self, max_sentences: int = 3, min_coverage: float = 0.34) -> None:
        self.max_sentences = max_sentences
        self.min_coverage = min_coverage
        self.tok = BM25Tokenizer(stem=True, remove_stopwords=True)

    def generate(self, query: str, analysis: QueryAnalysis, ctx: BuiltContext) -> GenerationResult:
        q_terms = set(self.tok.tokenize(query))
        for ent in analysis.entities:
            q_terms |= set(self.tok.tokenize(ent))
        scored: list[tuple[float, int, str, str]] = []
        for order, (label, ev) in enumerate(ctx.evidence_ids.items()):
            sents = (
                [ev.content] if ev.type != EvidenceType.TEXT_CHUNK else split_sentences(ev.content)
            )
            for s in sents:
                if len(s.split()) < 5 or len(s) > 500:
                    continue
                terms = set(self.tok.tokenize(s))
                cov = len(q_terms & terms) / max(1, len(q_terms))
                prior = 0.15 / (1 + order)
                bonus = (
                    0.1
                    if ev.type != EvidenceType.TEXT_CHUNK
                    and analysis.intent in {"relationship", "multi_hop"}
                    else 0.0
                )
                scored.append((cov + prior + bonus, order, label, s.strip()))
        scored.sort(key=lambda t: (-t[0], t[1]))
        if not scored or scored[0][0] - 0.15 < self.min_coverage:
            answer = f"{ABSTAIN} No retrieved passage covers enough of the question's key terms."
            return GenerationResult(
                answer,
                [],
                {
                    "valid": True,
                    "n_claim_sentences": 0,
                    "citation_coverage": 1.0,
                    "invalid_ids": [],
                    "uncited_claims": [],
                },
                True,
                prompt_version=self.prompt_version,
            )
        chosen: list[tuple[str, str]] = []
        seen_text: set[str] = set()
        for _, _, label, s in scored:
            key = " ".join(s.lower().split())
            if key in seen_text:
                continue
            seen_text.add(key)
            chosen.append((label, s))
            if len(chosen) >= self.max_sentences:
                break
        lines = []
        for label, s in chosen:
            s = s.rstrip()
            if label.startswith("G"):
                s = f"The knowledge graph records: {s}."
            elif not s.endswith((".", "!", "?")):
                s += "."
            lines.append(f"{s[:-1]} [{label}]{s[-1]}")
        answer = " ".join(lines)
        citations, validation = validate_citations(answer, ctx)
        return GenerationResult(
            answer, citations, validation, False, prompt_version=self.prompt_version
        )

    def stream(self, query: str, analysis: QueryAnalysis, ctx: BuiltContext) -> Iterator[str]:
        yield self.generate(query, analysis, ctx).answer
