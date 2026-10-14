"""Claim / evidence verification.

1. Claim extraction: each answer sentence (citations stripped) is an atomic claim
   (an LLM-based decomposition can be plugged in via ``claim_splitter``).
2. Per-claim support check against the evidence it cites:
     * ``lexical`` — content-term coverage of the claim by the cited evidence
     * ``nli``     — cross-encoder NLI (entailment / contradiction / neutral)
     * ``llm``     — LLM judge returning supported | contradicted | unsupported
   Labels: supported, contradicted, unsupported. An uncited claim is "unsupported".
3. Contradiction detection across the top text evidence items (NLI only).
4. Sufficiency: fraction supported >= threshold and the answer did not abstain.

The verifier only *reports*; the pipeline decides whether to re-retrieve (bounded loop).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from agvrag.config import VerificationConfig
from agvrag.graph.extraction import parse_json_object
from agvrag.interfaces import BuiltContext, LLMClient
from agvrag.retrieval.bm25 import BM25Tokenizer
from agvrag.schemas import EvidenceType
from agvrag.utils.text import split_sentences

_CITE_RE = re.compile(r"\[(?:[EG]\d+)(?:\s*[,;]\s*[EG]\d+)*\]")
_LABELS = ("supported", "contradicted", "unsupported")


def extract_claims(answer: str) -> list[dict[str, Any]]:
    claims = []
    for i, sent in enumerate(split_sentences(answer)):
        labels = re.findall(r"[EG]\d+", " ".join(_CITE_RE.findall(sent)))
        bare = _CITE_RE.sub("", sent).strip()
        if len(bare.split()) < 4 or bare.startswith("INSUFFICIENT_EVIDENCE"):
            continue
        claims.append({"index": i, "claim": bare, "cited": list(dict.fromkeys(labels))})
    return claims


class ClaimVerifier:
    def __init__(
        self,
        cfg: VerificationConfig,
        llm: LLMClient | None = None,
        nli_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.tok = BM25Tokenizer(stem=True, remove_stopwords=True)
        self._nli: Any = None
        self._nli_factory = nli_factory

    # ------------------------------------------------------------------ checks
    def _lexical(self, claim: str, evidence: str) -> tuple[str, float]:
        c = set(self.tok.tokenize(claim))
        e = set(self.tok.tokenize(evidence))
        cov = len(c & e) / max(1, len(c))
        return ("supported" if cov >= self.cfg.support_threshold else "unsupported"), round(cov, 4)

    def _nli_model(self) -> Any:
        if self._nli is None:
            if self._nli_factory is not None:
                self._nli = self._nli_factory()
            else:
                from sentence_transformers import CrossEncoder

                self._nli = CrossEncoder(self.cfg.nli_model)
        return self._nli

    def _nli_check(self, claim: str, evidence: str) -> tuple[str, float]:
        import numpy as np

        logits = self._nli_model().predict([(evidence[:2000], claim)], show_progress_bar=False)
        probs = np.exp(logits[0]) / np.exp(logits[0]).sum()
        # nli-deberta-v3 label order: contradiction, entailment, neutral
        contra, entail = float(probs[0]), float(probs[1])
        if entail >= self.cfg.support_threshold:
            return "supported", round(entail, 4)
        if contra >= self.cfg.support_threshold:
            return "contradicted", round(contra, 4)
        return "unsupported", round(entail, 4)

    def _llm_check(self, claim: str, evidence: str) -> tuple[str, float]:
        assert self.llm is not None
        resp = self.llm.complete(
            "You check whether EVIDENCE supports a CLAIM. Output JSON "
            '{"label": "supported|contradicted|unsupported", "confidence": 0..1}.',
            f"EVIDENCE:\n{evidence[:3000]}\n\nCLAIM: {claim}",
            max_tokens=200,
            json_mode=True,
        )
        try:
            d = parse_json_object(resp.text)
            label = str(d.get("label", "unsupported")).lower()
            return (label if label in _LABELS else "unsupported"), float(d.get("confidence", 0.5))
        except (ValueError, TypeError):
            return "unsupported", 0.0

    def check(self, claim: str, evidence: str) -> tuple[str, float]:
        if self.cfg.method == "nli":
            return self._nli_check(claim, evidence)
        if self.cfg.method == "llm" and self.llm is not None:
            return self._llm_check(claim, evidence)
        return self._lexical(claim, evidence)

    # ------------------------------------------------------------------ main
    def verify(
        self, answer: str, citations: list[dict[str, Any]], context: BuiltContext
    ) -> dict[str, Any]:
        abstained = answer.strip().startswith("INSUFFICIENT_EVIDENCE")
        results = []
        for c in extract_claims(answer):
            evs = [
                context.evidence_ids[label] for label in c["cited"] if label in context.evidence_ids
            ]
            if not evs:
                results.append(
                    {**c, "label": "unsupported", "score": 0.0, "reason": "no valid citation"}
                )
                continue
            best = ("unsupported", 0.0)
            for ev in evs:
                label, score = self.check(c["claim"], ev.content)
                if (
                    (label == "supported" and (best[0] != "supported" or score > best[1]))
                    or (label == "contradicted" and best[0] == "unsupported")
                    or (label == best[0] and score > best[1])
                ):
                    best = (label, score)
            results.append({**c, "label": best[0], "score": best[1]})
        contradictions: list[dict[str, Any]] = []
        if self.cfg.check_contradictions and self.cfg.method == "nli":
            texts = [
                (lab, ev)
                for lab, ev in context.evidence_ids.items()
                if ev.type == EvidenceType.TEXT_CHUNK
            ][:5]
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    label, score = self._nli_check(texts[j][1].content[:500], texts[i][1].content)
                    if label == "contradicted":
                        contradictions.append({"a": texts[i][0], "b": texts[j][0], "score": score})
        n = len(results)
        n_sup = sum(r["label"] == "supported" for r in results)
        frac = n_sup / n if n else 0.0
        sufficient = (not abstained) and n > 0 and frac >= self.cfg.sufficiency_threshold
        return {
            "method": self.cfg.method,
            "claims": results,
            "n_claims": n,
            "n_supported": n_sup,
            "n_contradicted": sum(r["label"] == "contradicted" for r in results),
            "n_unsupported": sum(r["label"] == "unsupported" for r in results),
            "support_rate": round(frac, 4),
            "contradictions": contradictions,
            "abstained": abstained,
            "sufficient": sufficient,
        }
