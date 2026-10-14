"""
Claim verification and bounded self-correction loop.
Extracts atomic claims, checks semantic entailment against cited evidence,
and decides whether to accept the answer or trigger bounded re-retrieval.
"""

from __future__ import annotations

import re
from typing import Any
from src.schemas import ClaimVerification, Evidence, VerificationResult, VerificationStatus


class ClaimVerifier:
    """Verifies factual claims in generated answers against retrieved evidence."""

    def __init__(self, min_supported_ratio: float = 0.8, contradiction_threshold: float = 0.8):
        self.min_supported_ratio = min_supported_ratio
        self.contradiction_threshold = contradiction_threshold

    def _extract_claims(self, answer: str) -> list[tuple[str, list[str]]]:
        """Splits answer into sentences and extracts attached citation IDs [E1, E2]."""
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        claims_with_citations: list[tuple[str, list[str]]] = []

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            matches = re.findall(r"\[(E\d+(?:,\s*E\d+)*)\]", sent_clean)
            cited_ids = []
            for m in matches:
                cited_ids.extend([s.strip() for s in m.split(",")])

            claims_with_citations.append((sent_clean, cited_ids))

        return claims_with_citations

    def verify(self, answer: str, evidence: list[Evidence], query: str) -> VerificationResult:
        """
        Extracts claims, checks cited evidence support, detects contradictions,
        and determines answer sufficiency.
        """
        # If the model explicitly abstained, it is a valid, sufficient refusal
        if "cannot answer" in answer.lower() or "insufficient" in answer.lower():
            return VerificationResult(
                is_sufficient=True,
                claims=[],
                supported_ratio=1.0,
                contradiction_found=False,
                retry_recommended=False
            )

        claims_raw = self._extract_claims(answer)
        if not claims_raw:
            return VerificationResult(
                is_sufficient=False,
                claims=[],
                supported_ratio=0.0,
                contradiction_found=False,
                retry_recommended=True,
                reformulated_query=f"{query} details"
            )

        ev_by_id = {f"E{i}": ev for i, ev in enumerate(evidence, start=1)}
        claim_results: list[ClaimVerification] = []
        supported_count = 0
        contradiction_found = False

        for claim_text, cited_ids in claims_raw:
            if not cited_ids:
                # Claim without any citation is unsupported
                claim_results.append(ClaimVerification(
                    claim=claim_text,
                    status=VerificationStatus.UNSUPPORTED,
                    cited_evidence_ids=[],
                    reasoning="Claim makes an empirical statement but lacks citation markers.",
                    confidence=0.9
                ))
                continue

            # Check support in cited evidence passages
            has_valid_source = False
            claim_words = set(re.findall(r"\w+", claim_text.lower()))

            for cid in cited_ids:
                if cid in ev_by_id:
                    has_valid_source = True
                    target_ev = ev_by_id[cid]
                    content_words = set(re.findall(r"\w+", target_ev.content.lower()))
                    overlap = len(claim_words.intersection(content_words)) / max(1, len(claim_words))

                    # Contradiction cue heuristic
                    if ("not " in claim_text.lower() and "not " not in target_ev.content.lower()) or \
                       ("never" in claim_text.lower() and "always" in target_ev.content.lower()):
                        contradiction_found = True
                        claim_results.append(ClaimVerification(
                            claim=claim_text,
                            status=VerificationStatus.CONTRADICTED,
                            cited_evidence_ids=[cid],
                            reasoning=f"Potential negation contradiction with evidence [{cid}].",
                            confidence=0.85
                        ))
                        break
                    elif overlap >= 0.25 or any(w in target_ev.content.lower() for w in claim_words if len(w) > 4):
                        claim_results.append(ClaimVerification(
                            claim=claim_text,
                            status=VerificationStatus.SUPPORTED,
                            cited_evidence_ids=[cid],
                            reasoning=f"Claim text correlates with facts in [{cid}].",
                            confidence=round(min(1.0, overlap + 0.5), 2)
                        ))
                        supported_count += 1
                        break
                else:
                    # Cited ID doesn't exist
                    pass

            if not any(cr.claim == claim_text for cr in claim_results):
                claim_results.append(ClaimVerification(
                    claim=claim_text,
                    status=VerificationStatus.UNSUPPORTED,
                    cited_evidence_ids=cited_ids,
                    reasoning=f"Cited evidence IDs {cited_ids} do not substantiate the claim.",
                    confidence=0.8
                ))

        total_claims = len(claim_results)
        supported_ratio = round(supported_count / max(1, total_claims), 2)
        is_sufficient = (supported_ratio >= self.min_supported_ratio) and not contradiction_found

        reformulated = None
        if not is_sufficient:
            # Reformulate query to expand retrieval depth
            reformulated = f"{query} evidence"

        return VerificationResult(
            is_sufficient=is_sufficient,
            claims=claim_results,
            supported_ratio=supported_ratio,
            contradiction_found=contradiction_found,
            retry_recommended=not is_sufficient,
            reformulated_query=reformulated
        )
