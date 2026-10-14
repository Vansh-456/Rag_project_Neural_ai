"""Query -> graph entity linking: exact alias, fuzzy string, then embedding fallback.

Short case-sensitive acronyms (``RAG``, ``EM``, ``QA``) only link with their exact case, to
avoid linking ordinary words ("rag", "em"). Longer aliases link case-insensitively.
Overlapping candidate spans are resolved longest-first.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

import numpy as np

from agvrag.interfaces import GraphStore
from agvrag.utils.text import normalize_name

_STOP_SPANS = {
    "what",
    "which",
    "who",
    "how",
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "for",
    "and",
    "in",
    "on",
    "to",
    "does",
    "do",
    "with",
    "by",
    "from",
    "paper",
    "papers",
    "method",
    "methods",
    "model",
    "models",
    "use",
    "used",
    "between",
    "differ",
    "compare",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-+.()^]*[A-Za-z0-9)]|[A-Za-z0-9]")


class EntityLinker:
    def __init__(
        self,
        store: GraphStore,
        fuzzy_threshold: float = 0.88,
        embedder: Any | None = None,
        embedding_threshold: float = 0.80,
        max_ngram: int = 8,
    ) -> None:
        self.store = store
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self.max_ngram = max_ngram
        self.embedder = embedder
        self._exact_ci: dict[str, set[str]] = {}
        self._exact_cs: dict[str, set[str]] = {}
        self._names: dict[str, str] = {}
        self._types: dict[str, str] = {}
        self._degree: dict[str, int] = {}
        for e in store.entities():
            self._names[e.entity_id] = e.name
            self._types[e.entity_id] = e.type
            cs = bool(e.properties.get("case_sensitive"))
            surfaces = [e.name, *e.aliases]
            if e.type == "Paper":
                surfaces = [e.name, e.name.split(":")[0]] if ":" in e.name else [e.name]
                surfaces.append(str(e.properties.get("arxiv_id", "")))
            for s in surfaces:
                if not s:
                    continue
                if cs and (len(s) <= 5 or s.isupper()):
                    self._exact_cs.setdefault(s, set()).add(e.entity_id)
                else:
                    self._exact_ci.setdefault(normalize_name(s), set()).add(e.entity_id)
        rels = store.relations()
        for r in rels:
            self._degree[r.source] = self._degree.get(r.source, 0) + 1
            self._degree[r.target] = self._degree.get(r.target, 0) + 1
        self._ent_ids: list[str] = []
        self._ent_mat: np.ndarray | None = None

    def _rank(self, ids: set[str]) -> list[str]:
        # Prefer non-paper entities, then better-connected ones (more useful seeds).
        return sorted(
            ids, key=lambda i: (self._types.get(i) == "Paper", -self._degree.get(i, 0), i)
        )

    def link(self, query: str, max_entities: int = 6) -> list[dict[str, Any]]:
        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(query)]
        cands: list[dict[str, Any]] = []
        n = len(tokens)
        for size in range(min(self.max_ngram, n), 0, -1):
            for i in range(0, n - size + 1):
                span_toks = tokens[i : i + size]
                start, end = span_toks[0][1], span_toks[-1][2]
                surface = query[start:end].rstrip("?.!,;:")
                surface = re.sub(r"'s$", "", surface)
                if size == 1 and surface.lower() in _STOP_SPANS:
                    continue
                ids: set[str] = set(self._exact_cs.get(surface, set()))
                method = "exact"
                norm = normalize_name(surface)
                if norm in self._exact_ci:
                    ids |= self._exact_ci[norm]
                if not ids and norm.endswith("s") and norm[:-1] in self._exact_ci:
                    ids = set(self._exact_ci[norm[:-1]])
                    method = "alias_plural"
                bounded_by_stop = (
                    span_toks[0][0].lower() in _STOP_SPANS
                    or span_toks[-1][0].lower() in _STOP_SPANS
                )
                if not ids and len(norm) >= 5 and size <= 4 and not bounded_by_stop:
                    best, best_key = 0.0, None
                    for key in self._exact_ci:
                        if abs(len(key) - len(norm)) > 3:
                            continue
                        r = difflib.SequenceMatcher(None, norm, key).ratio()
                        if r > best:
                            best, best_key = r, key
                    if best_key and best >= self.fuzzy_threshold:
                        ids = set(self._exact_ci[best_key])
                        method = f"fuzzy:{best:.2f}"
                if ids:
                    eid = self._rank(ids)[0]
                    score = (
                        1.0 if method == "exact" else (0.9 if method == "alias_plural" else 0.75)
                    )
                    cands.append(
                        {
                            "entity_id": eid,
                            "name": self._names[eid],
                            "type": self._types[eid],
                            "surface": surface,
                            "method": method,
                            "score": score,
                            "start": start,
                            "end": end,
                        }
                    )
        # longest non-overlapping spans
        cands.sort(key=lambda c: (c["method"] != "exact", -(c["end"] - c["start"]), c["start"]))
        taken: list[tuple[int, int]] = []
        linked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for c in cands:
            if any(not (c["end"] <= s or c["start"] >= e) for s, e in taken):
                continue
            taken.append((c["start"], c["end"]))
            if c["entity_id"] in seen:
                continue
            seen.add(c["entity_id"])
            linked.append(c)
        linked.sort(key=lambda c: c["start"])
        if not linked and self.embedder is not None:
            linked = self._embedding_link(query)
        for c in linked:
            c.pop("start", None)
            c.pop("end", None)
        return linked[:max_entities]

    def _embedding_link(self, query: str, top: int = 2) -> list[dict[str, Any]]:
        assert self.embedder is not None
        if self._ent_mat is None:
            self._ent_ids = [i for i, t in self._types.items() if t not in {"Paper", "Author"}]
            self._ent_mat = self.embedder.embed_documents([self._names[i] for i in self._ent_ids])
        if not self._ent_ids:
            return []
        q = self.embedder.embed_query(query)
        sims = self._ent_mat @ q
        order = np.argsort(-sims)[:top]
        return [
            {
                "entity_id": self._ent_ids[i],
                "name": self._names[self._ent_ids[i]],
                "type": self._types[self._ent_ids[i]],
                "surface": query,
                "method": "embedding",
                "score": round(float(sims[i]), 4),
                "start": 0,
                "end": 0,
            }
            for i in order
            if sims[i] >= self.embedding_threshold
        ]
