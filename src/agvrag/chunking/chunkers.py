"""Four chunking strategies behind one ``Chunker`` interface.

* ``fixed``            — sliding token window over the whole document; ignores structure.
* ``recursive``        — split on paragraph -> sentence -> word until pieces fit, then pack.
* ``semantic``         — split between adjacent sentences whose embedding similarity drops.
* ``structure_aware``  — never crosses a section boundary; prefixes "title > section" context.

Invariants (tested): no empty chunks; every chunk <= ``chunk_size`` tokens (except a single
word longer than the budget); deterministic ids; overlap < chunk_size.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from agvrag.config import ChunkingConfig
from agvrag.interfaces import Embedder
from agvrag.schemas import Chunk, Document
from agvrag.utils.ids import chunk_id
from agvrag.utils.text import count_tokens, split_sentences, words_with_tokens


def chunk_embed_text(chunk: Chunk) -> str:
    """Text used for embedding and BM25: optional context prefix + body."""
    ctx = chunk.metadata.get("context")
    return f"{ctx}\n{chunk.text}" if ctx else chunk.text


@dataclass
class _Unit:
    """A piece of text with its token count and originating section index."""

    text: str
    tokens: int
    section: int


class _BaseChunker:
    name = "base"

    def __init__(self, cfg: ChunkingConfig) -> None:
        if cfg.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= cfg.overlap < cfg.chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.cfg = cfg
        self.size = cfg.chunk_size
        self.overlap = cfg.overlap

    @property
    def params(self) -> dict[str, int | float | str]:
        return {"size": self.size, "overlap": self.overlap}

    def chunk(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError

    def _make(self, doc: Document, ordinal: int, units: list[_Unit], joiner: str = " ") -> Chunk:
        text = joiner.join(u.text for u in units).strip()
        secs = sorted({u.section for u in units})
        first = doc.sections[secs[0]] if doc.sections else None
        last = doc.sections[secs[-1]] if doc.sections else None
        return Chunk(
            chunk_id=chunk_id(doc.doc_id, self.name, self.params, ordinal),
            doc_id=doc.doc_id,
            text=text,
            section_heading=first.heading if first else None,
            page_start=first.page_start if first else None,
            page_end=last.page_end if last else None,
            strategy=self.name,
            token_count=count_tokens(text),
            metadata={
                "doc_title": doc.title,
                "year": doc.year,
                "crosses_section": len(secs) > 1,
                "sections": [doc.sections[i].heading for i in secs] if doc.sections else [],
                "ordinal": ordinal,
            },
        )

    def _doc_sections(self, doc: Document) -> list[tuple[int, str]]:
        return [(i, s.text) for i, s in enumerate(doc.sections) if s.text.strip()]

    def _split_long_sentence(self, sentence: str, section: int) -> list[_Unit]:
        """Break a sentence longer than the budget into word windows."""
        units: list[_Unit] = []
        buf: list[str] = []
        tok = 0
        for w, t in words_with_tokens(sentence):
            if buf and tok + t > self.size:
                units.append(_Unit(" ".join(buf), tok, section))
                buf, tok = [], 0
            buf.append(w)
            tok += t
        if buf:
            units.append(_Unit(" ".join(buf), tok, section))
        return units

    def _sentence_units(self, text: str, section: int) -> list[_Unit]:
        units: list[_Unit] = []
        for s in split_sentences(text):
            t = count_tokens(s)
            if t > self.size:
                units.extend(self._split_long_sentence(s, section))
            elif t > 0:
                units.append(_Unit(s, t, section))
        return units

    def _pack(self, units: list[_Unit]) -> list[list[_Unit]]:
        """Greedy packing of units into windows <= size with unit-level overlap."""
        windows: list[list[_Unit]] = []
        cur: list[_Unit] = []
        cur_tok = 0
        for u in units:
            if cur and cur_tok + u.tokens + 1 > self.size:
                windows.append(cur)
                carry: list[_Unit] = []
                carry_tok = 0
                for prev in reversed(cur):
                    if carry_tok + prev.tokens > self.overlap:
                        break
                    carry.insert(0, prev)
                    carry_tok += prev.tokens
                # Overlap must leave room for progress.
                if carry_tok + u.tokens + 1 > self.size:
                    carry, carry_tok = [], 0
                cur, cur_tok = list(carry), carry_tok
            cur.append(u)
            cur_tok += u.tokens + 1
        if cur:
            windows.append(cur)
        return windows


class FixedSizeChunker(_BaseChunker):
    """Token windows of ``size`` with step ``size - overlap`` over the concatenated document."""

    name = "fixed"

    def chunk(self, doc: Document) -> list[Chunk]:
        words: list[_Unit] = []
        for idx, text in self._doc_sections(doc):
            words.extend(_Unit(w, t, idx) for w, t in words_with_tokens(text))
        chunks: list[Chunk] = []
        start, ordinal = 0, 0
        while start < len(words):
            tok, end = 0, start
            while end < len(words) and (tok + words[end].tokens <= self.size or end == start):
                tok += words[end].tokens
                end += 1
            chunks.append(self._make(doc, ordinal, words[start:end]))
            ordinal += 1
            if end >= len(words):
                break
            # step back by `overlap` tokens, but always advance at least one word
            back, j = 0, end
            while j > start + 1 and back + words[j - 1].tokens <= self.overlap:
                back += words[j - 1].tokens
                j -= 1
            start = j
        return chunks


class RecursiveChunker(_BaseChunker):
    """Paragraph -> sentence -> word splitting, then greedy packing (may cross sections)."""

    name = "recursive"

    def chunk(self, doc: Document) -> list[Chunk]:
        units: list[_Unit] = []
        for idx, text in self._doc_sections(doc):
            for para in (p for p in text.split("\n\n") if p.strip()):
                t = count_tokens(para)
                if t <= self.size:
                    units.append(_Unit(para.strip(), t, idx))
                else:
                    units.extend(self._sentence_units(para, idx))
        return [self._make(doc, i, w) for i, w in enumerate(self._pack(units))]


class SemanticChunker(_BaseChunker):
    """Breaks where consecutive-sentence cosine similarity < threshold (within size limits).

    Requires an embedder; sentence embeddings are computed per document in one batch.
    """

    name = "semantic"

    def __init__(self, cfg: ChunkingConfig, embedder: Embedder) -> None:
        super().__init__(cfg)
        self.embedder = embedder
        self.threshold = cfg.semantic_threshold
        self.min_tokens = cfg.semantic_min_tokens

    @property
    def params(self) -> dict[str, int | float | str]:
        return {**super().params, "thr": self.threshold, "emb": self.embedder.model_name}

    def chunk(self, doc: Document) -> list[Chunk]:
        units: list[_Unit] = []
        for idx, text in self._doc_sections(doc):
            units.extend(self._sentence_units(text, idx))
        if not units:
            return []
        vecs = self.embedder.embed_documents([u.text for u in units])
        sims = np.einsum("ij,ij->i", vecs[:-1], vecs[1:]) if len(units) > 1 else np.array([])
        groups: list[list[_Unit]] = []
        cur: list[_Unit] = [units[0]]
        cur_tok = units[0].tokens
        for i in range(1, len(units)):
            u = units[i]
            topic_shift = sims[i - 1] < self.threshold and cur_tok >= self.min_tokens
            if topic_shift or cur_tok + u.tokens + 1 > self.size:
                groups.append(cur)
                cur, cur_tok = [], 0
            cur.append(u)
            cur_tok += u.tokens + 1
        groups.append(cur)
        return [self._make(doc, i, g) for i, g in enumerate(groups)]


class StructureAwareChunker(_BaseChunker):
    """Packs sentences within a section only; attaches "title > heading" as context."""

    name = "structure_aware"

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0
        for idx, text in self._doc_sections(doc):
            section = doc.sections[idx]
            ctx = f"{doc.title} > {section.heading}" if self.cfg.include_title_context else ""
            budget_cut = count_tokens(ctx) + 1 if ctx else 0
            saved = self.size
            self.size = max(16, self.size - budget_cut)  # context counts against the budget
            try:
                units = self._sentence_units(text, idx)
                windows = self._pack(units)
            finally:
                self.size = saved
            for w in windows:
                c = self._make(doc, ordinal, w)
                if ctx:
                    c.metadata["context"] = ctx
                chunks.append(c)
                ordinal += 1
        return chunks


def build_chunker(
    cfg: ChunkingConfig, embedder_factory: Callable[[], Embedder] | None = None
) -> _BaseChunker:
    if cfg.strategy == "fixed":
        return FixedSizeChunker(cfg)
    if cfg.strategy == "recursive":
        return RecursiveChunker(cfg)
    if cfg.strategy == "structure_aware":
        return StructureAwareChunker(cfg)
    if cfg.strategy == "semantic":
        if embedder_factory is None:
            raise ValueError("semantic chunking needs an embedder")
        return SemanticChunker(cfg, embedder_factory())
    raise ValueError(f"unknown chunking strategy {cfg.strategy}")


def chunk_stats(chunks: list[Chunk]) -> dict[str, float]:
    """Summary statistics used by the chunking sweep (Phase 3)."""
    if not chunks:
        return {"n_chunks": 0}
    toks = np.array([c.token_count for c in chunks])
    return {
        "n_chunks": len(chunks),
        "tokens_mean": round(float(toks.mean()), 2),
        "tokens_p50": float(np.percentile(toks, 50)),
        "tokens_p95": float(np.percentile(toks, 95)),
        "tokens_min": int(toks.min()),
        "tokens_max": int(toks.max()),
        "pct_crossing_sections": round(
            100 * sum(bool(c.metadata.get("crosses_section")) for c in chunks) / len(chunks), 2
        ),
        "pct_tiny_lt_32": round(100 * float((toks < 32).mean()), 2),
    }
