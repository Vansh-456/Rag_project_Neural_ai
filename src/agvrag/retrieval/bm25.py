"""BM25 sparse retrieval: a from-scratch implementation (for understanding) and a
``rank-bm25`` wrapper (for cross-checking). Both share the same tokenizer.

Okapi BM25 with the Lucene (non-negative) IDF:

    idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
    score(q, d) = sum_{t in q} idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))

k1 controls term-frequency saturation; b controls document-length normalization.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import snowballstemmer

from agvrag.config import BM25Config
from agvrag.interfaces import ScoredChunk
from agvrag.schemas import Chunk

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "also",
        "however",
        "thus",
        "via",
        "et",
        "al",
        "use",
        "used",
        "using",
        "show",
        "shows",
        "shown",
        "paper",
        "propose",
        "proposed",
    ]
)
# Negations and question words that change meaning are deliberately NOT stopwords.
STOPWORDS = STOPWORDS - {"not", "no", "nor", "who", "which", "how", "what"}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*")


class BM25Tokenizer:
    """Lowercases, optionally stems and drops stopwords.

    Compound/acronym handling: ``GPT-3`` -> ``gpt3``, ``gpt``, ``3``; ``BERT-large`` ->
    ``bertlarge``, ``bert``, ``large``. Emitting both the joined and split forms lets a query
    for "GPT-3" match "GPT3" and "GPT 3" in text.
    """

    def __init__(self, stem: bool = True, remove_stopwords: bool = True) -> None:
        self.stem = stem
        self.remove_stopwords = remove_stopwords
        self._stemmer = snowballstemmer.stemmer("english") if stem else None
        self._cache: dict[str, str] = {}

    def _norm(self, tok: str) -> str:
        if self._stemmer is None or tok.isdigit() or len(tok) <= 3:
            return tok
        cached = self._cache.get(tok)
        if cached is None:
            cached = self._stemmer.stemWord(tok)
            self._cache[tok] = cached
        return cached

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for raw in _TOKEN_RE.findall(text):
            low = raw.lower()
            parts = re.split(r"[-.]", low)
            candidates = [low.replace("-", "").replace(".", "")] if len(parts) > 1 else [low]
            if len(parts) > 1:
                candidates += [p for p in parts if p]
            for t in candidates:
                if self.remove_stopwords and t in STOPWORDS:
                    continue
                out.append(self._norm(t))
        return out


class ScratchBM25:
    """Inverted-index BM25. ``search`` touches only postings of query terms."""

    def __init__(self, chunks: Sequence[Chunk], texts: Sequence[str], cfg: BM25Config) -> None:
        if len(chunks) != len(texts):
            raise ValueError("chunks/texts length mismatch")
        self.cfg = cfg
        self.tokenizer = BM25Tokenizer(cfg.stem, cfg.remove_stopwords)
        self.chunks = list(chunks)
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, text in enumerate(texts):
            toks = self.tokenizer.tokenize(text)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.n_docs = len(self.chunks)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.idf = {
            t: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def scores(self, query: str) -> dict[int, float]:
        k1, b = self.cfg.k1, self.cfg.b
        acc: dict[int, float] = {}
        for term in set(self.tokenizer.tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc, tf in postings:
                norm = k1 * (1 - b + b * self.doc_len[doc] / self.avgdl) if self.avgdl else k1
                acc[doc] = acc.get(doc, 0.0) + idf * tf * (k1 + 1) / (tf + norm)
        return acc

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        acc = self.scores(query)
        top = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [ScoredChunk(self.chunks[i], s) for i, s in top]


class RankBM25Index:
    """Same contract backed by ``rank_bm25.BM25Okapi`` (note: its IDF differs slightly)."""

    def __init__(self, chunks: Sequence[Chunk], texts: Sequence[str], cfg: BM25Config) -> None:
        from rank_bm25 import BM25Okapi

        self.tokenizer = BM25Tokenizer(cfg.stem, cfg.remove_stopwords)
        self.chunks = list(chunks)
        corpus = [self.tokenizer.tokenize(t) or ["<empty>"] for t in texts]
        self._bm25 = BM25Okapi(corpus, k1=cfg.k1, b=cfg.b)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        q = self.tokenizer.tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def build_bm25(chunks: Sequence[Chunk], texts: Sequence[str], cfg: BM25Config) -> object:
    if cfg.implementation == "rank_bm25":
        return RankBM25Index(chunks, texts, cfg)
    return ScratchBM25(chunks, texts, cfg)
