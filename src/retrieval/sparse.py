"""
BM25 sparse lexical retrieval implementation using rank-bm25.
Features customizable tokenization, stopword filtering, and acronym preservation.
"""

from __future__ import annotations

import re
from rank_bm25 import BM25Okapi
from src.schemas import Chunk


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


class BM25Index:
    """In-memory BM25 lexical index with tokenization preprocessing."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, lowercase: bool = True, remove_stopwords: bool = True):
        self.k1 = k1
        self.b = b
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        # Tokenize words, preserving hyphenated terms and alphanumeric acronyms
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return tokens

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Builds the BM25 index over the provided text chunks."""
        self.chunks = list(chunks)
        tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        if tokenized_corpus:
            self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Queries the BM25 index and returns top-K chunks with raw BM25 scores."""
        if not self._bm25 or not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[tuple[Chunk, float]] = []
        for idx in top_indices:
            if scores[idx] > 0.0:  # Only return documents that had lexical overlap
                results.append((self.chunks[idx], float(scores[idx])))

        return results
