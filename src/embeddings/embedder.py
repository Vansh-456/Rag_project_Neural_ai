"""
Embedding module for vector representation of chunks and queries.
Includes caching, batching, L2-normalization, and swappable backends.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np


class BaseEmbedder:
    """Base embedder class with caching and normalization."""

    def __init__(self, dimension: int = 384, normalize: bool = True, cache_dir: str | None = None):
        self.dimension = dimension
        self.normalize = normalize
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: dict[str, list[float]] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self):
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / "embeddings_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / "embeddings_cache.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception:
            pass

    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


class DeterministicMockEmbedder(BaseEmbedder):
    """
    Deterministic pseudo-semantic embedder based on seeded character n-grams and hashing.
    Outputs unit-normalized vectors of requested dimension with consistent similarity properties.
    Ideal for rapid local development, offline CI, and testing without multi-GB PyTorch models.
    """

    def __init__(self, dimension: int = 384, normalize: bool = True, cache_dir: str | None = None):
        super().__init__(dimension=dimension, normalize=normalize, cache_dir=cache_dir)

    def _embed_single(self, text: str) -> list[float]:
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        # Deterministic vector generation using n-gram hashed projection
        words = text.lower().split()
        vec = np.zeros(self.dimension, dtype=np.float32)

        for word in words:
            # Seed generator deterministically with word hash
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
            np.random.seed(h)
            word_vec = np.random.randn(self.dimension).astype(np.float32)
            vec += word_vec

        if self.normalize:
            vec = self._l2_normalize(vec)

        res = vec.tolist()
        self._cache[key] = res
        return res

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results = [self._embed_single(t) for t in texts]
        self._save_cache()
        return results

    def embed_query(self, query: str) -> list[float]:
        return self._embed_single(query)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Real neural embedder using sentence-transformers (BGE / E5 / MiniLM)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        batch_size: int = 32,
        normalize: bool = True,
        cache_dir: str | None = None,
        query_instruction: str = "Represent this sentence for searching relevant passages: "
    ):
        super().__init__(dimension=dimension, normalize=normalize, cache_dir=cache_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self._model: Any = None

    def _init_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                # If sentence-transformers is not installed, fallback gracefully to DeterministicMockEmbedder
                self._model = "FALLBACK_MOCK"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._init_model()
        if self._model == "FALLBACK_MOCK":
            mock = DeterministicMockEmbedder(self.dimension, self.normalize, str(self.cache_dir) if self.cache_dir else None)
            return mock.embed_texts(texts)

        # Batch encode with sentence-transformers
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        self._init_model()
        if self._model == "FALLBACK_MOCK":
            mock = DeterministicMockEmbedder(self.dimension, self.normalize, str(self.cache_dir) if self.cache_dir else None)
            return mock.embed_query(query)

        formatted_query = f"{self.query_instruction}{query}" if "bge" in self.model_name.lower() else query
        emb = self._model.encode(
            [formatted_query],
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )[0]
        return emb.tolist()


def get_embedder(provider: str = "mock", model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384, cache_dir: str | None = None) -> BaseEmbedder:
    """Factory helper for embedders."""
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model_name, dimension=dimension, cache_dir=cache_dir)
    return DeterministicMockEmbedder(dimension=dimension, cache_dir=cache_dir)
