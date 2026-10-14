"""Embedders: text -> L2-normalized float32 vectors.

* ``SentenceTransformerEmbedder`` — real models (BGE/E5/GTE...). Applies the model-specific
  query/document instruction prefixes (BGE wants a query instruction; E5 wants
  "query: " / "passage: "). Forgetting the prefix is a classic silent quality loss.
* ``HashingEmbedder`` — deterministic feature-hashing bag of unigrams+bigrams. No model
  download; used in tests and as an offline fallback. It is a *lexical* model, so it is
  not a fair stand-in for semantic quality.
* ``CachedEmbedder`` — SQLite cache keyed by (model, prefix, text) hash; reports hit rate.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from agvrag.config import EmbeddingConfig
from agvrag.utils.logging import get_logger

log = get_logger(__name__)


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing (the "hashing trick") over lowercase unigrams and bigrams."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = re.findall(r"[a-z0-9]+", text.lower())
        feats = toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, lazy

        self.cfg = cfg
        self.model_name = cfg.model
        self._model = SentenceTransformer(cfg.model, device=cfg.device)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"cannot determine embedding dim for {cfg.model}")
        self.dim = int(dim)
        self._lock = threading.Lock()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:  # SentenceTransformer.encode is not re-entrant on CPU threads
            out = self._model.encode(
                list(texts),
                batch_size=self.cfg.batch_size,
                normalize_embeddings=False,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        out = np.asarray(out, dtype=np.float32)
        return l2_normalize(out) if self.cfg.normalize else out

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.cfg.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.cfg.query_prefix + text])[0]


class CachedEmbedder:
    """Wraps any embedder with a persistent per-text cache (thread-safe)."""

    def __init__(self, inner: object, cache_path: Path, prefix_key: str = "") -> None:
        self.inner = inner
        self.model_name: str = inner.model_name  # type: ignore[attr-defined]
        self.dim: int = inner.dim  # type: ignore[attr-defined]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self._lock = threading.Lock()
        self._prefix_key = prefix_key
        self.hits = 0
        self.misses = 0

    def _key(self, kind: str, text: str) -> str:
        raw = f"{self.model_name}|{self._prefix_key}|{kind}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _lookup(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                q = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(part))})"
                for k, v in self._db.execute(q, part):
                    found[k] = np.frombuffer(v, dtype=np.float32)
        return found

    def _store(self, items: list[tuple[str, np.ndarray]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)",
                [(k, v.astype(np.float32).tobytes()) for k, v in items],
            )
            self._db.commit()

    def _embed(self, kind: str, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(kind, t) for t in texts]
        found = self._lookup(keys)
        missing = [i for i, k in enumerate(keys) if k not in found]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            if kind == "doc":
                new = self.inner.embed_documents([texts[i] for i in missing])  # type: ignore[attr-defined]
            else:
                new = np.stack([self.inner.embed_query(texts[i]) for i in missing])  # type: ignore[attr-defined]
            self._store([(keys[i], new[j]) for j, i in enumerate(missing)])
            for j, i in enumerate(missing):
                found[keys[i]] = new[j]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed("doc", texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed("query", [text])[0]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def build_embedder(cfg: EmbeddingConfig, cache_dir: Path | None = None) -> object:
    """Factory. Returns a ``CachedEmbedder`` when ``cache_dir`` is given."""
    inner: object
    if cfg.provider == "hashing":
        inner = HashingEmbedder(cfg.hashing_dim)
    else:
        inner = SentenceTransformerEmbedder(cfg)
    if cache_dir is None:
        return inner
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", inner.model_name)
    prefix_key = f"{cfg.query_prefix}|{cfg.document_prefix}|{cfg.normalize}"
    return CachedEmbedder(inner, cache_dir / f"embeddings_{slug}.sqlite", prefix_key)
