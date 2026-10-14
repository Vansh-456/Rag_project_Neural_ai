import numpy as np
import pytest

from agvrag.embeddings.embedders import CachedEmbedder, HashingEmbedder
from agvrag.schemas import Chunk
from agvrag.vector_store.stores import InMemoryVectorStore, QdrantVectorStore

TEXTS = [
    "low rank adaptation of language models",
    "dense passage retrieval with bert",
    "bm25 sparse lexical retrieval",
    "graph based retrieval augmented generation",
    "adapter modules for transfer learning",
]


def _chunks(n=5):
    return [
        Chunk(f"d{i}::c", f"d{i % 2}", t, None, 1, 1, "fixed", 5) for i, t in enumerate(TEXTS[:n])
    ]


def test_hashing_embedder_normalized_and_deterministic():
    e = HashingEmbedder(64)
    v = e.embed_documents(["hello world", "hello world", ""])
    assert v.shape == (3, 64)
    assert np.allclose(np.linalg.norm(v[:2], axis=1), 1.0)
    assert np.array_equal(v[0], v[1])
    assert np.allclose(v[2], 0.0)  # empty text stays a zero vector (no NaN)


def test_cached_embedder_hits(tmp_path):
    e = CachedEmbedder(HashingEmbedder(32), tmp_path / "c.sqlite")
    a = e.embed_documents(["x y", "z"])
    b = e.embed_documents(["x y", "z"])
    assert np.array_equal(a, b)
    assert e.hits == 2 and e.misses == 2 and e.hit_rate == 0.5
    assert e.embed_query("x y").shape == (32,)


@pytest.fixture(params=["memory", "qdrant"])
def store(request):
    if request.param == "memory":
        return InMemoryVectorStore(64)
    return QdrantVectorStore("test", 64)  # in-process ':memory:' Qdrant


def test_store_search_upsert_filter(store):
    e = HashingEmbedder(64)
    chunks = _chunks()
    vecs = e.embed_documents([c.text for c in chunks])
    store.upsert(chunks, vecs)
    store.upsert(chunks, vecs)  # idempotent
    assert store.count() == 5
    hits = store.search(e.embed_query("dense passage retrieval"), 3)
    assert hits[0].chunk.chunk_id == "d1::c"
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)
    only = store.search(e.embed_query("retrieval"), 5, doc_filter=["d0"])
    assert only and all(h.chunk.doc_id == "d0" for h in only)
    assert [c.chunk_id for c in store.get(["d2::c", "missing"])] == ["d2::c"]


def test_memory_store_persistence(tmp_path):
    e = HashingEmbedder(64)
    chunks = _chunks()
    s = InMemoryVectorStore(64)
    s.upsert(chunks, e.embed_documents([c.text for c in chunks]))
    s.save(tmp_path)
    s2 = InMemoryVectorStore.load(tmp_path, {c.chunk_id: c for c in chunks})
    q = e.embed_query("bm25")
    assert [h.chunk.chunk_id for h in s.search(q, 3)] == [h.chunk.chunk_id for h in s2.search(q, 3)]


def test_dimension_mismatch():
    with pytest.raises(ValueError):
        InMemoryVectorStore(8).upsert(_chunks(1), np.zeros((1, 4), dtype=np.float32))
