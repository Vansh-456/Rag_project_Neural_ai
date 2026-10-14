import itertools

import pytest

from agvrag.chunking.chunkers import build_chunker, chunk_embed_text, chunk_stats
from agvrag.config import ChunkingConfig
from agvrag.embeddings.embedders import HashingEmbedder
from agvrag.utils.text import count_tokens

STRATEGIES = ["fixed", "recursive", "semantic", "structure_aware"]


def _chunker(strategy, size=40, overlap=8):
    return build_chunker(
        ChunkingConfig(strategy=strategy, chunk_size=size, overlap=overlap),
        lambda: HashingEmbedder(128),
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_invariants(strategy, corpus):
    ch = _chunker(strategy)
    for doc in corpus:
        chunks = ch.chunk(doc)
        assert chunks, "no chunks produced"
        assert all(c.text.strip() for c in chunks), "empty chunk"
        assert all(count_tokens(c.text) <= 42 for c in chunks), "chunk over budget"
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "duplicate ids"
        assert all(c.doc_id == doc.doc_id and c.strategy == strategy for c in chunks)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_deterministic_ids(strategy, corpus):
    a = [c.chunk_id for c in _chunker(strategy).chunk(corpus[0])]
    b = [c.chunk_id for c in _chunker(strategy).chunk(corpus[0])]
    assert a == b


def test_ids_change_with_params(corpus):
    assert (
        _chunker("fixed", 40, 8).chunk(corpus[0])[0].chunk_id
        != _chunker("fixed", 48, 8).chunk(corpus[0])[0].chunk_id
    )


def test_fixed_overlap_is_respected(corpus):
    chunks = _chunker("fixed", 30, 10).chunk(corpus[0])
    assert len(chunks) > 2
    for a, b in itertools.pairwise(chunks):
        assert " ".join(a.text.split()[-2:]) in b.text  # consecutive windows share words


def test_fixed_zero_overlap_partitions_text(corpus):
    chunks = _chunker("fixed", 30, 0).chunk(corpus[0])
    joined = " ".join(c.text for c in chunks).split()
    original = " ".join(s.text for s in corpus[0].sections).split()
    assert joined == original


def test_structure_aware_never_crosses_sections(corpus):
    for doc in corpus:
        for c in _chunker("structure_aware").chunk(doc):
            assert c.metadata["crosses_section"] is False
            assert c.metadata["context"].startswith(doc.title)
            assert chunk_embed_text(c).startswith(doc.title)


def test_invalid_params():
    with pytest.raises(ValueError):
        _chunker("fixed", 32, 32)
    with pytest.raises(ValueError):
        build_chunker(ChunkingConfig(strategy="semantic"), None)


def test_chunk_stats(corpus):
    chunks = [c for d in corpus for c in _chunker("recursive").chunk(d)]
    st = chunk_stats(chunks)
    assert st["n_chunks"] == len(chunks)
    assert st["tokens_max"] <= 42
