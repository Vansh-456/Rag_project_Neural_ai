"""
Unit tests for chunking strategies: Fixed, Recursive, Semantic, and Structure-Aware.
"""

from src.schemas import Document, Section
from src.chunking.chunker import (
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    StructureAwareChunker,
    get_chunker,
)


def sample_document() -> Document:
    return Document(
        doc_id="test_doc_01",
        title="Sample Academic Paper",
        authors=["Alice Smith", "Bob Jones"],
        year=2024,
        sections=[
            Section(
                heading="Introduction",
                level=1,
                text="Attention mechanisms revolutionized natural language processing. " * 15,
                page_start=1,
                page_end=1
            ),
            Section(
                heading="Methodology",
                level=1,
                text="We propose an adaptive retrieval algorithm. It integrates vector search with graph traversal. " * 15,
                page_start=2,
                page_end=2
            )
        ],
        references=["Vaswani et al., 2017"],
        source_path="test.pdf"
    )


def test_fixed_size_chunker():
    doc = sample_document()
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 0
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"test_doc_01_c{idx:04d}"
        assert c.strategy == "fixed"
        assert len(c.text.strip()) > 0


def test_recursive_chunker():
    doc = sample_document()
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=15)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 0
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"test_doc_01_c{idx:04d}"
        assert c.strategy == "recursive"


def test_semantic_chunker():
    doc = sample_document()
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 0
    for c in chunks:
        assert c.strategy == "semantic"


def test_structure_aware_chunker():
    doc = sample_document()
    chunker = StructureAwareChunker(chunk_size=150, chunk_overlap=20, attach_context=True)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 0
    for c in chunks:
        assert c.strategy == "structure_aware"
        # Must attach document title and section heading
        assert "Document: Sample Academic Paper" in c.text
        assert "Section: " in c.text
        assert c.section_heading in ["Introduction", "Methodology"]


def test_chunker_factory():
    fixed = get_chunker("fixed")
    assert isinstance(fixed, FixedSizeChunker)

    rec = get_chunker("recursive")
    assert isinstance(rec, RecursiveChunker)

    sem = get_chunker("semantic")
    assert isinstance(sem, SemanticChunker)

    struct = get_chunker("structure_aware")
    assert isinstance(struct, StructureAwareChunker)
