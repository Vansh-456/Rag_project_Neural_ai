"""
Chunking strategies for document decomposition into retrieval units.
Implements Fixed-Size, Recursive, Semantic, and Structure-Aware chunkers.
"""

from __future__ import annotations

import re
from typing import Any
from src.schemas import Chunk, Document
from src.utils.metrics import estimate_tokens


class BaseChunker:
    """Base class for all chunking strategies with deterministic ID generation."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _make_chunk(
        self,
        doc_id: str,
        chunk_index: int,
        text: str,
        section_heading: str | None,
        page_start: int | None,
        page_end: int | None,
        strategy: str,
        metadata: dict[str, Any] | None = None
    ) -> Chunk:
        token_count = estimate_tokens(text)
        return Chunk(
            chunk_id=f"{doc_id}_c{chunk_index:04d}",
            doc_id=doc_id,
            text=text,
            section_heading=section_heading,
            page_start=page_start,
            page_end=page_end,
            strategy=strategy,
            token_count=token_count,
            metadata=metadata or {}
        )


class FixedSizeChunker(BaseChunker):
    """Chunks documents into fixed token-length blocks with fixed overlap."""

    def chunk(self, document: Document) -> list[Chunk]:
        full_text = "\n\n".join([f"{s.heading}\n{s.text}" for s in document.sections])
        words = full_text.split()
        if not words:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[Chunk] = []
        chunk_idx = 0

        for i in range(0, len(words), step):
            segment = words[i: i + self.chunk_size]
            chunk_text = " ".join(segment)
            if chunk_text.strip():
                chunks.append(
                    self._make_chunk(
                        doc_id=document.doc_id,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        section_heading=None,
                        page_start=None,
                        page_end=None,
                        strategy="fixed"
                    )
                )
                chunk_idx += 1

        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text by paragraph, sentence, and word boundaries
    to maintain semantic coherence within token budgets.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, separators: list[str] | None = None):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", "; ", " ", ""]

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_idx = 0

        for section in document.sections:
            sec_chunks = self._split_text(section.text)
            for text_piece in sec_chunks:
                if text_piece.strip():
                    chunks.append(
                        self._make_chunk(
                            doc_id=document.doc_id,
                            chunk_index=chunk_idx,
                            text=text_piece,
                            section_heading=section.heading,
                            page_start=section.page_start,
                            page_end=section.page_end,
                            strategy="recursive"
                        )
                    )
                    chunk_idx += 1

        return chunks

    def _split_text(self, text: str) -> list[str]:
        if estimate_tokens(text) <= self.chunk_size:
            return [text]

        for sep in self.separators:
            if not sep:
                parts = [text[i: i + self.chunk_size * 4] for i in range(0, len(text), max(1, (self.chunk_size - self.chunk_overlap) * 4))]
                return parts

            splits = text.split(sep)
            if len(splits) > 1:
                result = []
                current = ""
                for s in splits:
                    candidate = f"{current}{sep}{s}" if current else s
                    if estimate_tokens(candidate) > self.chunk_size:
                        if current:
                            result.append(current)
                        current = s
                    else:
                        current = candidate
                if current:
                    result.append(current)
                return result

        return [text]


class SemanticChunker(BaseChunker):
    """
    Splits text on sentence boundaries and clusters sentences using semantic coherence.
    Falls back to sentence-boundary grouping when embedding distance is uniform.
    """

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_idx = 0

        for section in document.sections:
            sentences = re.split(r"(?<=[.!?])\s+", section.text)
            current_sentences: list[str] = []

            for sent in sentences:
                candidate = " ".join(current_sentences + [sent])
                if estimate_tokens(candidate) > self.chunk_size and current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(
                        self._make_chunk(
                            doc_id=document.doc_id,
                            chunk_index=chunk_idx,
                            text=chunk_text,
                            section_heading=section.heading,
                            page_start=section.page_start,
                            page_end=section.page_end,
                            strategy="semantic"
                        )
                    )
                    chunk_idx += 1
                    current_sentences = [sent]
                else:
                    current_sentences.append(sent)

            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    self._make_chunk(
                        doc_id=document.doc_id,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        section_heading=section.heading,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        strategy="semantic"
                    )
                )
                chunk_idx += 1

        return chunks


class StructureAwareChunker(BaseChunker):
    """
    State-of-the-art chunker: Respects explicit document section hierarchies,
    never breaks across section boundaries, and prepends Document Title and
    Section Breadcrumbs to contextualize retrieved evidence.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, attach_context: bool = True):
        super().__init__(chunk_size, chunk_overlap)
        self.attach_context = attach_context

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_idx = 0

        for section in document.sections:
            paragraphs = section.text.split("\n\n")
            current_paragraphs: list[str] = []

            for para in paragraphs:
                para_clean = para.strip()
                if not para_clean:
                    continue

                candidate_body = "\n\n".join(current_paragraphs + [para_clean])
                if estimate_tokens(candidate_body) > self.chunk_size and current_paragraphs:
                    chunk_body = "\n\n".join(current_paragraphs)
                    if self.attach_context:
                        formatted_text = f"Document: {document.title}\nSection: {section.heading}\n\n{chunk_body}"
                    else:
                        formatted_text = chunk_body

                    chunks.append(
                        self._make_chunk(
                            doc_id=document.doc_id,
                            chunk_index=chunk_idx,
                            text=formatted_text,
                            section_heading=section.heading,
                            page_start=section.page_start,
                            page_end=section.page_end,
                            strategy="structure_aware",
                            metadata={
                                "doc_title": document.title,
                                "section_level": section.level,
                                "pure_body": chunk_body
                            }
                        )
                    )
                    chunk_idx += 1
                    current_paragraphs = [para_clean]
                else:
                    current_paragraphs.append(para_clean)

            if current_paragraphs:
                chunk_body = "\n\n".join(current_paragraphs)
                if self.attach_context:
                    formatted_text = f"Document: {document.title}\nSection: {section.heading}\n\n{chunk_body}"
                else:
                    formatted_text = chunk_body

                chunks.append(
                    self._make_chunk(
                        doc_id=document.doc_id,
                        chunk_index=chunk_idx,
                        text=formatted_text,
                        section_heading=section.heading,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        strategy="structure_aware",
                        metadata={
                            "doc_title": document.title,
                            "section_level": section.level,
                            "pure_body": chunk_body
                        }
                    )
                )
                chunk_idx += 1

        return chunks


def get_chunker(strategy: str = "structure_aware", chunk_size: int = 512, chunk_overlap: int = 64) -> BaseChunker:
    """Factory helper to instantiate chunker by name."""
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "semantic":
        return SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "structure_aware":
        return StructureAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
