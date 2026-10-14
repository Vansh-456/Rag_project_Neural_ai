"""
Document parser module for academic papers.
Supports extracting structured Document objects from text, markdown, or PDF files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from src.schemas import Document, Section
from src.ingestion.cleaner import clean_text, calculate_garble_score


class DocumentParser:
    """Extracts structured Document objects from file inputs with section hierarchy detection."""

    def __init__(self):
        self.heading_patterns = [
            re.compile(r"^(#+)\s+(.+)$", re.MULTILINE),  # Markdown headings
            re.compile(r"^(\d+\.?\d*)\s+([A-Z][A-Za-z0-9\s,\-:]{2,60})$", re.MULTILINE),  # 1. Introduction
            re.compile(r"^(Abstract|Introduction|Related Work|Methodology|Method|Architecture|Experiments|Evaluation|Results|Discussion|Conclusion|References)$", re.MULTILINE | re.IGNORECASE),
        ]

    def parse_text(
        self,
        text: str,
        doc_id: str,
        title: str = "",
        authors: list[str] | None = None,
        year: int | None = None,
        source_path: str = ""
    ) -> Document:
        """Parses a raw text string into a structured Document with sections."""
        cleaned = clean_text(text)
        lines = cleaned.split("\n")

        # Fallback title if not provided
        if not title:
            first_line = lines[0].strip().lstrip("#").strip() if lines else doc_id
            title = first_line[:120]

        # Extract abstract and sections
        sections: list[Section] = []
        current_heading = "Overview"
        current_level = 1
        current_lines: list[str] = []
        current_page = 1
        abstract_text = ""
        references: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for page markers e.g. [Page 2] or --- Page 2 ---
            page_match = re.search(r"\[Page\s+(\d+)\]|---\s*Page\s+(\d+)\s*---", line, re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1) or page_match.group(2))
                i += 1
                continue

            # Check if line matches a section heading
            heading_matched = False
            for pat in self.heading_patterns:
                m = pat.match(line)
                if m:
                    # Flush current section
                    if current_lines:
                        body = "\n".join(current_lines).strip()
                        if current_heading.lower() == "abstract":
                            abstract_text = body
                        elif "reference" in current_heading.lower():
                            references.extend([ref.strip() for ref in body.split("\n") if ref.strip()])
                        else:
                            sections.append(Section(
                                heading=current_heading,
                                level=current_level,
                                text=body,
                                page_start=current_page,
                                page_end=current_page
                            ))
                        current_lines = []

                    current_heading = m.group(m.lastindex).strip() if m.lastindex else line
                    current_level = len(m.group(1)) if m.group(1).startswith("#") else 1
                    heading_matched = True
                    break

            if not heading_matched:
                current_lines.append(line)

            i += 1

        # Flush final section
        if current_lines:
            body = "\n".join(current_lines).strip()
            if current_heading.lower() == "abstract":
                abstract_text = body
            elif "reference" in current_heading.lower():
                references.extend([ref.strip() for ref in body.split("\n") if ref.strip()])
            else:
                sections.append(Section(
                    heading=current_heading,
                    level=current_level,
                    text=body,
                    page_start=current_page,
                    page_end=current_page
                ))

        # If no sections were identified, create a default full-text section
        if not sections and not abstract_text:
            sections.append(Section(
                heading="Content",
                level=1,
                text=cleaned,
                page_start=1,
                page_end=current_page
            ))

        return Document(
            doc_id=doc_id,
            title=title,
            authors=authors or [],
            year=year,
            sections=sections,
            references=references,
            source_path=source_path,
            abstract=abstract_text,
            metadata={"garble_score": calculate_garble_score(cleaned)}
        )

    def parse_file(self, file_path: str | Path) -> Document:
        """Parses a document file (.txt, .md, or .json)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return self.parse_text(
                    text=data.get("text", "") or data.get("content", ""),
                    doc_id=data.get("doc_id", path.stem),
                    title=data.get("title", ""),
                    authors=data.get("authors", []),
                    year=data.get("year"),
                    source_path=str(path)
                )

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return self.parse_text(
            text=content,
            doc_id=path.stem,
            source_path=str(path)
        )


def generate_parse_quality_report(documents: list[Document]) -> dict[str, Any]:
    """Generates an inspection report detailing parse quality across the corpus."""
    total = len(documents)
    empty_sections_count = 0
    garbled_count = 0
    total_sections = 0

    per_paper_report = []
    for doc in documents:
        doc_empty = sum(1 for s in doc.sections if not s.text.strip())
        empty_sections_count += doc_empty
        total_sections += len(doc.sections)
        garble = doc.metadata.get("garble_score", 0.0)
        if garble > 0.05:
            garbled_count += 1

        per_paper_report.append({
            "doc_id": doc.doc_id,
            "title": doc.title,
            "section_count": len(doc.sections),
            "empty_sections": doc_empty,
            "garble_score": garble,
            "status": "PASS" if len(doc.sections) > 0 and doc_empty == 0 else "WARNING"
        })

    return {
        "total_documents": total,
        "total_sections": total_sections,
        "empty_sections": empty_sections_count,
        "high_garble_documents": garbled_count,
        "parse_success_rate": round((total - garbled_count) / max(1, total) * 100, 2),
        "details": per_paper_report
    }
