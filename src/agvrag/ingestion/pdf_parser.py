"""PDF -> structured ``Document``.

Two parsers behind the ``Parser`` protocol so they can be compared (Phase 2 experiment):

* ``PyMuPDFLayoutParser`` — layout-aware: uses span font sizes / bold flags to find headings,
  orders two-column pages column-by-column, strips running headers/footers and arXiv stamps.
* ``PyMuPDFPlainParser`` — baseline: plain page text in reading order + regex headings.

Both fill title/authors/year from supplied arXiv metadata when available, because metadata
is far more reliable than visual title detection.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf as fitz

from agvrag.ingestion.cleaning import (
    _ARXIV_STAMP_RE,
    collapse_whitespace,
    find_repeated_margin_lines,
    garbled_score,
    is_noise_line,
    normalize_chars,
    reflow_paragraphs,
)
from agvrag.schemas import Document, Section
from agvrag.utils.logging import get_logger

log = get_logger(__name__)

KNOWN_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "related works",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "models",
    "experiments",
    "experimental setup",
    "experimental results",
    "results",
    "evaluation",
    "discussion",
    "analysis",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "acknowledgements",
    "acknowledgments",
    "references",
    "bibliography",
    "appendix",
    "preliminaries",
    "ethics statement",
    "broader impact",
    "conclusion and future work",
    "conclusions and future work",
}
REFERENCE_HEADINGS = {"references", "bibliography", "reference"}
_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>(?:[1-9]\d?|[A-H])(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][^\n]{1,90})$"
)
_NUMBER_ONLY_RE = re.compile(r"^(?:[1-9]\d?|[A-H]|[IVX]{1,4})(?:\.\d{1,2}){0,3}\.?$")
_BOLD_FONT_RE = re.compile(r"bold|medi|semibold|black|heavy|\.B$|-B$|CMBX", re.I)
_ROMAN_HEADING_RE = re.compile(r"^(?P<num>[IVX]{1,4})\.\s+(?P<title>[A-Z][A-Za-z \-:&]{2,80})$")


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    page: int
    y0: float
    y1: float


def heading_level(text: str) -> int | None:
    """Return the heading level implied by the text pattern, or None if not heading-like."""
    s = text.strip()
    low = s.lower().rstrip(":").strip()
    if low in KNOWN_SECTIONS:
        return 1
    m = _NUMBERED_HEADING_RE.match(s)
    if m:
        title = m.group("title")
        if title.endswith(".") or len(title.split()) > 12 or re.search(r"\d{3,}", title):
            return None
        if sum(c.isalpha() for c in title) < 3:
            return None
        return m.group("num").count(".") + 1
    m = _ROMAN_HEADING_RE.match(s)
    if m:
        return 1
    return None


def split_references(text: str) -> list[str]:
    """Split a references section into entries (``[n]``-style or author-year style)."""
    text = text.strip()
    if not text:
        return []
    if re.search(r"(^|\n)\s*\[\d{1,3}\]", text):
        parts = re.split(r"(?:^|\n)\s*\[\d{1,3}\]\s*", text)
    else:
        # Author-year: a new entry begins on a line starting with "Surname, X." / "Surname X"
        # after a line that ended with a period.
        parts = re.split(r"(?<=[.\d])\n(?=[A-Z][A-Za-z'`\-À-ſ]+,?\s+[A-Z])", text)
    refs = [re.sub(r"\s+", " ", p).strip() for p in parts]
    return [r for r in refs if len(r) > 25]


class _BaseParser:
    def __init__(self, min_section_chars: int = 40, max_heading_words: int = 12) -> None:
        self.min_section_chars = min_section_chars
        self.max_heading_words = max_heading_words

    def _finalize(
        self,
        path: Path,
        raw_sections: list[tuple[str, int, list[str], int, int]],
        fallback_title: str,
        metadata: dict[str, Any] | None,
        parser_name: str,
    ) -> Document:
        meta = dict(metadata or {})
        sections: list[Section] = []
        references: list[str] = []
        abstract = ""
        in_refs = False
        for heading, level, lines, p0, p1 in raw_sections:
            raw_text = "\n".join(lines)
            low = heading.lower().strip().rstrip(":")
            if low not in KNOWN_SECTIONS:
                low = re.sub(r"^(?:\d+(?:\.\d+)*|[a-h](?:\.\d+)*|[ivx]{1,4})\.?\s+", "", low)
            if low in REFERENCE_HEADINGS:
                in_refs = True
                references.extend(split_references(collapse_whitespace(raw_text)))
                continue
            if in_refs and level > 1 and not re.match(r"^[A-H](\.\d+)*\s", heading):
                # sub-headings inside references are almost always false positives
                references.extend(split_references(collapse_whitespace(raw_text)))
                continue
            if in_refs and (re.match(r"^[A-H](\.\d+)*\s", heading) or low.startswith("appendix")):
                in_refs = False
            text = reflow_paragraphs(collapse_whitespace(raw_text))
            if low == "abstract" and not abstract:
                abstract = text
            if len(text) < self.min_section_chars:
                if sections and text:
                    sections[-1].text = f"{sections[-1].text}\n\n{heading}\n{text}".strip()
                continue
            sections.append(
                Section(
                    heading=heading, level=level, text=text, page_start=p0, page_end=max(p0, p1)
                )
            )
        if not abstract and meta.get("abstract"):
            abstract = str(meta["abstract"]).strip()
        if not abstract and sections:
            abstract = sections[0].text[:1500]
        doc_id = str(meta.get("arxiv_id") or meta.get("doc_id") or path.stem)
        year = meta.get("year")
        doc = Document(
            doc_id=doc_id,
            title=str(meta.get("title") or fallback_title or path.stem),
            authors=list(meta.get("authors") or []),
            year=int(year) if year else None,
            sections=sections,
            references=references,
            source_path=str(path),
            abstract=abstract,
            metadata={k: v for k, v in meta.items() if k not in {"title", "authors", "year"}}
            | {"parser": parser_name},
        )
        return doc


class PyMuPDFLayoutParser(_BaseParser):
    """Layout-aware parser. See module docstring."""

    name = "pymupdf_layout"

    def parse(self, path: Path, metadata: dict[str, Any] | None = None) -> Document:
        try:
            pdf = fitz.open(path)
        except Exception as exc:  # fitz raises several private exception types
            raise ValueError(f"cannot open PDF {path}: {exc}") from exc
        with pdf:
            lines = self._extract_lines(pdf)
        if not lines:
            raise ValueError(f"no extractable text in {path}")
        body_size = self._body_font_size(lines)
        title = self._guess_title(lines)
        raw_sections = self._segment(lines, body_size)
        return self._finalize(path, raw_sections, title, metadata, self.name)

    # -- extraction ---------------------------------------------------------------------
    def _extract_lines(self, pdf: fitz.Document) -> list[_Line]:
        pages_lines: list[list[_Line]] = []
        page: Any
        for pno, page in enumerate(pdf, start=1):  # type: ignore[arg-type]
            width, height = page.rect.width, page.rect.height
            d = page.get_text("dict")
            blocks = [b for b in d.get("blocks", []) if b.get("type") == 0]
            ordered = self._column_order(blocks, width)
            page_lines: list[_Line] = []
            for b in ordered:
                for ln in b.get("lines", []):
                    if tuple(round(x) for x in ln.get("dir", (1, 0))) != (1, 0):
                        continue  # rotated text: arXiv side stamp, axis labels
                    spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = normalize_chars("".join(s["text"] for s in ln["spans"])).strip()
                    if not text:
                        continue
                    y0, y1 = ln["bbox"][1], ln["bbox"][3]
                    in_margin = y1 < 0.05 * height or y0 > 0.95 * height
                    if (
                        in_margin
                        and len(text) < 100
                        and (is_noise_line(text) or len(text.split()) <= 8)
                    ):
                        continue
                    if _ARXIV_STAMP_RE.search(text):
                        continue
                    chars = sum(len(s["text"]) for s in spans)
                    size = sum(s["size"] * len(s["text"]) for s in spans) / max(1, chars)
                    bold = (
                        sum(
                            len(s["text"])
                            for s in spans
                            if s.get("flags", 0) & 16 or _BOLD_FONT_RE.search(s.get("font", ""))
                        )
                        > 0.6 * chars
                    )
                    prev = page_lines[-1] if page_lines else None
                    if (
                        prev is not None
                        and _NUMBER_ONLY_RE.match(prev.text)
                        and abs(prev.y0 - y0) < 3
                        and len(text) < 100
                    ):
                        # "1" | "Introduction" laid out as two lines on the same baseline
                        page_lines[-1] = _Line(
                            f"{prev.text} {text}",
                            max(prev.size, round(size, 1)),
                            bold or prev.bold,
                            pno,
                            prev.y0,
                            y1,
                        )
                        continue
                    page_lines.append(_Line(text, round(size, 1), bold, pno, y0, y1))
            pages_lines.append(page_lines)
        repeated = find_repeated_margin_lines([[ln.text for ln in pl] for pl in pages_lines])
        out: list[_Line] = []
        for pl in pages_lines:
            for ln in pl:
                if re.sub(r"\d+", "#", ln.text.strip()) in repeated:
                    continue
                out.append(ln)
        return out

    @staticmethod
    def _column_order(blocks: list[dict[str, Any]], width: float) -> list[dict[str, Any]]:
        """Order blocks for 1- or 2-column layouts.

        Full-width blocks (> 55% of page width, or straddling the midline — e.g. a centered
        abstract on a two-column page) split the page into bands; inside each band the left
        column is read top-to-bottom before the right column.
        """
        mid = width / 2
        blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        ordered: list[dict[str, Any]] = []
        band_left: list[dict[str, Any]] = []
        band_right: list[dict[str, Any]] = []

        def flush() -> None:
            ordered.extend(sorted(band_left, key=lambda b: b["bbox"][1]))
            ordered.extend(sorted(band_right, key=lambda b: b["bbox"][1]))
            band_left.clear()
            band_right.clear()

        for b in blocks:
            x0, _, x1, _ = b["bbox"]
            straddles = x0 < mid - 0.04 * width and x1 > mid + 0.04 * width
            if (x1 - x0) > 0.55 * width or straddles:
                flush()
                ordered.append(b)
            elif (x0 + x1) / 2 < mid:
                band_left.append(b)
            else:
                band_right.append(b)
        flush()
        return ordered

    @staticmethod
    def _body_font_size(lines: list[_Line]) -> float:
        weights: dict[float, int] = {}
        for ln in lines:
            weights[ln.size] = weights.get(ln.size, 0) + len(ln.text)
        return max(weights.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _guess_title(lines: list[_Line]) -> str:
        first_page = [ln for ln in lines if ln.page == 1][:40]
        if not first_page:
            return ""
        biggest = max(ln.size for ln in first_page)
        title_lines = [ln.text for ln in first_page if ln.size >= biggest - 0.5][:3]
        return " ".join(title_lines).strip()

    def _is_heading(self, ln: _Line, body_size: float) -> int | None:
        words = ln.text.split()
        if not words or len(words) > self.max_heading_words or len(ln.text) > 100:
            return None
        level = heading_level(ln.text)
        if level is None:
            return None
        title_part = re.sub(r"^[\dA-HIVX.]+\s+", "", ln.text.strip())
        all_caps = len(title_part) >= 4 and title_part.isupper()
        visually_distinct = ln.bold or ln.size >= body_size + 0.8 or all_caps
        low = ln.text.lower().strip().rstrip(":")
        if low in KNOWN_SECTIONS:
            # Named sections must still look like headings (avoids "Results" inside a sentence).
            return level if (visually_distinct or ln.text.isupper()) else None
        return level if visually_distinct else None

    def _segment(
        self, lines: list[_Line], body_size: float
    ) -> list[tuple[str, int, list[str], int, int]]:
        sections: list[tuple[str, int, list[str], int, int]] = []
        buf: list[str] = []
        heading, level, p0 = "Front matter", 1, lines[0].page
        last_page = p0
        for ln in lines:
            lvl = self._is_heading(ln, body_size)
            if lvl is not None:
                sections.append((heading, level, buf, p0, last_page))
                heading, level, buf, p0 = ln.text.strip().rstrip(":"), lvl, [], ln.page
            else:
                buf.append(ln.text)
            last_page = ln.page
        sections.append((heading, level, buf, p0, last_page))
        return sections


class PyMuPDFPlainParser(_BaseParser):
    """Baseline parser: page text in PyMuPDF's default order, regex-only heading detection."""

    name = "pymupdf_plain"

    def parse(self, path: Path, metadata: dict[str, Any] | None = None) -> Document:
        try:
            pdf = fitz.open(path)
        except Exception as exc:
            raise ValueError(f"cannot open PDF {path}: {exc}") from exc
        sections: list[tuple[str, int, list[str], int, int]] = []
        buf: list[str] = []
        heading, level, p0, last = "Front matter", 1, 1, 1
        title = ""
        with pdf:
            page: Any
            for pno, page in enumerate(pdf, start=1):  # type: ignore[arg-type]
                text = normalize_chars(page.get_text("text"))
                for line in text.splitlines():
                    s = line.strip()
                    if not s or is_noise_line(s):
                        continue
                    if pno == 1 and not title:
                        title = s
                    lvl = heading_level(s) if len(s.split()) <= self.max_heading_words else None
                    if lvl is not None:
                        sections.append((heading, level, buf, p0, last))
                        heading, level, buf, p0 = s.rstrip(":"), lvl, [], pno
                    else:
                        buf.append(s)
                    last = pno
        sections.append((heading, level, buf, p0, last))
        if not any(b for _, _, b, _, _ in sections):
            raise ValueError(f"no extractable text in {path}")
        return self._finalize(path, sections, title, metadata, self.name)


def get_parser(name: str, min_section_chars: int = 40, max_heading_words: int = 12) -> Any:
    if name == "pymupdf":
        return PyMuPDFLayoutParser(min_section_chars, max_heading_words)
    if name == "pymupdf_plain":
        return PyMuPDFPlainParser(min_section_chars, max_heading_words)
    raise ValueError(f"unknown parser: {name}")


def parse_quality(doc: Document) -> dict[str, Any]:
    """Per-document parse quality metrics used by the parse-quality report."""
    lengths = [len(s.text) for s in doc.sections]
    headings = [s.heading.lower() for s in doc.sections]
    has_intro = any("introduction" in h for h in headings)
    return {
        "doc_id": doc.doc_id,
        "title": doc.title[:100],
        "parser": doc.metadata.get("parser"),
        "n_sections": len(doc.sections),
        "n_empty_sections": sum(1 for n in lengths if n < 100),
        "median_section_chars": int(statistics.median(lengths)) if lengths else 0,
        "n_references": len(doc.references),
        "has_abstract": bool(doc.abstract),
        "has_introduction": has_intro,
        "sections_identified": len(doc.sections) >= 3 and has_intro,
        "garbled_score": garbled_score(doc.full_text),
        "total_chars": sum(lengths),
        "pages": max((s.page_end for s in doc.sections), default=0),
    }
