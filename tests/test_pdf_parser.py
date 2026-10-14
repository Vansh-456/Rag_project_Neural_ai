"""Parser tests on PDFs generated at test time (two-column layout, split heading numbers,
running headers, hyphenation, references)."""

from pathlib import Path

import pymupdf as fitz
import pytest

from agvrag.ingestion.pdf_parser import (
    PyMuPDFLayoutParser,
    PyMuPDFPlainParser,
    heading_level,
    parse_quality,
    split_references,
)

LEFT = (
    "Large models are expensive to adapt. Fine-tuning updates every parameter, which is costly. "
    "We study cheaper alternatives that keep quality high across tasks."
)
RIGHT = (
    "Our method freezes the backbone and trains small low-rank matrices. This reduces memory "
    "and storage while matching the accuracy of full fine-tuning on standard benchmarks."
)


def _two_column_pdf(path: Path) -> Path:
    doc = fitz.open()
    for pno in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((200, 30), "Preprint under review at a venue", fontsize=8)
        if pno == 0:
            page.insert_text((150, 80), "A Study of Cheap Adaptation", fontsize=18, fontname="hebo")
            page.insert_text((280, 120), "Abstract", fontsize=12, fontname="hebo")
            page.insert_textbox(
                fitz.Rect(150, 130, 460, 200),
                "We study adaptation of large language models with few trainable parameters.",
                fontsize=10,
            )
            page.insert_text((72, 230), "1", fontsize=12, fontname="hebo")
            page.insert_text((90, 230), "Introduction", fontsize=12, fontname="hebo")
            page.insert_textbox(
                fitz.Rect(72, 240, 296, 500), LEFT + " The adapta-\ntion cost is high.", fontsize=10
            )
            page.insert_text((316, 230), "2", fontsize=12, fontname="hebo")
            page.insert_text((334, 230), "Method", fontsize=12, fontname="hebo")
            page.insert_textbox(fitz.Rect(316, 240, 540, 500), RIGHT, fontsize=10)
        elif pno == 1:
            page.insert_text((72, 80), "3 Experiments", fontsize=12, fontname="hebo")
            page.insert_textbox(
                fitz.Rect(72, 90, 540, 300), "We evaluate on GLUE. " * 20, fontsize=10
            )
        else:
            page.insert_text((72, 80), "References", fontsize=12, fontname="hebo")
            page.insert_textbox(
                fitz.Rect(72, 90, 540, 400),
                "[1] A. Author and B. Author. A paper about retrieval models. In ACL, 2020.\n"
                "[2] C. Writer. Another study on transformers and adapters. In NeurIPS, 2021.",
                fontsize=9,
            )
        page.insert_text((300, 770), str(pno + 1), fontsize=9)
    doc.save(path)
    return path


@pytest.fixture()
def pdf(tmp_path: Path) -> Path:
    return _two_column_pdf(tmp_path / "paper.pdf")


def test_layout_parser_sections_and_order(pdf: Path) -> None:
    doc = PyMuPDFLayoutParser().parse(
        pdf, {"arxiv_id": "0000.00001", "title": "T", "authors": ["A"], "year": 2024}
    )
    heads = [s.heading for s in doc.sections]
    assert "1 Introduction" in heads and "2 Method" in heads and "3 Experiments" in heads
    intro = next(s for s in doc.sections if s.heading == "1 Introduction")
    assert "expensive to adapt" in intro.text
    assert "low-rank matrices" not in intro.text  # right column not interleaved
    assert "adaptation cost" in intro.text  # de-hyphenated
    assert "Preprint under review" not in doc.full_text  # running header removed
    assert doc.abstract.startswith("We study adaptation")
    assert len(doc.references) == 2
    assert doc.doc_id == "0000.00001" and doc.year == 2024


def test_quality_report(pdf: Path) -> None:
    q = parse_quality(PyMuPDFLayoutParser().parse(pdf))
    assert q["sections_identified"] is True
    assert q["n_references"] == 2


def test_plain_parser_runs(pdf: Path) -> None:
    assert PyMuPDFPlainParser().parse(pdf).sections


def test_unreadable_pdf_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(ValueError):
        PyMuPDFLayoutParser().parse(bad)


@pytest.mark.parametrize(
    "text,level",
    [
        ("1 Introduction", 1),
        ("3.2 Training details", 2),
        ("A.1 Hyperparameters", 2),
        ("Related Work", 1),
        ("IV. EXPERIMENTS", 1),
        ("The model is trained.", None),
        ("2 3 4 5", None),
        ("1 We trained the model for 3 epochs.", None),
    ],
)
def test_heading_level(text: str, level: int | None) -> None:
    assert heading_level(text) == level


def test_split_references_author_year() -> None:
    refs = split_references(
        "Devlin, J., Chang, M. BERT: pre-training. In NAACL, 2019.\n"
        "Hu, E. J., Shen, Y. LoRA: low-rank adaptation. In ICLR, 2022."
    )
    assert len(refs) == 2
