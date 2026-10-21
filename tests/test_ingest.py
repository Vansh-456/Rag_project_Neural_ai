"""Incremental ingestion of one PDF into an existing index + graph (offline)."""

from pathlib import Path
from typing import Any

import pymupdf as fitz
import pytest

from agvrag.config import Settings
from agvrag.pipeline import indexing
from agvrag.pipeline.components import ComponentRegistry, load_chunks
from agvrag.pipeline.indexing import build_graph_index, build_index, ingest_pdf
from agvrag.utils.io import write_json

from .conftest import make_corpus, make_settings


def _paper_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((150, 80), "QLoRA Revisited", fontsize=18, fontname="hebo")
    page.insert_text((72, 130), "1 Introduction", fontsize=12, fontname="hebo")
    page.insert_textbox(
        fitz.Rect(72, 140, 540, 400),
        "We extend LoRA with quantization. LoRA is based on low-rank updates. "
        "We evaluate on GLUE and compare against Adapters. " * 3,
        fontsize=10,
    )
    page.insert_text((72, 430), "References", fontsize=12, fontname="hebo")
    page.insert_textbox(
        fitz.Rect(72, 440, 540, 600),
        "[1] E. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR, 2022.",
        fontsize=9,
    )
    doc.save(path)
    return path


def _fresh_index(tmp: Path, **graph: Any) -> tuple[Settings, ComponentRegistry]:
    s = make_settings(tmp)
    if graph:
        s = s.model_copy(update={"graph": s.graph.model_copy(update=graph)})
    for d in make_corpus():
        write_json(s.path("documents_dir") / f"{d.doc_id}.json", d.to_dict())
    reg = ComponentRegistry()
    build_index(s, reg)
    build_graph_index(s, reg)
    return s, reg


def test_ingest_adds_chunks_and_graph(tmp_path: Path) -> None:
    s, reg = _fresh_index(tmp_path)
    n_before = len(load_chunks(s))
    rep = ingest_pdf(
        s, _paper_pdf(tmp_path / "new.pdf"), {"doc_id": "9999.00001", "title": None}, reg
    )
    assert rep["chunks"] > 0 and rep["graph_relations_added"] > 0
    assert len(load_chunks(s)) == n_before + rep["chunks"]
    g = reg.graph(s)  # registry was invalidated: reloads graph.json
    assert g.get_entity("paper:9999.00001") is not None


def test_ingest_pushes_delta_to_neo4j(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Building with backend=neo4j would sync the whole graph; build with networkx, then switch.
    s, reg = _fresh_index(tmp_path)
    s = s.model_copy(update={"graph": s.graph.model_copy(update={"backend": "neo4j"})})
    pushed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        indexing,
        "sync_neo4j",
        lambda settings, store, registry, clear=True: pushed.append(
            {"store": store, "clear": clear}
        ),
    )
    ingest_pdf(s, _paper_pdf(tmp_path / "new.pdf"), {"doc_id": "9999.00002", "title": None}, reg)
    assert len(pushed) == 1 and pushed[0]["clear"] is False
    delta = pushed[0]["store"]
    assert delta.get_entity("paper:9999.00002") is not None
    # every pushed relation has both endpoints in the delta (Neo4j MATCHes them)
    for r in delta.relations():
        assert delta.get_entity(r.source) and delta.get_entity(r.target)
