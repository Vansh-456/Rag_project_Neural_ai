"""arXiv corpus acquisition: metadata via the Atom API, PDFs via arxiv.org, rate-limited.

arXiv's API terms ask for <= 1 request every 3 seconds; ``delay_seconds`` enforces that for
both metadata and PDF requests. Downloads are idempotent: existing files are skipped.
Metadata titles are compared with the expected titles in ``papers.yaml`` so that a wrong
arXiv id in the corpus list is caught instead of silently ingesting the wrong paper.
"""

from __future__ import annotations

import difflib
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from agvrag.utils.io import ensure_dir, read_json, read_yaml, write_json
from agvrag.utils.logging import get_logger

log = get_logger(__name__)

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "agvrag-research-corpus/1.0 (mailto:research@example.org)"


@dataclass
class PaperSpec:
    arxiv_id: str
    title: str
    topic: str = ""
    verify: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def load_corpus_list(path: Path) -> list[PaperSpec]:
    data = read_yaml(path)
    out: list[PaperSpec] = []
    for group in data.get("topics", []):
        topic = group["name"]
        for p in group["papers"]:
            out.append(
                PaperSpec(
                    arxiv_id=str(p["arxiv_id"]),
                    title=p["title"],
                    topic=topic,
                    verify=bool(p.get("verify", False)),
                    extra={k: v for k, v in p.items() if k not in {"arxiv_id", "title", "verify"}},
                )
            )
    ids = [p.arxiv_id for p in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate arXiv ids in corpus list: {sorted(dupes)}")
    return out


def parse_atom(xml_text: str) -> list[dict[str, Any]]:
    """Parse an arXiv Atom feed into metadata dicts (id without version)."""
    root = ET.fromstring(xml_text)
    out: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", ATOM):
        raw_id = (entry.findtext("a:id", default="", namespaces=ATOM) or "").strip()
        m = re.search(r"abs/([^v\s]+?)(v\d+)?$", raw_id)
        if not m:
            continue
        title = re.sub(r"\s+", " ", entry.findtext("a:title", default="", namespaces=ATOM)).strip()
        if title.lower() == "error":
            continue
        published = entry.findtext("a:published", default="", namespaces=ATOM)
        out.append(
            {
                "arxiv_id": m.group(1),
                "version": m.group(2) or "",
                "title": title,
                "authors": [
                    (a.findtext("a:name", default="", namespaces=ATOM) or "").strip()
                    for a in entry.findall("a:author", ATOM)
                ],
                "abstract": re.sub(
                    r"\s+", " ", entry.findtext("a:summary", default="", namespaces=ATOM)
                ).strip(),
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "published": published,
                "categories": [c.attrib.get("term", "") for c in entry.findall("a:category", ATOM)],
            }
        )
    return out


def title_similarity(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())  # noqa: E731
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


class ArxivDownloader:
    def __init__(
        self,
        out_dir: Path,
        delay_seconds: float = 3.0,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.out_dir = ensure_dir(out_dir)
        self.delay = delay_seconds
        self.client = client or httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        self._last_request = 0.0

    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(
        self, url: str, params: dict[str, Any] | None = None, retries: int = 4
    ) -> httpx.Response:
        for attempt in range(retries):
            self._throttle()
            try:
                r = self.client.get(url, params=params)
                if r.status_code == 200:
                    return r
                if r.status_code in {429, 500, 502, 503, 504}:
                    log.warning(
                        "arxiv retryable status", extra={"status": r.status_code, "url": url}
                    )
                else:
                    r.raise_for_status()
            except httpx.TransportError as exc:
                log.warning("arxiv transport error", extra={"error": str(exc), "url": url})
            time.sleep(self.delay * (2**attempt))
        raise RuntimeError(f"failed to fetch {url} after {retries} attempts")

    def fetch_metadata(
        self, arxiv_ids: list[str], batch_size: int = 20
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i : i + batch_size]
            r = self._get(API_URL, {"id_list": ",".join(batch), "max_results": len(batch)})
            for meta in parse_atom(r.text):
                out[meta["arxiv_id"]] = meta
        return out

    def download(
        self, specs: list[PaperSpec], min_title_similarity: float = 0.75
    ) -> dict[str, Any]:
        """Download metadata + PDFs. Returns a report with successes, mismatches and failures."""
        report: dict[str, Any] = {
            "downloaded": [],
            "skipped": [],
            "title_mismatch": [],
            "missing_metadata": [],
            "failed": [],
        }
        need_meta = [
            s.arxiv_id for s in specs if not (self.out_dir / f"{s.arxiv_id}.json").exists()
        ]
        fetched = self.fetch_metadata(need_meta) if need_meta else {}
        for spec in specs:
            meta_path = self.out_dir / f"{spec.arxiv_id}.json"
            pdf_path = self.out_dir / f"{spec.arxiv_id}.pdf"
            if meta_path.exists():
                meta = read_json(meta_path)
            else:
                meta = fetched.get(spec.arxiv_id)
                if meta is None:
                    report["missing_metadata"].append(spec.arxiv_id)
                    log.error("no arXiv metadata", extra={"arxiv_id": spec.arxiv_id})
                    continue
                meta["topic"] = spec.topic
                meta["expected_title"] = spec.title
                meta.update({k: v for k, v in spec.extra.items() if k not in meta})
                sim = title_similarity(spec.title, meta["title"])
                meta["title_similarity"] = round(sim, 3)
                if sim < min_title_similarity:
                    report["title_mismatch"].append(
                        {
                            "arxiv_id": spec.arxiv_id,
                            "expected": spec.title,
                            "got": meta["title"],
                            "similarity": round(sim, 3),
                        }
                    )
                    log.warning("title mismatch; skipping", extra=report["title_mismatch"][-1])
                    continue
                write_json(meta_path, meta)
            if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
                report["skipped"].append(spec.arxiv_id)
                continue
            try:
                r = self._get(PDF_URL.format(arxiv_id=spec.arxiv_id))
                if not r.content.startswith(b"%PDF"):
                    raise RuntimeError("response is not a PDF")
                pdf_path.write_bytes(r.content)
                report["downloaded"].append(spec.arxiv_id)
                log.info("downloaded", extra={"arxiv_id": spec.arxiv_id, "bytes": len(r.content)})
            except Exception as exc:
                report["failed"].append({"arxiv_id": spec.arxiv_id, "error": str(exc)})
                log.error(
                    "pdf download failed", extra={"arxiv_id": spec.arxiv_id, "error": str(exc)}
                )
        return report
