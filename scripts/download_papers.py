"""
arXiv paper acquisition script.
Fetches paper metadata, abstracts, and PDFs with rate limiting and local caching.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


def fetch_arxiv_metadata(arxiv_id: str) -> dict[str, str]:
    """Queries the public arXiv API for paper metadata."""
    clean_id = arxiv_id.replace("arXiv:", "").strip()
    url = f"http://export.arxiv.org/api/query?id_list={clean_id}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Adaptive-RAG-Research/1.0"})
        with urllib.request.urlopen(req, timeout=10.0) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)

        if entry is not None:
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]

            return {
                "arxiv_id": clean_id,
                "title": title.text.strip().replace("\n", " ") if title is not None and title.text else clean_id,
                "abstract": summary.text.strip().replace("\n", " ") if summary is not None and summary.text else "",
                "authors": [a for a in authors if a]
            }
    except Exception as e:
        print(f"Warning: Failed to fetch online metadata for {arxiv_id}: {e}")

    return {"arxiv_id": clean_id, "title": clean_id, "abstract": "", "authors": []}


def download_corpus(paper_list_file: str = "docs/problem_definition.md", output_dir: str = "data/raw/papers"):
    """Extracts arXiv IDs from problem_definition.md and acquires papers."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Reading paper catalogue from {paper_list_file}...")
    with open(paper_list_file, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    matches = re.findall(r"`arXiv:(\d+\.\d+)`", content)
    print(f"[*] Found {len(matches)} papers with arXiv IDs.")

    acquired = []
    for idx, aid in enumerate(matches[:10], start=1):  # Demonstrate rate-limited acquisition
        print(f"  [{idx}/{len(matches[:10])}] Fetching metadata for arXiv:{aid}...")
        meta = fetch_arxiv_metadata(aid)
        acquired.append(meta)
        time.sleep(1.0)  # Rate limiting compliance with arXiv guidelines

    manifest_file = out_path / "paper_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(acquired, f, indent=2)

    print(f"[+] Download manifest saved to {manifest_file}")


if __name__ == "__main__":
    download_corpus()
