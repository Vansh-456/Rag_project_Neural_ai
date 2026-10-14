"""Text utilities: token counting, sentence splitting, normalization.

Token counting uses a deterministic BPE-approximating heuristic rather than a downloaded
tokenizer, so chunk boundaries (and therefore chunk ids) are identical on every machine,
offline or in Docker. On English scientific prose it tracks ``cl100k_base`` with
~7% mean per-passage error (calibration in ``tests/test_text.py``); exact counts are not needed for chunk *sizing*.
"""

from __future__ import annotations

import math
import re
import unicodedata

_WORD_RE = re.compile(r"\S+")
_TOKEN_PIECE_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
# Sentence boundary: ., !, ? followed by whitespace + capital/digit/bracket; avoids splitting
# common abbreviations used in papers.
_ABBREV = {
    "e.g",
    "i.e",
    "et al",
    "fig",
    "eq",
    "sec",
    "vs",
    "cf",
    "resp",
    "approx",
    "al",
    "no",
    "tab",
    "figs",
    "eqs",
    "dr",
    "mr",
    "ms",
    "etc",
}
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"])")


def count_tokens(text: str) -> int:
    """Approximate subword-token count of ``text`` (deterministic, >= 0).

    Alphabetic runs: 1 token up to 6 chars, +1 per further 4 chars; digit runs: 1 per 3
    digits; every other non-space character: 1. Calibrated against ``cl100k_base`` on this
    corpus: aggregate ratio 1.00, mean per-passage absolute error 6.7%.
    """
    n = 0
    for piece in _TOKEN_PIECE_RE.findall(text):
        if piece[0].isalpha():
            n += 1 + (len(piece) - 6) // 4 if len(piece) > 6 else 1
        elif piece[0].isdigit():
            n += math.ceil(len(piece) / 3)
        else:
            n += 1
    return n


def words_with_tokens(text: str) -> list[tuple[str, int]]:
    """Split into whitespace-delimited words with their approximate token counts."""
    return [(w, count_tokens(w)) for w in _WORD_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter tuned for scientific prose (keeps abbreviations intact)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_END_RE.split(text)
    out: list[str] = []
    buf = ""
    for p in parts:
        buf = f"{buf} {p}".strip() if buf else p
        last_word = buf.rsplit(" ", 1)[-1].rstrip(".").lower()
        tail2 = " ".join(buf.rsplit(" ", 2)[-2:]).rstrip(".").lower()
        if last_word in _ABBREV or tail2 in _ABBREV or re.search(r"\b[A-Z]\.$", buf):
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def normalize_name(name: str) -> str:
    """Canonical form for entity matching: NFKC, lowercase, unify dashes, strip punctuation."""
    s = unicodedata.normalize("NFKC", name).lower()
    s = re.sub(r"[\u2010-\u2015\-_/]", " ", s)
    s = re.sub(r"[^\w\s+.]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(".")
    return s


def singularize(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))
