"""Text cleaning for PDF extraction artifacts.

Handles: ligatures / compatibility characters (NFKC), soft and line-end hyphenation,
running headers/footers, page numbers, arXiv side-stamps, and whitespace.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_ARXIV_STAMP_RE = re.compile(r"arXiv:\d{4}\.\d{4,5}(v\d+)?\s*\[[\w.\-]+\]", re.I)
_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d{1,3}\s*(of\s*\d+)?\s*$", re.I)
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "­": ""}


def normalize_chars(text: str) -> str:
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKC", text)
    # Replace private-use / control characters (common with math fonts) by a space.
    return "".join(
        ch if (ch in "\n\t" or unicodedata.category(ch)[0] != "C") else " " for ch in text
    )


def dehyphenate(text: str) -> str:
    """Join words broken across lines: ``adapta-\\ntion`` -> ``adaptation``.

    Keeps the hyphen when the continuation starts uppercase (``BERT-\\nLarge``), since that is
    usually a genuine compound.
    """

    def _join(m: re.Match[str]) -> str:
        left, right = m.group(1), m.group(2)
        return f"{left}-{right}" if right.isupper() else f"{left}{right}"

    return _HYPHEN_BREAK_RE.sub(_join, text)


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return bool(_PAGE_NUM_RE.match(s) or _ARXIV_STAMP_RE.search(s))


def find_repeated_margin_lines(pages_lines: list[list[str]], min_fraction: float = 0.5) -> set[str]:
    """Lines that recur on many pages' first/last 2 lines are running headers/footers."""
    if len(pages_lines) < 3:
        return set()
    counter: Counter[str] = Counter()
    for lines in pages_lines:
        margin = {re.sub(r"\d+", "#", ln.strip()) for ln in lines[:2] + lines[-2:] if ln.strip()}
        counter.update(margin)
    threshold = max(3, int(min_fraction * len(pages_lines)))
    return {ln for ln, c in counter.items() if c >= threshold and len(ln) < 120}


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def reflow_paragraphs(text: str) -> str:
    """Join hard-wrapped lines inside paragraphs; keep blank-line paragraph breaks."""
    text = dehyphenate(text)
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(re.sub(r"\s*\n\s*", " ", p).strip() for p in paragraphs if p.strip())


def garbled_score(text: str) -> float:
    """0 = clean prose, 1 = garbage. Mix of non-alphabetic char ratio and vowel-less words."""
    if not text:
        return 1.0
    sample = text[:20000]
    letters = sum(ch.isalpha() for ch in sample)
    weird = sum(
        1 for ch in sample if not (ch.isalnum() or ch.isspace() or ch in ".,;:()[]-'\"%/=+<>!?")
    )
    words = re.findall(r"[A-Za-z]{4,}", sample)
    vowelless = sum(1 for w in words if not re.search(r"[aeiouyAEIOUY]", w))
    char_ratio = weird / max(1, len(sample))
    letter_deficit = max(0.0, 0.6 - letters / max(1, len(sample)))
    vowel_ratio = vowelless / max(1, len(words))
    return round(min(1.0, 2.0 * char_ratio + letter_deficit + vowel_ratio), 4)
