"""
Text cleaning utilities for document ingestion.
Fixes hyphenated line breaks, ligatures, header/footer noise, and whitespace.
"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Cleans raw parsed text from academic papers."""
    if not text:
        return ""

    # Replace standard unicode ligatures
    ligatures = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "—": " - ",
        "–": " - ",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    for lig, replacement in ligatures.items():
        text = text.replace(lig, replacement)

    # Rejoin hyphenated line breaks (e.g. "trans- \n former" -> "transformer")
    text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)

    # Normalize arbitrary line breaks within sentences to single space
    # (preserve double newlines for paragraph breaks)
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []
    for p in paragraphs:
        # replace single newline with space
        cleaned_p = re.sub(r"\s*\n\s*", " ", p.strip())
        # remove consecutive spaces
        cleaned_p = re.sub(r"[ \t]+", " ", cleaned_p)
        if cleaned_p:
            cleaned_paragraphs.append(cleaned_p)

    return "\n\n".join(cleaned_paragraphs)


def calculate_garble_score(text: str) -> float:
    """
    Computes a heuristic score (0.0 to 1.0) indicating garbled/unprintable characters.
    0.0 means completely clean ASCII/printable; 1.0 means completely unreadable.
    """
    if not text:
        return 0.0

    non_ascii_or_special = sum(1 for c in text if ord(c) > 127 or (ord(c) < 32 and c not in "\n\r\t"))
    return round(non_ascii_or_special / max(1, len(text)), 4)
