"""Deterministic identifiers."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def stable_hash(*parts: Any, length: int = 16) -> str:
    """SHA-256 over a canonical JSON encoding of ``parts``; first ``length`` hex chars."""
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def chunk_id(doc_id: str, strategy: str, params: dict[str, Any], ordinal: int) -> str:
    return f"{doc_id}::{strategy}::{stable_hash(params, length=6)}::{ordinal:04d}"


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def uuid_from_string(s: str) -> str:
    """Deterministic UUID (Qdrant point ids must be uint or UUID)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))
