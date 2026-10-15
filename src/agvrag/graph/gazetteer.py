"""Gazetteer: the seed alias table used for mention detection and entity resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agvrag.schemas import ENTITY_TYPES
from agvrag.utils.io import read_yaml
from agvrag.utils.text import normalize_name


def make_entity_id(name: str, etype: str) -> str:
    return f"{etype.lower()}:{normalize_name(name)}"


@dataclass
class GazEntry:
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    introduced_in: str | None = None
    case_sensitive: bool = False

    @property
    def entity_id(self) -> str:
        return make_entity_id(self.name, self.type)

    @property
    def surfaces(self) -> list[str]:
        return [self.name, *self.aliases]


@dataclass
class Mention:
    entity_id: str
    surface: str
    start: int
    end: int


_BOUND_L = r"(?<![A-Za-z0-9\-])"
_BOUND_R = r"(?![A-Za-z0-9])"


class Gazetteer:
    """Longest-match-first surface matcher over all entries.

    Case-sensitive entries (short acronyms like ``RAG``, ``EM``) only match with exact case;
    the rest match case-insensitively. Word boundaries forbid matching inside other tokens,
    e.g. ``RAG`` does not match inside ``Self-RAG`` or ``GraphRAG``.
    """

    def __init__(self, entries: list[GazEntry]) -> None:
        for e in entries:
            if e.type not in ENTITY_TYPES:
                raise ValueError(f"gazetteer entry {e.name!r} has unknown type {e.type!r}")
        self.entries = entries
        self.by_id: dict[str, GazEntry] = {}
        for e in entries:
            if e.entity_id in self.by_id:
                raise ValueError(f"duplicate gazetteer entity {e.entity_id}")
            self.by_id[e.entity_id] = e
        self._surface_cs: dict[str, str] = {}
        self._surface_ci: dict[str, str] = {}
        for e in entries:
            for s in e.surfaces:
                if e.case_sensitive and (len(s) <= 5 or s.isupper()):
                    self._surface_cs.setdefault(s, e.entity_id)
                else:
                    self._surface_ci.setdefault(s.lower(), e.entity_id)
        self._re_cs = self._compile(list(self._surface_cs), flags=0)
        self._re_ci = self._compile(list(self._surface_ci), flags=re.IGNORECASE)

    @staticmethod
    def _compile(surfaces: list[str], flags: int) -> re.Pattern[str] | None:
        if not surfaces:
            return None
        alts = "|".join(re.escape(s) for s in sorted(surfaces, key=len, reverse=True))
        return re.compile(f"{_BOUND_L}(?:{alts}){_BOUND_R}", flags)

    @classmethod
    def load(cls, path: Path) -> Gazetteer:
        data = read_yaml(path)
        return cls(
            [
                GazEntry(
                    name=str(e["name"]),
                    type=e["type"],
                    aliases=[str(a) for a in e.get("aliases", [])],
                    introduced_in=str(e["introduced_in"]) if e.get("introduced_in") else None,
                    case_sensitive=bool(e.get("case_sensitive", False)),
                )
                for e in data.get("entities", [])
            ]
        )

    def find_mentions(self, text: str) -> list[Mention]:
        """Non-overlapping mentions, longest span wins on overlap, ordered by position."""
        found: list[Mention] = []
        if self._re_cs is not None:
            for rm in self._re_cs.finditer(text):
                found.append(
                    Mention(self._surface_cs[rm.group(0)], rm.group(0), rm.start(), rm.end())
                )
        if self._re_ci is not None:
            for rm in self._re_ci.finditer(text):
                eid = self._surface_ci[rm.group(0).lower()]
                found.append(Mention(eid, rm.group(0), rm.start(), rm.end()))
        found.sort(key=lambda m: (m.start, -(m.end - m.start)))
        out: list[Mention] = []
        last_end = -1
        for m in found:
            if m.start >= last_end:
                out.append(m)
                last_end = m.end
        return out
