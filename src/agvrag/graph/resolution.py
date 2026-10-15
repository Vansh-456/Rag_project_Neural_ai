"""Entity normalization and resolution.

Pipeline for an incoming surface form (``name``, ``type``):
  1. exact match of the normalized form against the alias table (gazetteer + learned aliases)
  2. singular/plural and punctuation-insensitive match ("adapters" == "adapter")
  3. acronym match (``LoRA`` <-> ``Low-Rank Adaptation`` via initials)
  4. string-similarity candidates (difflib ratio) within the same type:
       >= ``merge_threshold``          -> merge automatically
       [ambiguous_low, merge_threshold) -> ask the adjudicator (LLM) if configured, else keep apart
  5. otherwise create a new canonical entity

Every merge decision is appended to ``audit`` so resolution errors can be catalogued.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable
from typing import Any

from agvrag.graph.gazetteer import Gazetteer, make_entity_id
from agvrag.schemas import Entity
from agvrag.utils.text import normalize_name, singularize

Adjudicator = Callable[[str, str, str], bool]  # (name_a, name_b, type) -> same entity?

_ACRONYM_DEF_RE = re.compile(
    r"\b((?:[A-Z][a-z]+|[a-z]+)(?:[\s\-](?:[A-Za-z][a-z]+|of|for|and|in|to)){1,6})\s*\(([A-Z][A-Za-z0-9\-]{1,10})\)"
)


def _key(name: str) -> str:
    return " ".join(singularize(w) for w in normalize_name(name).replace(".", " ").split())


def initials(phrase: str) -> str:
    words = [
        w
        for w in re.split(r"[\s\-]+", phrase)
        if w and w.lower() not in {"of", "for", "and", "the", "in", "to", "a"}
    ]
    return "".join(w[0] for w in words).lower()


def acronym_matches(acronym: str, phrase: str) -> bool:
    """True when ``acronym`` (a single short token) spells the initials of ``phrase``."""
    if not acronym or any(ch.isspace() for ch in acronym) or not 3 <= len(acronym) <= 10:
        return False
    if len(phrase.split()) < 2:
        return False
    caps = "".join(ch for ch in acronym if ch.isupper() or ch.isdigit()).lower()
    if len(caps) < 2:
        return False
    return caps == initials(phrase)


def mine_acronyms(text: str) -> list[tuple[str, str]]:
    """Find ``Long Form (LF)`` definitions whose initials agree. Returns (long, short)."""
    out: list[tuple[str, str]] = []
    for m in _ACRONYM_DEF_RE.finditer(text):
        long_form, short = m.group(1).strip(), m.group(2)
        words = long_form.split()
        # trim leading words until the initials line up (regex is greedy on the left)
        for i in range(len(words)):
            cand = " ".join(words[i:])
            if acronym_matches(short, cand):
                out.append((cand, short))
                break
    return out


class EntityResolver:
    def __init__(
        self,
        gazetteer: Gazetteer | None = None,
        merge_threshold: float = 0.92,
        ambiguous_low: float = 0.82,
        adjudicator: Adjudicator | None = None,
    ) -> None:
        self.merge_threshold = merge_threshold
        self.ambiguous_low = ambiguous_low
        self.adjudicator = adjudicator
        self.entities: dict[str, Entity] = {}
        self._alias: dict[tuple[str, str], str] = {}  # (type, key) -> entity_id
        self._alias_any: dict[str, set[str]] = {}  # key -> entity_ids (any type)
        self.audit: list[dict[str, Any]] = []
        if gazetteer:
            for g in gazetteer.entries:
                ent = Entity(
                    g.entity_id,
                    g.name,
                    g.type,
                    list(g.aliases),
                    {
                        "source": "gazetteer",
                        "case_sensitive": g.case_sensitive,
                        **({"introduced_in": g.introduced_in} if g.introduced_in else {}),
                    },
                )
                self.add_canonical(ent)

    def add_canonical(self, ent: Entity) -> Entity:
        existing = self.entities.get(ent.entity_id)
        if existing:
            for a in ent.aliases:
                self.add_alias(existing.entity_id, a)
            existing.properties.update(
                {k: v for k, v in ent.properties.items() if k not in existing.properties}
            )
            return existing
        self.entities[ent.entity_id] = ent
        for s in [ent.name, *ent.aliases]:
            self._register(ent, s)
        return ent

    def _register(self, ent: Entity, surface: str) -> None:
        k = _key(surface)
        if not k:
            return
        self._alias.setdefault((ent.type, k), ent.entity_id)
        self._alias_any.setdefault(k, set()).add(ent.entity_id)

    def add_alias(self, entity_id: str, alias: str) -> None:
        ent = self.entities[entity_id]
        if alias != ent.name and alias not in ent.aliases:
            ent.aliases.append(alias)
        self._register(ent, alias)

    def lookup(self, name: str, etype: str | None = None) -> str | None:
        k = _key(name)
        if etype is not None:
            hit = self._alias.get((etype, k))
            if hit:
                return hit
        ids = self._alias_any.get(k)
        if ids and len(ids) == 1:
            return next(iter(ids))
        return None

    def resolve(self, name: str, etype: str, context: str = "") -> str:
        """Return the canonical entity id for (name, type), creating one if needed."""
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            raise ValueError("empty entity name")
        hit = self.lookup(name, etype)
        if hit:
            return hit
        # acronym <-> long form within same type
        for ent in self._same_type(etype):
            for s in [ent.name, *ent.aliases]:
                if acronym_matches(name, s) or acronym_matches(s, name):
                    self._merge_log(name, ent, "acronym", 1.0)
                    self.add_alias(ent.entity_id, name)
                    return ent.entity_id
        best: tuple[float, Entity | None] = (0.0, None)
        k = _key(name)
        for ent in self._same_type(etype):
            for s in [ent.name, *ent.aliases]:
                r = difflib.SequenceMatcher(None, k, _key(s)).ratio()
                if r > best[0]:
                    best = (r, ent)
        score, cand = best
        if cand is not None and score >= self.merge_threshold and len(k) > 4:
            self._merge_log(name, cand, "string_similarity", score)
            self.add_alias(cand.entity_id, name)
            return cand.entity_id
        if cand is not None and score >= self.ambiguous_low and self.adjudicator is not None:
            same = self.adjudicator(name, cand.name, etype)
            self._merge_log(name, cand, "llm_adjudication", score, accepted=same)
            if same:
                self.add_alias(cand.entity_id, name)
                return cand.entity_id
        eid = make_entity_id(name, etype)
        if eid in self.entities:  # normalized id collision across differently-typed aliases
            return eid
        self.add_canonical(Entity(eid, name, etype, [], {"source": "extracted"}))
        return eid

    def _same_type(self, etype: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.type == etype]

    def _merge_log(
        self, name: str, ent: Entity, method: str, score: float, accepted: bool = True
    ) -> None:
        self.audit.append(
            {
                "surface": name,
                "merged_into": ent.entity_id,
                "canonical": ent.name,
                "method": method,
                "score": round(score, 4),
                "accepted": accepted,
            }
        )
