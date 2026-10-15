"""Entity / relation extraction.

``RuleExtractor`` (baseline, zero cost, deterministic):
  * metadata edges: Paper -AUTHORED_BY-> Author (arXiv metadata, grounded to the front-matter
    chunk that mentions the author when possible)
  * INTRODUCES: gazetteer prior (``introduced_in``) **grounded** in the paper's own text, or a
    title prefix ("LoRA: ...") naming an entity
  * cue-pattern edges inside one sentence: ``<subj> <cue> <obj>`` for IMPROVES, BASED_ON,
    EXTENDS, USES, EVALUATES_ON, APPLIED_TO, where ``we/our`` resolves to the paper's
    introduced method; type constraints filter nonsensical pairs
  * frequency edges: Paper -USES-> X and Paper -EVALUATES_ON-> Dataset when mentioned
    repeatedly in method/experiment sections
  * CITES from reference lists resolved to corpus papers (arXiv id or fuzzy title match)

``LLMExtractor`` asks an ``LLMClient`` for JSON matching a schema, validates it with pydantic,
rejects relations whose evidence quote is not found in the chunk (hallucination guard), and
maps names through the shared ``EntityResolver``.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from agvrag.graph.gazetteer import Gazetteer, Mention
from agvrag.graph.resolution import EntityResolver, mine_acronyms
from agvrag.interfaces import LLMClient
from agvrag.schemas import ENTITY_TYPES, RELATION_TYPES, Chunk, Document, Entity, Relation
from agvrag.utils.logging import get_logger
from agvrag.utils.text import normalize_name, split_sentences

log = get_logger(__name__)

ARTIFACT_TYPES = {"Method", "Model"}
TYPE_CONSTRAINTS: dict[str, tuple[set[str], set[str]]] = {
    "IMPROVES": (ARTIFACT_TYPES, ARTIFACT_TYPES),
    "BASED_ON": (ARTIFACT_TYPES, ARTIFACT_TYPES | {"Concept"}),
    "EXTENDS": (ARTIFACT_TYPES, ARTIFACT_TYPES),
    "USES": (ARTIFACT_TYPES | {"Paper"}, ARTIFACT_TYPES | {"Dataset", "Metric", "Concept"}),
    "EVALUATES_ON": (ARTIFACT_TYPES | {"Paper"}, {"Dataset"}),
    "APPLIED_TO": (ARTIFACT_TYPES, {"Task"}),
    "INTRODUCES": ({"Paper"}, ARTIFACT_TYPES | {"Dataset", "Metric", "Concept", "Task"}),
    "AUTHORED_BY": ({"Paper"}, {"Author"}),
    "CITES": ({"Paper"}, {"Paper"}),
    "INFLUENCES": (ARTIFACT_TYPES | {"Concept", "Paper"}, ARTIFACT_TYPES | {"Concept", "Task"}),
}

# (relation, regex). Order matters: more specific cues first.
CUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "IMPROVES",
        re.compile(
            r"\b(outperform\w*|surpass\w*|improv\w* (?:up)?on|improv\w* over|"
            r"better than|beats?|exceed\w*|superior to)\b",
            re.I,
        ),
    ),
    ("EXTENDS", re.compile(r"\b(extends?|extension of|generali[sz]es?)\b", re.I)),
    (
        "BASED_ON",
        re.compile(
            r"\b(based on|builds? (?:up)?on|built (?:up)?on|inspired by|"
            r"initiali[sz]ed (?:from|with)|derived from|variant of|follows?)\b",
            re.I,
        ),
    ),
    (
        "EVALUATES_ON",
        re.compile(
            r"\b(evaluat\w* on|experiments? on|results? on|benchmarks? on|"
            r"tested on|test on|accuracy on|performance on)\b",
            re.I,
        ),
    ),
    (
        "APPLIED_TO",
        re.compile(
            r"\b(appl(?:y|ied|ies|ying) (?:\w+ ){0,3}to|for the task of|"
            r"tasks? such as)\b",
            re.I,
        ),
    ),
    (
        "USES",
        re.compile(r"\b(uses?|using|employs?|leverag\w+|adopts?|utili[sz]\w+|relies on)\b", re.I),
    ),
]
_NEGATION_RE = re.compile(r"\b(not|no|never|fails? to|does not|do not|cannot|worse than)\b", re.I)
_SELF_RE = re.compile(
    r"\b(we|our|ours|this paper|this work|our method|our approach|our model)\b", re.I
)
_PROPOSE_RE = re.compile(
    r"\b(we (?:propose|introduce|present|call|name|dub)|called|named|"
    r"which we call|we refer to)\b",
    re.I,
)
_EXPERIMENT_SECTION_RE = re.compile(r"experiment|result|evaluation|benchmark|setup|empirical", re.I)
_SKIP_SECTION_RE = re.compile(r"related work|background|reference|acknowledg|bibliograph", re.I)


@dataclass
class ExtractionResult:
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    mentions: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    rejected: list[dict[str, Any]] = field(default_factory=list)


def paper_entity_id(doc_id: str) -> str:
    return f"paper:{doc_id}"


def author_entity_id(name: str) -> str:
    return f"author:{normalize_name(name)}"


def _type_ok(rel: str, s_type: str, t_type: str) -> bool:
    allowed_s, allowed_t = TYPE_CONSTRAINTS[rel]
    return s_type in allowed_s and t_type in allowed_t


class RuleExtractor:
    name = "rule"

    def __init__(
        self,
        gazetteer: Gazetteer,
        resolver: EntityResolver,
        corpus_titles: dict[str, str] | None = None,
        use_propose_pattern: bool = False,
    ) -> None:
        self.use_propose_pattern = use_propose_pattern
        self.gaz = gazetteer
        self.resolver = resolver
        self.corpus_titles = corpus_titles or {}
        self._title_norm = {
            d: re.sub(r"[^a-z0-9 ]", "", t.lower()) for d, t in self.corpus_titles.items()
        }

    # ---------------------------------------------------------------- helpers
    def _entity(self, eid: str) -> Entity:
        ent = self.resolver.entities.get(eid)
        if ent is None:
            raise KeyError(eid)
        return ent

    def _paper_entity(self, doc: Document) -> Entity:
        return Entity(
            entity_id=paper_entity_id(doc.doc_id),
            name=doc.title,
            type="Paper",
            aliases=[doc.doc_id],
            properties={
                "arxiv_id": doc.doc_id,
                "year": doc.year or 0,
                "topic": str(doc.metadata.get("topic", "")),
                "title": doc.title,
            },
        )

    def resolve_reference(self, ref: str, own_id: str) -> str | None:
        """Map a reference string to a corpus doc id (arXiv id in text, else fuzzy title)."""
        for m in re.finditer(r"(\d{4}\.\d{4,5})", ref):
            if m.group(1) in self.corpus_titles and m.group(1) != own_id:
                return m.group(1)
        ref_norm = re.sub(r"[^a-z0-9 ]", "", ref.lower())
        best, best_id = 0.0, None
        for doc_id, tnorm in self._title_norm.items():
            if doc_id == own_id or len(tnorm) < 12:
                continue
            if tnorm in ref_norm:
                return doc_id
            # fuzzy: compare title against a same-length window around the best block match
            sm = difflib.SequenceMatcher(None, tnorm, ref_norm, autojunk=False)
            block = sm.find_longest_match(0, len(tnorm), 0, len(ref_norm))
            if block.size < 10:
                continue
            start = max(0, block.b - block.a)
            window = ref_norm[start : start + len(tnorm)]
            r = difflib.SequenceMatcher(None, tnorm, window, autojunk=False).ratio()
            if r > best:
                best, best_id = r, doc_id
        return best_id if best >= 0.88 else None

    # ---------------------------------------------------------------- main
    def extract(self, doc: Document, chunks: list[Chunk]) -> ExtractionResult:
        res = ExtractionResult()
        paper = self._paper_entity(doc)
        res.entities[paper.entity_id] = paper
        chunks = sorted(chunks, key=lambda c: c.metadata.get("ordinal", 0))
        first_chunk = chunks[0].chunk_id if chunks else ""

        # Acronym definitions found in this paper become aliases of known entities.
        for long_form, short in mine_acronyms(
            doc.abstract + "\n" + " ".join(c.text for c in chunks[:6])
        ):
            eid = self.resolver.lookup(short) or self.resolver.lookup(long_form)
            if eid:
                self.resolver.add_alias(eid, long_form)
                self.resolver.add_alias(eid, short)

        # Authors (metadata), grounded to the chunk that contains the surname if any.
        for author in doc.authors:
            aid = author_entity_id(author)
            res.entities[aid] = Entity(aid, author, "Author", [], {})
            surname = author.split()[-1] if author.split() else author
            grounded = next((c.chunk_id for c in chunks[:4] if surname in c.text), first_chunk)
            res.relations.append(
                Relation(
                    paper.entity_id,
                    aid,
                    "AUTHORED_BY",
                    0.99,
                    [grounded] if grounded else [],
                    evidence_text=f"{doc.title} — {author}",
                    extractor="metadata",
                    properties={"origin": "arxiv_metadata"},
                )
            )

        # Mentions per chunk
        chunk_mentions: list[tuple[Chunk, list[Mention]]] = []
        for c in chunks:
            ms = self.gaz.find_mentions(c.text)
            chunk_mentions.append((c, ms))
            for m in ms:
                res.mentions[doc.doc_id][m.entity_id] += 1

        # INTRODUCES via gazetteer prior (grounded) and title prefix.
        introduced: set[str] = set()
        title_prefix = doc.title.split(":")[0].strip() if ":" in doc.title else ""
        title_eid = self.resolver.lookup(title_prefix) if title_prefix else None
        for g in self.gaz.entries:
            if g.introduced_in == doc.doc_id or g.entity_id == title_eid:
                ground = next(
                    (c for c, ms in chunk_mentions if any(m.entity_id == g.entity_id for m in ms)),
                    None,
                )
                if ground is None and g.entity_id != title_eid:
                    res.rejected.append(
                        {
                            "doc": doc.doc_id,
                            "entity": g.entity_id,
                            "reason": "introduced_in prior not grounded in text",
                        }
                    )
                    continue
                introduced.add(g.entity_id)
                sent = self._sentence_with(ground.text, g.surfaces) if ground else doc.title
                res.relations.append(
                    Relation(
                        paper.entity_id,
                        g.entity_id,
                        "INTRODUCES",
                        0.95 if g.entity_id == title_eid else 0.9,
                        [ground.chunk_id] if ground else ([first_chunk] if first_chunk else []),
                        evidence_text=sent[:400],
                        extractor="rule:introduces",
                    )
                )
        # "we propose X" in early chunks for entities without a prior elsewhere.
        # Disabled by default: the 50-edge audit + a full review of INTRODUCES edges found 0/17
        # correct (it attached any nearby mention: "RAG introduces Wikipedia"). See ADR-016.
        for c, _ in chunk_mentions[:6] if self.use_propose_pattern else []:
            for sent in split_sentences(c.text):
                if not _PROPOSE_RE.search(sent):
                    continue
                for m in self.gaz.find_mentions(sent):
                    gz = self.gaz.by_id.get(m.entity_id)
                    if (
                        gz is None
                        or m.entity_id in introduced
                        or gz.introduced_in not in (None, doc.doc_id)
                    ):
                        continue
                    if gz.type not in ARTIFACT_TYPES | {"Dataset", "Metric"}:
                        continue
                    introduced.add(m.entity_id)
                    res.relations.append(
                        Relation(
                            paper.entity_id,
                            m.entity_id,
                            "INTRODUCES",
                            0.6,
                            [c.chunk_id],
                            evidence_text=sent[:400],
                            extractor="rule:propose_pattern",
                        )
                    )

        self_method = sorted(i for i in introduced if self._entity(i).type in ARTIFACT_TYPES)
        self_subject = self_method[0] if len(self_method) >= 1 else paper.entity_id

        # Cue-pattern relations within a sentence.
        for c, _ in chunk_mentions:
            heading = c.section_heading or ""
            if _SKIP_SECTION_RE.search(heading):
                continue
            for sent in split_sentences(c.text):
                if len(sent) > 600:
                    continue
                ms = self.gaz.find_mentions(sent)
                for rel, pat in CUE_PATTERNS:
                    cue = pat.search(sent)
                    if not cue:
                        continue
                    if rel == "IMPROVES" and _NEGATION_RE.search(
                        sent[max(0, cue.start() - 30) : cue.end()]
                    ):
                        continue
                    # Only the nearest mention on each side of the cue, within a short window:
                    # taking every later mention produced "LoRA IMPROVES DeBERTa" (audit, ADR-016).
                    before = [m for m in ms if m.end <= cue.start() and cue.start() - m.end <= 80]
                    after = [m for m in ms if m.start >= cue.end() and m.start - cue.end() <= 60]
                    subj: str | None = before[-1].entity_id if before else None
                    pre = sent[: cue.start()]
                    self_ref = _SELF_RE.search(pre[-60:]) is not None
                    if (subj is None or self_ref) and _SELF_RE.search(pre):
                        subj = self_subject
                    if subj is None:
                        continue
                    for obj_m in after[:1]:
                        obj = obj_m.entity_id
                        if obj == subj:
                            continue
                        s_t = self._entity(subj).type if subj != paper.entity_id else "Paper"
                        t_t = self._entity(obj).type
                        if not _type_ok(rel, s_t, t_t):
                            continue
                        conf = 0.6 if subj != self_subject else 0.55
                        res.relations.append(
                            Relation(
                                subj,
                                obj,
                                rel,
                                conf,
                                [c.chunk_id],
                                evidence_text=sent[:400],
                                extractor=f"rule:cue:{cue.group(0).lower()}",
                            )
                        )
                    break  # one cue relation per sentence keeps precision up

        # Frequency-based paper-level edges.
        sec_counts: dict[str, Counter[str]] = {"exp": Counter(), "body": Counter()}
        for c, ms in chunk_mentions:
            heading = c.section_heading or ""
            if _SKIP_SECTION_RE.search(heading):
                continue
            bucket = "exp" if _EXPERIMENT_SECTION_RE.search(heading) else "body"
            for m in ms:
                sec_counts[bucket][m.entity_id] += 1
        grounding: dict[str, str] = {}
        for c, ms in chunk_mentions:
            for m in ms:
                grounding.setdefault(m.entity_id, c.chunk_id)
        for eid, n_exp in sec_counts["exp"].items():
            ent = self._entity(eid)
            if ent.type == "Dataset" and n_exp >= 2:
                res.relations.append(
                    Relation(
                        paper.entity_id,
                        eid,
                        "EVALUATES_ON",
                        round(min(0.9, 0.5 + 0.1 * n_exp), 3),
                        [grounding[eid]],
                        evidence_text=f"{ent.name} mentioned {n_exp}x in experiments",
                        extractor="rule:frequency",
                    )
                )
        total = sec_counts["exp"] + sec_counts["body"]
        for eid, n in total.items():
            if eid in introduced:
                continue
            ent = self._entity(eid)
            if ent.type in ARTIFACT_TYPES | {"Metric"} and n >= 3:
                res.relations.append(
                    Relation(
                        paper.entity_id,
                        eid,
                        "USES",
                        round(min(0.85, 0.35 + 0.08 * n), 3),
                        [grounding[eid]],
                        evidence_text=f"{ent.name} mentioned {n}x",
                        extractor="rule:frequency",
                    )
                )
        # Introduced method applied to tasks named in the abstract / first chunks.
        if self_method:
            early = " ".join(c.text for c in chunks[:3])
            for m in self.gaz.find_mentions(doc.abstract + " " + early):
                if self._entity(m.entity_id).type == "Task":
                    task_chunk = grounding.get(m.entity_id, first_chunk)
                    for meth in self_method:
                        res.relations.append(
                            Relation(
                                meth,
                                m.entity_id,
                                "APPLIED_TO",
                                0.55,
                                [task_chunk] if task_chunk else [],
                                evidence_text=self._sentence_with(doc.abstract, [m.surface])[:400],
                                extractor="rule:abstract_task",
                            )
                        )

        # Citations
        for ref in doc.references:
            target = self.resolve_reference(ref, doc.doc_id)
            if target:
                res.relations.append(
                    Relation(
                        paper.entity_id,
                        paper_entity_id(target),
                        "CITES",
                        0.9,
                        [],
                        evidence_text=ref[:300],
                        extractor="rule:reference",
                        properties={"origin": "reference_list"},
                    )
                )

        for eid in {r.source for r in res.relations} | {r.target for r in res.relations}:
            if eid in self.resolver.entities:
                res.entities[eid] = self.resolver.entities[eid]
        return res

    @staticmethod
    def _sentence_with(text: str, surfaces: list[str]) -> str:
        for s in split_sentences(text):
            if any(x.lower() in s.lower() for x in surfaces):
                return s
        return text[:300]


# ------------------------------------------------------------------------ LLM extraction


class _LLMEntity(BaseModel):
    name: str
    type: Literal["Method", "Model", "Dataset", "Metric", "Task", "Concept"]


class _LLMRelation(BaseModel):
    source: str
    source_type: Literal["Method", "Model", "Dataset", "Metric", "Task", "Concept", "Paper"]
    relation: Literal[
        "USES",
        "EVALUATES_ON",
        "IMPROVES",
        "EXTENDS",
        "BASED_ON",
        "APPLIED_TO",
        "INFLUENCES",
        "INTRODUCES",
    ]
    target: str
    target_type: Literal["Method", "Model", "Dataset", "Metric", "Task", "Concept"]
    evidence: str
    confidence: float = 0.7


class _LLMOutput(BaseModel):
    entities: list[_LLMEntity] = []
    relations: list[_LLMRelation] = []


EXTRACTION_SYSTEM = (
    "You extract a knowledge graph from research-paper text. Output ONLY JSON matching the "
    "schema. Use only facts stated in the passage. 'evidence' must be a verbatim quote "
    "(<= 30 words) from the passage. Use source='THIS_PAPER' with source_type='Paper' for "
    "statements about the current paper's own contribution."
)
EXTRACTION_SCHEMA = {
    "entities": [
        {"name": "str", "type": "|".join(t for t in ENTITY_TYPES if t not in {"Paper", "Author"})}
    ],
    "relations": [
        {
            "source": "str",
            "source_type": "str",
            "relation": "|".join(r for r in RELATION_TYPES if r not in {"AUTHORED_BY", "CITES"}),
            "target": "str",
            "target_type": "str",
            "evidence": "verbatim quote",
            "confidence": "0..1",
        }
    ],
}


def _evidence_supported(evidence: str, chunk_text: str, min_overlap: float = 0.6) -> bool:
    ev = re.findall(r"\w+", evidence.lower())
    if not ev:
        return False
    text = set(re.findall(r"\w+", chunk_text.lower()))
    return sum(w in text for w in ev) / len(ev) >= min_overlap


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output (tolerates code fences / prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in output")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    return obj


class LLMExtractor:
    name = "llm"

    def __init__(self, llm: LLMClient, resolver: EntityResolver, max_tokens: int = 1200) -> None:
        self.llm = llm
        self.resolver = resolver
        self.max_tokens = max_tokens
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "invalid": 0}

    def extract_chunk(self, doc: Document, chunk: Chunk) -> ExtractionResult:
        res = ExtractionResult()
        prompt = (
            f"Paper: {doc.title}\nSection: {chunk.section_heading}\n\nPassage:\n{chunk.text}\n\n"
            f"JSON schema:\n{json.dumps(EXTRACTION_SCHEMA)}"
        )
        resp = self.llm.complete(
            EXTRACTION_SYSTEM, prompt, max_tokens=self.max_tokens, temperature=0.0, json_mode=True
        )
        self.usage["calls"] += 1
        self.usage["input_tokens"] += resp.input_tokens
        self.usage["output_tokens"] += resp.output_tokens
        try:
            out = _LLMOutput.model_validate(parse_json_object(resp.text))
        except (ValueError, ValidationError) as exc:
            self.usage["invalid"] += 1
            res.rejected.append({"chunk": chunk.chunk_id, "reason": f"invalid json: {exc}"[:300]})
            return res
        paper_id = paper_entity_id(doc.doc_id)
        for r in out.relations:
            if not _evidence_supported(r.evidence, chunk.text):
                res.rejected.append(
                    {
                        "chunk": chunk.chunk_id,
                        "reason": "evidence not in text",
                        "relation": r.model_dump(),
                    }
                )
                continue
            if r.source_type == "Paper" or r.source.upper() == "THIS_PAPER":
                s_id, s_type = paper_id, "Paper"
            else:
                s_id, s_type = self.resolver.resolve(r.source, r.source_type), r.source_type
            t_id = self.resolver.resolve(r.target, r.target_type)
            if s_id == t_id or not _type_ok(r.relation, s_type, r.target_type):
                res.rejected.append(
                    {
                        "chunk": chunk.chunk_id,
                        "reason": "type constraint",
                        "relation": r.model_dump(),
                    }
                )
                continue
            res.relations.append(
                Relation(
                    s_id,
                    t_id,
                    r.relation,
                    round(max(0.1, min(0.95, r.confidence)), 3),
                    [chunk.chunk_id],
                    evidence_text=r.evidence[:400],
                    extractor=f"llm:{self.llm.model}",
                )
            )
        for e in out.entities:
            self.resolver.resolve(e.name, e.type)
        for eid in {r.source for r in res.relations} | {r.target for r in res.relations}:
            if eid in self.resolver.entities:
                res.entities[eid] = self.resolver.entities[eid]
        return res
