"""Bounded multi-hop traversal, path scoring and path serialization.

Path score for a path p with edges e_1..e_h starting at seed s:

    score(p) = link(s) * prod_i conf(e_i) * decay^(h-1) * prod_i bonus(e_i) * connect(p)

* ``decay`` (< 1) penalizes longer paths — each hop compounds extraction error.
* ``bonus`` = ``relation_cue_bonus`` when the edge type matches a relation cue in the query
  ("who introduced" -> INTRODUCES/AUTHORED_BY); 1 otherwise.
* ``connect`` = 2 when the path ends at a *different* seed (paths that connect two query
  entities answer comparison / relationship questions directly).

Traversal is a beam search: at each depth keep the best ``beam`` partial paths; stop after
``max_expansions`` neighbor expansions (hard cost bound). Hub nodes (very high degree, e.g.
"BERT" or "Wikipedia") are expanded but their neighbor lists are truncated to the strongest
``max_neighbors`` edges.
"""

from __future__ import annotations

from collections.abc import Sequence

from agvrag.interfaces import GraphStore
from agvrag.schemas import GraphPath

# Query cue words -> relation types (shared with the query analyzer).
RELATION_CUES: dict[str, tuple[str, ...]] = {
    "introduc": ("INTRODUCES",),
    "propos": ("INTRODUCES",),
    "invent": ("INTRODUCES",),
    "author": ("AUTHORED_BY",),
    "wrote": ("AUTHORED_BY",),
    "who": ("AUTHORED_BY", "INTRODUCES"),
    "cite": ("CITES",),
    "citing": ("CITES",),
    "reference": ("CITES",),
    "evaluat": ("EVALUATES_ON",),
    "benchmark": ("EVALUATES_ON",),
    "dataset": ("EVALUATES_ON",),
    "tested on": ("EVALUATES_ON",),
    "improv": ("IMPROVES",),
    "outperform": ("IMPROVES",),
    "better": ("IMPROVES",),
    "based on": ("BASED_ON", "EXTENDS"),
    "derived": ("BASED_ON", "EXTENDS"),
    "build on": ("BASED_ON", "EXTENDS"),
    "built on": ("BASED_ON", "EXTENDS"),
    "extend": ("EXTENDS", "BASED_ON"),
    "variant": ("EXTENDS", "BASED_ON"),
    "use": ("USES",),
    "using": ("USES",),
    "employ": ("USES",),
    "applied": ("APPLIED_TO",),
    "task": ("APPLIED_TO",),
    "influenc": ("INFLUENCES",),
}


def relation_cues_for(query: str) -> list[str]:
    q = query.lower()
    out: list[str] = []
    for cue, rels in RELATION_CUES.items():
        if cue in q:
            for r in rels:
                if r not in out:
                    out.append(r)
    return out


class GraphTraverser:
    def __init__(
        self,
        store: GraphStore,
        hop_decay: float = 0.6,
        cue_bonus: float = 1.5,
        max_expansions: int = 2000,
        beam: int = 200,
        max_neighbors: int = 60,
    ) -> None:
        self.store = store
        self.hop_decay = hop_decay
        self.cue_bonus = cue_bonus
        self.max_expansions = max_expansions
        self.beam = beam
        self.max_neighbors = max_neighbors
        self.last_expansions = 0

    def paths(
        self,
        seeds: dict[str, float],
        max_hops: int,
        relation_types: Sequence[str] | None = None,
        cue_relations: Sequence[str] | None = None,
        max_paths: int = 20,
    ) -> list[GraphPath]:
        cues = set(cue_relations or [])
        frontier: list[GraphPath] = [GraphPath([s], [], [], score) for s, score in seeds.items()]
        results: list[GraphPath] = []
        expansions = 0
        for depth in range(1, max_hops + 1):
            nxt: list[GraphPath] = []
            for p in frontier:
                if expansions >= self.max_expansions:
                    break
                tail = p.nodes[-1]
                for rel, other, direction in self.store.neighbors(tail, relation_types)[
                    : self.max_neighbors
                ]:
                    expansions += 1
                    if other in p.nodes:
                        continue  # no cycles
                    if any(e.key == rel.key for e in p.edges):
                        continue
                    bonus = self.cue_bonus if rel.type in cues else 1.0
                    score = (
                        p.score * rel.confidence * bonus * (self.hop_decay if depth > 1 else 1.0)
                    )
                    np_ = GraphPath(
                        [*p.nodes, other], [*p.edges, rel], [*p.directions, direction], score
                    )
                    final = score * (2.0 if other in seeds and other != p.nodes[0] else 1.0)
                    results.append(GraphPath(np_.nodes, np_.edges, np_.directions, final))
                    nxt.append(np_)
            nxt.sort(key=lambda p: -p.score)
            frontier = nxt[: self.beam]
            if not frontier:
                break
        self.last_expansions = expansions
        # Deduplicate paths with identical edge sets (e.g. found from both seeds).
        best: dict[frozenset[tuple[str, str, str]], GraphPath] = {}
        for p in results:
            key = frozenset(e.key for e in p.edges)
            if key not in best or p.score > best[key].score:
                best[key] = p
        ordered = sorted(best.values(), key=lambda p: (-p.score, p.hops, p.nodes))
        return ordered[:max_paths]


def serialize_path(path: GraphPath, store: GraphStore) -> str:
    """Human/LLM-readable path, e.g. ``LoRA [Method] <-INTRODUCES- LoRA: Low-Rank ... [Paper]``."""

    def label(eid: str) -> str:
        e = store.get_entity(eid)
        if e is None:
            return eid
        name = e.name if len(e.name) <= 80 else e.name[:77] + "..."
        year = e.properties.get("year")
        suffix = f", {year}" if e.type == "Paper" and year else ""
        return f"{name} [{e.type}{suffix}]"

    parts = [label(path.nodes[0])]
    for rel, direction, node in zip(path.edges, path.directions, path.nodes[1:], strict=True):
        arrow = f" -{rel.type}-> " if direction == "out" else f" <-{rel.type}- "
        parts.append(arrow + label(node))
    return "".join(parts)
