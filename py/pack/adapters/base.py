"""pack.adapters.base — the document-adapter CONTRACT + registry.

A "document expert" is built from a SOURCE through a pluggable adapter that yields one canonical object: `Extraction`.
Everything downstream (grounding, the uniform strategy table, the runtime) consumes the Extraction, so adding a new
source type — a spec, a PreTeXt book, a glossary, an arXiv/LaTeXML paper, a PDF — is just writing + registering an
adapter. The builder and runtime never change.

An adapter is a callable `adapt(source, **opts) -> Extraction` registered under a name via @register("name")."""
from __future__ import annotations

from dataclasses import dataclass, field

Pair = tuple[str, str]


@dataclass
class Extraction:
    """The canonical output every document adapter produces. Only `passages` is required; the rest populate the
    strategy tables when the source supports them (a spec has items; a textbook has defines + statements; plain prose
    has neither). `section` is "<id> · <facet>" so the id before the middle dot is the citation handle.
    """
    passages: list[Pair]                                 # (section, text)     citable knowledge → the grounding corpus
    defines: list[Pair] = field(default_factory=list)    # (passage_id, term)  → the define strategy (term → its passage)
    statements: list[Pair] = field(default_factory=list)  # (passage_id, name) → the theorem strategy (named statement)
    items: list[Pair] = field(default_factory=list)      # (name, group)       → count/list aggregates (the inventory)
    cards: list = field(default_factory=list)            # [{handle,title,summary,sections}] → the LIBRARIAN catalog
    citation: str = ""                                   # default source label (per-passage section overrides it)

    def write_corpus(self, path):
        """Write the passages as `[section] text` lines — the no-split grounding corpus the rest of pack reads."""
        with open(path, "w", encoding="utf-8") as f:
            for section, text in self.passages:
                f.write(f"[{section}] {text}\n")
        return path

    def summary(self):
        return (f"{len(self.passages)} passages, {len(self.defines)} defines, "
                f"{len(self.statements)} statements, {len(self.items)} items")

    @staticmethod
    def merge(exts: list["Extraction"]) -> "Extraction":
        """Combine N documents (of any adapter types) into ONE expert. Passage ids are adapter/document-namespaced so
        they never collide; defines/statements are deduped by entity (first document wins on a shared term — a cross-
        document tie-break refinement is a follow-up); items accumulate (count/list aggregates dedupe at materialize).
        Output order is DOCUMENT ENCOUNTER ORDER (deterministic for a fixed document list); no downstream step depends
        on passage order, so it isn't sorted."""
        passages, defines, statements, items, cards = [], [], [], [], []
        seen_def, seen_stmt = set(), set()
        for e in exts:
            passages.extend(e.passages)
            items.extend(e.items)
            cards.extend(e.cards)
            for pid, term in e.defines:
                if term not in seen_def:
                    seen_def.add(term)
                    defines.append((pid, term))
            for pid, name in e.statements:
                if name not in seen_stmt:
                    seen_stmt.add(name)
                    statements.append((pid, name))
        return Extraction(passages, defines=defines, statements=statements, items=items, cards=cards)


_REGISTRY = {}


def register(name):
    """Register a document adapter under `name`. The decorated callable is `adapt(source, **opts) -> Extraction`."""
    def deco(fn):
        _REGISTRY[name] = fn
        fn.adapter_name = name
        return fn
    return deco


def get(name):
    if name not in _REGISTRY:
        raise KeyError(f"unknown document adapter {name!r} — registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names():
    return sorted(_REGISTRY)
