"""pack.adapters.glossary — a term→definition glossary as a model-free expert.

The simplest structured source there is: a controlled vocabulary where each entry is a term and its definition (with an
optional category/group). A glossary is ALREADY the exact structure the uniform strategy table wants, so the mapping is
exact — no heuristics, no formatting loss:
  * one citable passage per entry            "[glossary:<slug> · <Category>] <term> — <definition>"
  * `defines(passage, term)`                  → the define intent ("what is X" → its definition passage)
  * `items(term, category)`                   → the count/list inventory ("how many terms", "list the <category> terms")

So a glossary rides the whole model-free stack (define + count + list) with the EXISTING intent vocabulary — no new ergo
cue, no runtime change (REASONING.md: adding a strategy is adding rows, never code).

Source formats (auto-detected):
  .json         {term: definition}  OR  [{"term", "definition"|"def", "category"|"group"}]
  .tsv / .txt   `term <TAB> definition [<TAB> category]`  — one entry per line; blank lines and #comments skipped.
                A tab-free line falls back to the FIRST ` — ` (em dash) or `: ` separator (common glossary punctuation),
                so a plain "Term: definition" list also parses; a definition's own later colons are preserved.

Like the other document adapters, the DERIVED package is the committed artifact; a large source glossary need not be.
"""
from __future__ import annotations

import json
import re

from .base import Extraction, register

_WS = re.compile(r"\s+")
_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(term: str) -> str:
    return _SLUG.sub("-", term.strip().lower()).strip("-") or "term"


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _split_line(line: str):
    """A glossary line → (term, definition, category|None). Tab is authoritative; else the first ' — '/' - '/': '."""
    if "\t" in line:
        parts = [p.strip() for p in line.split("\t")]
        term, definition = parts[0], parts[1] if len(parts) > 1 else ""
        category = parts[2] if len(parts) > 2 and parts[2] else None
        return term, definition, category
    for sep in (" — ", " - ", ": "):                       # em dash, spaced hyphen, then colon (first occurrence only)
        if sep in line:
            term, definition = line.split(sep, 1)
            return term.strip(), definition.strip(), None
    return line.strip(), "", None                          # a bare term with no definition (dropped downstream)


def _entries_from_json(obj):
    """{term: def} or [{term, definition/def, category/group}] → iterable of (term, definition, category|None)."""
    if isinstance(obj, dict):
        for term, definition in obj.items():
            yield str(term), str(definition), None
    elif isinstance(obj, list):
        for e in obj:
            if not isinstance(e, dict):
                continue
            term = e.get("term") or e.get("name") or ""
            definition = e.get("definition") or e.get("def") or e.get("meaning") or ""
            category = e.get("category") or e.get("group") or None
            yield str(term), str(definition), (str(category) if category else None)
    else:
        raise ValueError("glossary JSON must be an object {term: def} or a list of {term, definition} entries")


def _entries(source):
    """Yield (term, definition, category|None) from the source file, format by extension (.json vs line-based)."""
    if source.lower().endswith(".json"):
        with open(source, encoding="utf-8") as f:
            yield from _entries_from_json(json.load(f))
        return
    with open(source, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            yield _split_line(line)


@register("glossary")
def adapt(source, *, prefix="glossary", default_group="terms", citation="glossary", **_):
    """A glossary file → Extraction (passages + exact defines + a term inventory). `default_group` names the group of
    uncategorized entries — the entity for a bare "list the <default_group>" query."""
    passages, defines, items, seen = [], [], [], set()
    for term, definition, category in _entries(source):
        term, definition = _clean(term), _clean(definition)
        if not (term and definition):                      # an entry needs both a term and a definition
            continue
        key = term.lower()
        if key in seen:                                    # first definition of a term wins (deterministic)
            continue
        seen.add(key)
        group = _clean(category) if category else default_group
        pid = f"{prefix}:{_slug(term)}"
        passages.append((f"{pid} · {group}", f"{term} — {definition}"))
        defines.append((pid, key))                         # define intent: "what is <term>" → this passage
        items.append((term, group))                        # count/list inventory: distinct terms, grouped by category
    if not passages:
        raise ValueError(f"glossary adapter: {source} yielded no usable entries (each needs a term and a definition)")
    return Extraction(passages, defines=defines, items=items, citation=citation)


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="pack.adapters.glossary",
                                 description="a term→definition glossary → citable passages + defines + inventory")
    ap.add_argument("source", help="glossary file (.json | .tsv | .txt)")
    ap.add_argument("out", help="output corpus path (.txt)")
    ap.add_argument("--prefix", default="glossary")
    a = ap.parse_args()
    ext = adapt(a.source, prefix=a.prefix)
    ext.write_corpus(a.out)
    print(f"[glossary] {a.prefix}: {ext.summary()} -> {a.out}")


if __name__ == "__main__":
    main()
