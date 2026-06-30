"""pack.adapters.pretext — PreTeXt / MathBook-XML open textbooks → citable passages + exact `defines` + theorem rows.

PreTeXt source TAGS its structure (unlike a PDF/spec): definitions, theorems, and the defined term itself are explicit
elements. So a textbook expert gets, for free and exactly (no heuristics, no formatting loss):
  * one citable passage per block (definition / theorem / proposition / lemma / corollary / example) + per prose <p>
  * `defines(passage, term)` from a <definition>'s <title> and from inline <term>/<define> tags  → the define strategy
  * `theorem(name, passage)` from titled theorems/propositions/…                                 → a NEW intent, no
    runtime change (the uniform answer(intent,entity,section) table already routes any intent — REASONING.md)

Parsed with stdlib ElementTree (each src file individually, so xi:includes need not be expanded; entities are standard).
Handles both dialects seen in the wild: modern PreTeXt (<term>, <m>…</m>, xml:id) and older MathBook XML (<define>,
acro=). Inline math (<m>/<me>/<md>) carries its LaTeX as element text, so itertext() keeps it; <math> MathML is a
LaTeXML thing, handled by the latexml adapter instead.

Source = an open PreTeXt book's src/ dir (e.g. github.com/twjudson/aata, github.com/rbeezer/fcla). GFDL/CC-BY-SA are
copyleft — keep the DERIVED corpus out of the consuming repo; clone + build locally (see examples/pretext/README).
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

from .base import Extraction, register

_XMLID = "{http://www.w3.org/XML/1998/namespace}id"
_BLOCK_KINDS = {"definition", "theorem", "proposition", "lemma", "corollary", "example", "fact", "axiom"}
_STATEMENT_KINDS = {"theorem", "proposition", "lemma", "corollary"}    # the "state the X" (theorem) intent
_SECTIONING = {"part", "chapter", "section", "subsection", "subsubsection"}
_TERM_TAGS = {"term", "define"}
_WS = re.compile(r"\s+")


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else tag    # strip any XML namespace


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()                                    # ET already unescapes entities; collapse whitespace


def _child_title(elem) -> str:
    for c in elem:
        if _local(c.tag) == "title":
            return _clean(" ".join(c.itertext()))
    return ""


def _body_without_title(elem) -> str:
    parts = [elem.text or ""]
    for c in elem:
        if _local(c.tag) != "title":
            parts.append(" ".join(c.itertext()))
        parts.append(c.tail or "")
    return _clean(" ".join(parts))


def _terms(elem):
    for t in elem.iter():
        if _local(t.tag) in _TERM_TAGS:
            w = _clean(" ".join(t.itertext())).lower()
            if 2 < len(w) < 40:
                yield w


def _walk(elem, prefix, section, st):
    """Recursive extraction. `st` accumulates passages/defines/statements; `section` is the enclosing section title."""
    tag = _local(elem.tag)
    if tag in _SECTIONING:
        for c in elem:                                               # recurse with this section's own title in scope
            _walk(c, prefix, _child_title(elem) or section, st)
        return
    if tag in _BLOCK_KINDS:
        title, body = _child_title(elem), _body_without_title(elem)
        if len(body) >= 25:
            st["n"] += 1
            bid = elem.get(_XMLID) or elem.get("acro") or f"{tag}-{st['n']}"
            pid = f"{prefix}:{tag}:{bid}"
            st["passages"].append((f"{pid} · {tag.capitalize()}{': ' + title if title else ''}", body))
            terms = set(_terms(elem))
            if tag == "definition" and title:
                terms.add(title.lower())
            for w in terms:
                if w not in st["seen"]:
                    st["seen"].add(w)
                    st["defines"].append((pid, w))
            if tag in _STATEMENT_KINDS and title:
                st["statements"].append((pid, title.lower()))
        return                                                       # a block's <p>s belong to it — don't recurse
    if tag == "p":
        body = _clean(" ".join(elem.itertext()))
        if len(body) >= 40:
            st["n"] += 1
            pid = f"{prefix}:p:{st['file']}-{st['n']}"
            st["passages"].append((f"{pid} · {section}", body))
            for w in _terms(elem):                                   # a term first defined in running prose
                if w not in st["seen"]:
                    st["seen"].add(w)
                    st["defines"].append((pid, w))
        return                                                       # whole paragraph captured (incl. nested lists)
    for c in elem:
        _walk(c, prefix, section, st)


def extract(src_dir, prefix):
    """A PreTeXt book's src/ → (passages, defines, statements). passage_id is the part before ' · '."""
    st = {"passages": [], "defines": [], "statements": [], "seen": set(), "n": 0, "file": ""}
    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith(".xml"):
            continue
        try:
            root = ET.parse(os.path.join(src_dir, fn)).getroot()
        except ET.ParseError as e:
            print(f"[pretext] skip {fn} (parse error: {str(e)[:60]})")
            continue
        st["file"] = fn[:-4]
        _walk(root, prefix, prefix, st)
    return st["passages"], st["defines"], st["statements"]


@register("pretext")
def adapt(source, *, prefix="book", citation="PreTeXt open textbook", **_):
    """Document-adapter entry: a PreTeXt book's src/ → Extraction (passages + exact defines + named theorems)."""
    passages, defines, statements = extract(source, prefix)
    if not passages:
        raise ValueError(f"pretext adapter: 0 passages from {source} — is this a PreTeXt book's src/ dir?")
    return Extraction(passages, defines=defines, statements=statements, citation=citation)


def to_corpus(src_dir, out, prefix, *, citation="PreTeXt open textbook"):
    """Write <out>/<prefix>_corpus.txt (citable passages) + return (corpus_path, defines, theorems). The derived corpus
    is the build input; the book source is not committed (mirrors normrules / riscv_prose)."""
    os.makedirs(out, exist_ok=True)
    ext = adapt(src_dir, prefix=prefix, citation=citation)
    corpus = os.path.join(out, f"{prefix}_corpus.txt")
    ext.write_corpus(corpus)
    return corpus, ext.defines, ext.statements


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="pack.adapters.pretext")
    ap.add_argument("src_dir", help="a PreTeXt book's src/ directory")
    ap.add_argument("out")
    ap.add_argument("prefix", help="citation namespace, e.g. 'aata'")
    a = ap.parse_args()
    corpus, defines, statements = to_corpus(a.src_dir, a.out, a.prefix)
    print(f"[pretext] {a.prefix}: corpus -> {corpus}; {len(defines)} defines, {len(statements)} theorems")


if __name__ == "__main__":
    main()
