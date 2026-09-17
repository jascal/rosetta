"""pack.adapters.nh_legal — a statute / legal code (NH RSA-style §-numbered text) → cited passages + defines + inventory.

Closes the TODO in examples/nh_family (SOURCES.md): a structured legal source → one citable passage per section, in the
documented cite-id scheme `section = "RSA <chap>:<sec> · <chapter title>"` so the citation handle before the middle dot
is the pinpoint legal cite (`RSA 461-A:6`) that /lookup resolves. Maps onto the uniform strategy table with the EXISTING
intents (no new ergo cue, no runtime change), which for law is deliberately SAFER than any model synthesis:
  * one citable passage per statute section  "[RSA <chap>:<sec> · <Chapter Title>] <section text>"
  * `defines(passage, term)`                  → define intent, from a Definitions section's `"term" means …` /
                                                `"term" shall mean …` — statutes DEFINE terms verbatim (high precision)
  * `items("RSA <chap>:<sec>", "RSA <chap>")` → count/list ("how many sections in RSA 458", "list the RSA 458 sections")

Not legal advice: this is retrieval over public-domain primary sources — the answer is the verbatim provision + its cite,
or abstain. Source = a directory of RSA chapter text files (or a single statutes.txt); the DERIVED package is the
committed artifact (the raw statute text is fetched per SOURCES.md), mirroring normrules / riscv_prose.
"""
from __future__ import annotations

import os
import re

from .base import Extraction, register

_WS = re.compile(r"\s+")
# a section header: "461-A:6 Parental Rights. – <text>" — RSA chap (digits[+ '-A']) ':' sec, a title, an en/em-dash, text.
_SEC = re.compile(r"^\s*(?:Section\s+)?(\d+[A-Z]?(?:-[A-Z])?:\d+(?:-[a-z])?)\s+(.+?)\.\s*[–—-]+\s*(.*)$")
# a bare "Section 461-A:6" marker line (some sources put the number on its own line above the titled body).
_SECMARK = re.compile(r"^\s*Section\s+(\d+[A-Z]?(?:-[A-Z])?:\d+(?:-[a-z])?)\s*$")
_CHAPTER = re.compile(r"^\s*CHAPTER\s+(\d+[A-Z]?(?:-[A-Z])?)\b\s*(.*)$", re.I)
# a statutory definition: `"parent" means …` / `"income" shall mean …` / `"court" includes …` — quoted term + operator.
_QDEF = re.compile(r'["“]([^"”]{2,60})["”]\s+(?:means|shall mean|includes|shall include|has the meaning)\b', re.I)


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _chap_of(cite: str) -> str:
    return cite.split(":", 1)[0]                              # "461-A:6" → "461-A"


def _iter_files(source):
    if os.path.isdir(source):
        for fn in sorted(os.listdir(source)):
            p = os.path.join(source, fn)
            if os.path.isfile(p) and fn.lower().endswith((".txt", ".text")):
                yield p
    else:
        yield source


def _parse(path, chap_titles):
    """One statute text file → [(cite, section_title, chapter, text)]. Tracks the current CHAPTER title for the facet."""
    out = []
    cur_chapter_title = ""
    cite = title = None
    buf = []
    pending_mark = None

    def flush():
        nonlocal cite, title, buf
        if cite is not None:
            chap = _chap_of(cite)
            facet = chap_titles.get(chap) or cur_chapter_title or title or f"RSA {chap}"
            out.append((cite, title or "", facet, _clean(" ".join(buf))))
        cite, title, buf = None, None, []

    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.rstrip("\n")
        ch = _CHAPTER.match(line)
        if ch:
            flush()
            cur_chapter_title = _clean(ch.group(2)) or cur_chapter_title
            if ch.group(2):
                chap_titles.setdefault(ch.group(1), _clean(ch.group(2)))
            continue
        mark = _SECMARK.match(line)
        if mark:                                             # a standalone "Section X:Y" marker → next titled line is it
            pending_mark = mark.group(1)
            continue
        m = _SEC.match(line)
        if m:
            flush()
            cite, title = (pending_mark or m.group(1)), _clean(m.group(2))
            pending_mark = None
            buf = [m.group(3)] if m.group(3).strip() else []
            continue
        if cite is not None:
            buf.append(line.strip())
    flush()
    return out


@register("nh_legal")
def adapt(source, *, prefix="RSA", citation="NH RSA (public domain)", **_):
    """A statute source (dir of chapter files or one .txt) → Extraction (cited section passages + defines + inventory)."""
    chap_titles = {}
    sections = []
    for path in _iter_files(source):
        sections.extend(_parse(path, chap_titles))
    passages, defines, items, seen_cite, seen_term = [], [], [], set(), set()
    for cite, title, facet, text in sections:
        if cite in seen_cite or not text:
            continue
        seen_cite.add(cite)
        pid = f"{prefix} {cite}"                              # the pinpoint legal cite IS the handle (before the ' · ')
        body = f"{title}. {text}" if title else text
        passages.append((f"{pid} · {facet}", body))
        items.append((f"{prefix} {cite}", f"{prefix} {_chap_of(cite)}"))   # list/count by chapter
        for m in _QDEF.finditer(body):                       # verbatim statutory definitions (quoted term)
            term = _clean(m.group(1)).lower()
            if term and term not in seen_term:
                seen_term.add(term)
                defines.append((pid, term))
    if not passages:
        raise ValueError(f"nh_legal adapter: 0 sections from {source} — expected RSA-style '<chap>:<sec> Title. – text'.")
    return Extraction(passages, defines=defines, items=items, citation=citation)


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="pack.adapters.nh_legal",
                                 description="a statute / legal code (RSA §-numbered) → cited passages + defines + inventory")
    ap.add_argument("source", help="a statute .txt or a dir of chapter files")
    ap.add_argument("out", help="output corpus path (.txt)")
    ap.add_argument("--prefix", default="RSA")
    a = ap.parse_args()
    ext = adapt(a.source, prefix=a.prefix)
    ext.write_corpus(a.out)
    print(f"[nh_legal] {ext.summary()} -> {a.out}")


if __name__ == "__main__":
    main()
