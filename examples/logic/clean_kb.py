#!/usr/bin/env python
"""examples/logic/clean_kb.py — a lightweight OLP cleaner: raw PDF-to-text dump -> `[OLP §N.N Title] prose` lines.

The P1 ablation used the RAW logic_kb.txt (no section handles, TOC noise) as the document baseline — a weak retrieval
tier. This gives retrieval a FAIR shot: (1) drop TOC / boilerplate / page-number lines, (2) attach a rolling section
citation (`[OLP §N.N Title]`) to every prose line so retrieval can CITE under --require-citation. Not a real adapter —
a heuristic strong enough to test whether the model-tier win survives a cleaned document baseline (EXPERTS.md P1-b).

Usage: .venv/bin/python examples/logic/clean_kb.py  ->  writes examples/logic/logic_kb_clean.txt
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "logic_kb.txt")
OUT = os.path.join(HERE, "logic_kb_clean.txt")

HEADER = re.compile(r"^(\d+\.\d+)\s+([A-ZÀ-Þ].*)$")          # "7.2 Propositional Formulas ..."
DIGITS = re.compile(r"[0-9]")
BOILER = re.compile(r"Release\s*:|CONTENTS|Open Logic (Project|Text)|Creative Commons|licensed under|Revision:")
TOC_DOTS = re.compile(r"\.\s+\.\s+\.|\.{4,}|\s\.\s+\d+\s*$")   # dot-leaders / trailing " . <page>"
ENDS_DOT = re.compile(r"\s\.\s*$")                            # ends in " ." — a dotted TOC / heading fragment
LEADING_PAGE = re.compile(r"^\d+\s+\d+\.\d+\b")               # "<page> N.N Title" — a TOC entry


def _is_toc(line):
    """A TOC / heading fragment: real sentences don't end in ' .' and don't start '<page> N.N'."""
    return bool(ENDS_DOT.search(line) or LEADING_PAGE.match(line) or TOC_DOTS.search(line))


def _is_boilerplate(line):
    if len(line) < 30:
        return True
    if BOILER.search(line):
        return True
    digitfrac = len(DIGITS.findall(line)) / max(1, len(line))
    return digitfrac > 0.20                                    # page-number / index runs


def _title(rest):
    """First few Capitalized words of a section title (up to where prose begins or 6 words)."""
    words = rest.split()
    keep = []
    for w in words[:6]:
        keep.append(w)
        if w.endswith((".", ":")) or (len(keep) >= 2 and w[:1].islower()):
            break
    return " ".join(keep).rstrip(".:").strip()[:50]


def clean():
    section, title = "0.0", "Front matter"
    out = []
    for line in open(RAW, encoding="utf-8"):
        line = " ".join(line.split())                          # normalize whitespace
        if not line:
            continue
        if _is_toc(line):                                      # TOC / heading fragment — skip, keep section context
            continue
        m = HEADER.match(line)
        # a REAL section start = header token + substantial prose on the line (not a bare TOC entry)
        if m and len(line) > 55:
            section, title = m.group(1), _title(m.group(2))
            out.append((f"OLP §{section} {title}", m.group(2)))  # keep title+prose as the section's lead passage
            continue
        if _is_boilerplate(line):
            continue
        out.append((f"OLP §{section} {title}", line))          # prose under the current section
    with open(OUT, "w", encoding="utf-8") as w:
        for sec, text in out:
            w.write(f"[{sec}] {text}\n")
    return len(out)


if __name__ == "__main__":
    n = clean()
    print(f"[clean] {n} cited passages -> {OUT}")
