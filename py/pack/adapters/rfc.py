"""pack.adapters.rfc — an IETF RFC (internet-standard plain-text spec) → citable passages + defines + a section index.

RFCs are the canonical fixed-layout plain text from rfc-editor.org: a front-matter block, a table of contents, then
`N.`, `N.M.` numbered sections whose paragraphs are indented, with page furniture (form-feed + a running header/footer
carrying `[Page k]`) between pages. This adapter strips the furniture and maps the structure onto the uniform strategy
table with the EXISTING intents (no new ergo cue, no runtime change):
  * one citable passage per paragraph        "[rfc<N>:<sec> · <Section Title>] <text>"   (page furniture removed)
  * `defines(passage, term)`                  → the define intent, from (a) QUOTED-term definitions (`A "resource" is …`)
                                                and (b) the genus–differentia form a spec opens a paragraph with,
                                                `An <term> is/are …` (indefinite article + single-word subject only —
                                                e.g. "An object is an unordered collection…" → object). Both are
                                                precision-first: ordinary mid-sentence words never become entities.
  * `items(section, "sections")`              → count/list ("how many sections", "list the sections")

Normative RFC-2119 requirement language (MUST/SHOULD/MAY) is preserved verbatim in the passages (it is what an RFC
expert must cite); it is not turned into a second inventory (that would make "how many …" ambiguous).

Source = an RFC `.txt` (e.g. `curl -O https://www.rfc-editor.org/rfc/rfc9110.txt`). The DERIVED package is committable;
RFCs themselves are freely publishable but large — clone/fetch + build locally.
"""
from __future__ import annotations

import os
import re

from .base import Extraction, register

_WS = re.compile(r"\s+")
# a section header at column 0: "1.  Introduction" / "2.1.6.  Idempotent Methods" (number, then a title with letters).
_SEC = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*?)\s*$")
_TOC_DOTS = re.compile(r"\.{3,}\s*\d+\s*$")                    # a table-of-contents dot-leader line ("... 12")
_PAGE_FOOT = re.compile(r"\[Page\s+\d+\]\s*$")                 # "Fielding, et al.   Standards Track   [Page 5]"
_RFC_NO = re.compile(r"Request for Comments:\s*(\d+)")
# quoted-term definition: `"resource" is` / `"origin server" refers to` / `"cache" means` — quoted term + a copula.
_QDEF = re.compile(r'"([A-Za-z][A-Za-z0-9 /_.\-]{1,48})"\s+(?:is|are|refers to|means|denotes|represents)\b')
# genus–differentia definition opening a paragraph: `An object is …`, `A string is …` (indefinite article + one word).
_ADEF = re.compile(r"^An?\s+([a-z][a-z0-9-]{1,30})\s+(?:is|are)\b")


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _strip_furniture(text: str):
    """Drop page headers/footers and form feeds; return the content lines. A footer carries '[Page N]'; the line right
    after a form feed is the next page's running header (RFC number / title / date) — both are removed."""
    out, after_ff = [], False
    for raw in text.split("\n"):
        line = raw.rstrip()
        if "\f" in raw:                                       # form feed: the page break itself
            after_ff = True
            continue
        if _PAGE_FOOT.search(line):                           # page footer
            continue
        if after_ff:                                          # the running header immediately after a page break
            after_ff = False
            if line.strip():
                continue
        out.append(line)
    return out


def _rfc_number(lines, source):
    m = next((_RFC_NO.search(ln) for ln in lines[:60] if _RFC_NO.search(ln)), None)
    if m:
        return m.group(1)
    base = os.path.basename(source).lower()
    fm = re.search(r"rfc[_-]?(\d+)", base)
    return fm.group(1) if fm else "0"


def _is_section_header(line: str):
    """A real section header (col-0 number + title), not a TOC dot-leader line and not an ordinary numbered list item."""
    if line.startswith(" ") or _TOC_DOTS.search(line):
        return None
    m = _SEC.match(line)
    if not m:
        return None
    num, title = m.group(1), m.group(2)
    if not re.search(r"[A-Za-z]", title):                     # a title must contain letters (else it's data, e.g. "1. 2")
        return None
    return num, _clean(title)


@register("rfc")
def adapt(source, *, prefix=None, citation=None, **_):
    """An RFC .txt → Extraction (per-paragraph passages + quoted-term defines + a section inventory)."""
    with open(source, encoding="utf-8", errors="replace") as f:
        lines = _strip_furniture(f.read())
    num = _rfc_number(lines, source)
    prefix = prefix or f"rfc{num}"
    citation = citation or f"IETF RFC {num}"

    passages, items, seen_sec = [], [], set()
    cur_num, cur_title, started = "0", "Front Matter", False
    buf = []

    def flush():
        if not buf:
            return
        text = _clean(" ".join(buf))
        buf.clear()
        if started and len(text) >= 40:                       # skip the front matter / TOC; keep real paragraphs
            n = len([p for p in passages if p[0].startswith(f"{prefix}:{cur_num}#")]) + 1
            passages.append((f"{prefix}:{cur_num}#{n} · {cur_title}", text))

    for line in lines:
        hdr = _is_section_header(line)
        if hdr:
            flush()
            cur_num, cur_title = hdr
            if cur_title.lower() == "table of contents":      # the TOC section itself carries no passages
                started = False
                continue
            started = True                                    # the first real section starts the body
            if cur_num not in seen_sec:
                seen_sec.add(cur_num)
                items.append((f"{cur_num}. {cur_title}", "sections"))
            continue
        if not line.strip():                                  # blank line = paragraph boundary
            flush()
            continue
        buf.append(line.strip())
    flush()

    if not passages:
        raise ValueError(f"rfc adapter: 0 passages from {source} — is this a canonical RFC .txt (numbered sections)?")

    # defines: quoted-term definitions, first occurrence wins, ordinary words excluded (quoted only).
    defines, seen_term = [], set()
    for pid_facet, text in passages:
        pid = pid_facet.split(" · ", 1)[0]
        terms = [m.group(1) for m in _QDEF.finditer(text)]    # quoted-term definitions anywhere in the paragraph
        am = _ADEF.match(text)                                # + a genus–differentia opening ("An object is …")
        if am:
            terms.append(am.group(1))
        for t in terms:
            term = _clean(t).lower()
            if term and term not in seen_term:
                seen_term.add(term)
                defines.append((pid, term))
    return Extraction(passages, defines=defines, items=items, citation=citation)


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="pack.adapters.rfc",
                                 description="an IETF RFC .txt → citable passages + defines + a section inventory")
    ap.add_argument("source", help="an RFC .txt (e.g. rfc9110.txt)")
    ap.add_argument("out", help="output corpus path (.txt)")
    ap.add_argument("--prefix", default=None)
    a = ap.parse_args()
    ext = adapt(a.source, prefix=a.prefix)
    ext.write_corpus(a.out)
    print(f"[rfc] {ext.summary()} -> {a.out}")


if __name__ == "__main__":
    main()
