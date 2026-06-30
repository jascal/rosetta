"""pack.adapters.pretext — PreTeXt / MathBook-XML open textbooks → citable passages + exact `defines` + theorem rows.

PreTeXt source TAGS its structure (unlike a PDF/spec): definitions, theorems, and the defined term itself are explicit
elements. So a textbook expert gets, for free and exactly (no heuristics, no formatting loss):
  * one citable passage per block (definition / theorem / proposition / lemma / corollary / example) + per prose <p>
  * `defines(passage, term)` from a <definition>'s <title> and from inline <term>/<define> tags  → the define strategy
  * `theorem(name, passage)` from titled theorems/propositions/…                                 → a NEW intent, no
    runtime change (the uniform answer(intent,entity,section) table already routes any intent — REASONING.md)

Handles both dialects seen in the wild: modern PreTeXt (<term>, <m>…</m>, xml:id) and older MathBook XML (<define>,
$…$, acro=). Regex/stream extraction (not a strict XML parse) so custom entities / xi:includes don't derail it.

Source = an open PreTeXt book's src/ dir (e.g. github.com/twjudson/aata, github.com/rbeezer/fcla). GFDL/CC-BY-SA are
copyleft — keep the DERIVED corpus out of the consuming repo; clone + build locally (see examples/pretext/README).
"""
import html
import os
import re

from .base import Extraction, register

_BLOCK_KINDS = ["definition", "theorem", "proposition", "lemma", "corollary", "example", "fact", "axiom"]
_STATEMENT = ["theorem", "proposition", "lemma", "corollary"]     # the "state the X" (theorem) intent applies to these

_TAGRE = {k: re.compile(rf"<{k}\b([^>]*)>(.*?)</{k}>", re.S) for k in _BLOCK_KINDS}
_ID = re.compile(r'(?:xml:id|acro)\s*=\s*"([^"]+)"')
_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
_TERM = re.compile(r"<(?:term|define)>(.*?)</(?:term|define)>", re.S)
_MATH = re.compile(r"<(?:m|me|men|md|mdn)>(.*?)</(?:m|me|men|md|mdn)>", re.S)   # <m>…</m> → keep the LaTeX
_TAG = re.compile(r"<[^>]+>")                                     # any remaining tag → drop
_ENT = re.compile(r"&[#a-zA-Z0-9]+;")                            # leftover entity → drop
_WS = re.compile(r"\s+")


def _clean(s):
    """Inline PreTeXt markup → readable plain text: keep math LaTeX + defined-term words, drop tags/xrefs/entities."""
    s = _MATH.sub(lambda m: " " + m.group(1).strip() + " ", s)   # math: keep the LaTeX source inline
    s = _TERM.sub(lambda m: m.group(1), s)                        # <term>/<define> → the word itself
    s = _TAG.sub(" ", s)                                          # every other tag → space
    s = html.unescape(s)
    s = _ENT.sub("", s)
    return _WS.sub(" ", s).strip()


def _title_text(inner):
    m = _TITLE.search(inner)
    return _clean(m.group(1)) if m else ""


def extract(src_dir, prefix):
    """A PreTeXt book's src/ → (passages, defines, theorems).
      passages : [(section, text)]   section = "<id> · <Kind>: <Title>"  (one per block + per prose paragraph)
      defines  : [(passage_id, term)]   exact term → its defining passage
      theorems : [(passage_id, name)]   titled theorem/proposition/… → the "theorem"/"state" intent
    passage_id is the part before ' · ' (the citation handle)."""
    passages, defines, theorems = [], [], []
    seen_term = set()
    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith(".xml"):
            continue
        raw = open(os.path.join(src_dir, fn), encoding="utf-8", errors="replace").read()
        used = []                                                # (start,end) spans consumed by block environments
        for kind in _BLOCK_KINDS:
            for m in _TAGRE[kind].finditer(raw):
                attrs, inner = m.group(1), m.group(2)
                idm = _ID.search(attrs)
                bid = idm.group(1) if idm else f"{fn[:-4]}-{kind}-{len(passages)}"
                pid = f"{prefix}:{kind}:{bid}"
                title = _title_text(inner)
                body = _clean(_TITLE.sub("", inner, count=1))
                if len(body) < 25:
                    continue
                sec = f"{pid} · {kind.capitalize()}{': ' + title if title else ''}"
                passages.append((sec, body))
                used.append((m.start(), m.end()))
                # exact defines: a definition's title is the term; inline <term>/<define> anywhere is a defined term
                terms = set()
                if kind == "definition" and title:
                    terms.add(title.lower())
                for t in _TERM.findall(inner):
                    tt = _clean(t).lower()
                    if 2 < len(tt) < 40:
                        terms.add(tt)
                for t in terms:
                    if t not in seen_term:
                        seen_term.add(t)
                        defines.append((pid, t))
                if kind in _STATEMENT and title:                 # "state the <named> theorem" → this passage
                    theorems.append((pid, title.lower()))
        # prose paragraphs OUTSIDE any block (for grounding/retrieval); also mine their inline defined terms
        masked = raw
        for a, b in used:
            masked = masked[:a] + (" " * (b - a)) + masked[b:]   # blank out block spans so <p>s aren't double-counted
        for i, pm in enumerate(re.finditer(r"<p\b[^>]*>(.*?)</p>", masked, re.S)):
            body = _clean(pm.group(1))
            if len(body) < 40:
                continue
            pid = f"{prefix}:p:{fn[:-4]}-{i}"
            passages.append((f"{pid} · {fn[:-4]}", body))
            for t in _TERM.findall(pm.group(1)):                 # a term first defined in running prose
                tt = _clean(t).lower()
                if 2 < len(tt) < 40 and tt not in seen_term:
                    seen_term.add(tt)
                    defines.append((pid, tt))
    return passages, defines, theorems


@register("pretext")
def adapt(source, *, prefix="book", citation="PreTeXt open textbook", **_):
    """Document-adapter entry: a PreTeXt book's src/ → Extraction (passages + exact defines + named theorems)."""
    passages, defines, statements = extract(source, prefix)
    return Extraction(passages, defines=defines, statements=statements, citation=citation)


def to_corpus(src_dir, out, prefix, *, citation="PreTeXt open textbook"):
    """Write <out>/<prefix>_corpus.txt (citable passages) + return (corpus_path, defines, theorems) for the strategy
    builder. Mirrors normrules/riscv_prose: the derived corpus is the build input; the book source is not committed."""
    os.makedirs(out, exist_ok=True)
    passages, defines, theorems = extract(src_dir, prefix)
    if not passages:
        raise ValueError(f"pretext adapter: 0 passages from {src_dir} — is this a PreTeXt book's src/ dir?")
    corpus = os.path.join(out, f"{prefix}_corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for sec, text in passages:
            f.write(f"[{sec}] {text}\n")
    return corpus, defines, theorems


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="pack.adapters.pretext")
    ap.add_argument("src_dir", help="a PreTeXt book's src/ directory")
    ap.add_argument("out")
    ap.add_argument("prefix", help="citation namespace, e.g. 'aata'")
    a = ap.parse_args()
    corpus, defines, theorems = to_corpus(a.src_dir, a.out, a.prefix)
    print(f"[pretext] {a.prefix}: corpus -> {corpus}; {len(defines)} defines, {len(theorems)} theorems")


if __name__ == "__main__":
    main()
