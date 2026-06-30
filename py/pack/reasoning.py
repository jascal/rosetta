"""pack.reasoning — the authored-deductive tier at BUILD time (REASONING.md), first use: count/list aggregates.

Run ergo's verified Datalog rules over facts EXTRACTED from the corpus, via souffle, and materialize the derived facts
as **cited KB passages** — so the thin runtime can answer "how many / list all X" by *retrieving a stated, derived
fact*, not by counting at query time (no runtime engine). The aggregates (ergo/aggregate.dl) are CLOSED-WORLD: the
count is "how many EXIST in the build-time fact base" — sound relative to the extracted inventory, not a universal claim.

The inventory extractor here is RISC-V-specific (the spec's `insn:X[]` markup); the materialize/passages machinery is
generic over any `item(name, group)` inventory + any ergo rule file.
"""
import os
import re
import shutil
import subprocess
import tempfile

INSN = re.compile(r'insn:([A-Za-z0-9._]+)\[\]')                    # the spec's insn:X[] cross-reference markup
SEC = re.compile(r'^\[([^\]]+)\]')                                 # leading "[norm:id · Facet]" on a corpus line


def instruction_inventory(rules_txt):
    """Extract (instruction, extension) pairs from a riscv rules corpus: instructions = the distinct insn:X[] in the
    text; extension = the facet (after the '·') of the rule's section. Deduped (a set)."""
    inv = set()
    for line in open(rules_txt, encoding="utf-8"):
        facet = ""
        m = SEC.match(line)
        if m:
            s = m.group(1)
            dot = s.find("·")
            facet = (s[dot + 1:] if dot >= 0 else s).strip()
        for insn in INSN.findall(line):
            inv.add((insn.lower(), facet or "unknown"))
    return sorted(inv)


def materialize(items, rules_dl):
    """Run souffle(rules_dl) over the item(name,group) facts → {total, groups:{g:n}, items:[(name,group)]}.
    rules_dl must declare `.input item` and `.output total_count/group_count/item` (ergo/aggregate.dl)."""
    souffle = shutil.which("souffle")
    if not souffle:
        raise RuntimeError("souffle not found — the authored-reasoning (aggregate) tier needs souffle at build time")
    with tempfile.TemporaryDirectory() as d:
        fdir, odir = os.path.join(d, "facts"), os.path.join(d, "out")
        os.makedirs(fdir)
        os.makedirs(odir)
        with open(os.path.join(fdir, "item.facts"), "w", encoding="utf-8") as f:
            for name, group in items:
                f.write(f"{name}\t{group}\n")
        subprocess.run([souffle, rules_dl, "-F", fdir, "-D", odir], check=True)

        def rd(rel):
            p = os.path.join(odir, rel + ".csv")
            return [ln.rstrip("\n").split("\t") for ln in open(p, encoding="utf-8")] if os.path.exists(p) else []

        tot = rd("total_count")
        return {
            "total": int(tot[0][0]) if tot else 0,
            "groups": {r[0]: int(r[1]) for r in rd("group_count")},
            "items": [(r[0], r[1]) for r in rd("item")],
        }


def inventory_passages(mat, *, label="instruction", prefix="riscv:inventory", collection="the collection"):
    """Materialized counts/list → cited KB passages, section = "id · Facet" (so cite-as-handle + /lookup work). One
    'total' passage + one per group (carrying the enumerated names — the count ships with what it counted). Domain-
    neutral wording (`collection` names the whole, e.g. "the RISC-V ISA" / "the catalog") so the same ergo aggregate
    serves any inventory — instructions, documents, anything."""
    out = [(f"{prefix}:total · {label.capitalize()} inventory",
            f"There are {mat['total']} distinct {label}s in {collection} (closed-world: counted at build time).")]
    by_group = {}
    for name, g in mat["items"]:
        by_group.setdefault(g, set()).add(name)
    for g, n in sorted(mat["groups"].items()):
        names = ", ".join(sorted(by_group.get(g, ())))
        out.append((f"{prefix}:{g} · {g}", f"{g} contains {n} {label}s: {names}."))
    return out


_HEAD = re.compile(r'^\[([^\]]+)\]')                               # leading "[id · Chapter › Section]" on a corpus line
_PAREN = re.compile(r'\(([a-z][a-z0-9.]{2,})\)')                   # a parenthesized heading abbrev (≥3 chars), e.g. "(misa)"


def _sections(corpus):
    """Yield (passage_id, section_title_lc) from either a corpus FILE path or an iterable of (section, text) passages."""
    if isinstance(corpus, str):
        rows = (_HEAD.match(ln).group(1) for ln in open(corpus, encoding="utf-8") if _HEAD.match(ln))
    else:
        rows = (sec for sec, _text in corpus)
    for head in rows:
        dot = head.find("·")
        if dot < 0:
            continue
        yield head[:dot].strip(), head[dot + 1:].split("›")[-1].strip().lower()


def extract_defines(corpus, *, min_len=3):
    """Build-time `defines(passage_id, term)` from a source's OWN canonical markup: a parenthesized abbreviation in a
    section title names what that section defines — "Machine ISA (misa) Register" → misa. These are unique technical
    identifiers (misa, satp, mstatus, mcause, …), so as match entities they neither collide with each other nor with
    off-domain queries. The defining passage is the section's FIRST passage (its topic sentence); first occurrence wins.
    `corpus` is a corpus file path OR an iterable of (section, text) passages (the adapter Extraction).

    Deliberately NOT used: distinctive heading *content words* (hart, cause, …). They are ordinary English, so they
    match unrelated queries ("cause" → "what causes earthquakes", "mode" → any "...mode" query) — a precision/leak
    source. Conceptual terms with no parenthesized abbrev are left to ordinary retrieval. Returns [(passage_id, term)]."""
    out, seen_term, seen_title = [], set(), set()
    for cid, title in _sections(corpus):
        if title in seen_title:                                    # only the FIRST passage of each section (its intro)
            continue
        seen_title.add(title)
        for ab in _PAREN.findall(title):
            if len(ab) >= min_len and ab not in seen_term:
                seen_term.add(ab)
                out.append((cid, ab))
    return out


def strategy_tables(strategy_dl, out_tsv, *, mat=None, label="instruction", prefix="riscv:inventory",
                    defines=None, theorems=None):
    """Materialize the package's strategy.tsv — the UNIFORM (intent, entity, passage) table the thin runtime applies
    (no runtime engine; build-time only). One row shape for every strategy — count/list/define/theorem are just
    different intents, never special cases:
        cue    <word>   <intent>             the i18n-able intent lexicon (from ergo/strategy.dl — language as DATA)
        answer <intent> <entity> <passage>   a query of <intent> that NAMES <entity> is answered by <passage>
    count → entity = the inventory label; list → each group name; define → each defined term; theorem → each named
    statement. The runtime returns the answer whose entity the query names — so entity-must-appear IS the domain gate.
    A NEW intent (theorem) needs NO runtime change — only rows here + a cue word in ergo. Returns (out, n_cues, n_ans)."""
    souffle = shutil.which("souffle")
    if not souffle:
        raise RuntimeError("souffle not found — the strategy tier needs souffle at build time")
    with tempfile.TemporaryDirectory() as d:
        fdir, odir = os.path.join(d, "facts"), os.path.join(d, "out")
        os.makedirs(fdir)
        os.makedirs(odir)
        with open(os.path.join(fdir, "defines.facts"), "w", encoding="utf-8") as f:
            for section, term in (defines or []):                     # term -> defining passage (define rows via ergo)
                f.write(f"{section}\t{term}\n")
        subprocess.run([souffle, strategy_dl, "-F", fdir, "-D", odir], check=True)

        def rd(rel):
            p = os.path.join(odir, rel + ".csv")
            return [ln.rstrip("\n").split("\t") for ln in open(p, encoding="utf-8")] if os.path.exists(p) else []

        cues, ans = rd("cue"), rd("answer")                          # answer holds the ergo-derived define rows
    rows = [("answer",) + tuple(a) for a in ans]                     # define rows: (answer, "define", term, section)
    if mat:                                                          # inventory present → count/list rows
        rows.append(("answer", "count", label, f"{prefix}:total"))
        for g in sorted(mat.get("groups", {})):
            rows.append(("answer", "list", g.lower(), f"{prefix}:{g}"))
    for pid, name in (theorems or []):                              # theorem rows: a named statement → its passage
        rows.append(("answer", "theorem", name, pid))
    with open(out_tsv, "w", encoding="utf-8") as f:
        for w, i in cues:
            f.write(f"cue\t{w}\t{i}\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    return out_tsv, len(cues), len(rows)


def augment_corpus_with_inventory(corpus_txt, rules_dl, out_txt, *, label="instruction", prefix="riscv:inventory"):
    """Append cited inventory/count passages (computed by the ergo aggregates) to a COPY of the corpus at out_txt, so
    grounding embeds them alongside the rules. Returns (out_txt, n_passages, materialized)."""
    mat = materialize(instruction_inventory(corpus_txt), rules_dl)
    passages = inventory_passages(mat, label=label, prefix=prefix)
    shutil.copyfile(corpus_txt, out_txt)
    with open(out_txt, "a", encoding="utf-8") as f:
        for sec, text in passages:
            f.write(f"[{sec}] {text}\n")
    return out_txt, len(passages), mat
