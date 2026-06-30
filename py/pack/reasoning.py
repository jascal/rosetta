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


def inventory_passages(mat, *, label="instruction", prefix="riscv:inventory"):
    """Materialized counts/list → cited KB passages, section = "id · Facet" (so cite-as-handle + /lookup work). One
    'total' passage + one per group (carrying the enumerated names — the count ships with what it counted)."""
    out = [(f"{prefix}:total · {label.capitalize()} inventory",
            f"There are {mat['total']} distinct RISC-V {label}s across the indexed extensions "
            f"(closed-world: counted from the spec rules at build time).")]
    by_group = {}
    for name, g in mat["items"]:
        by_group.setdefault(g, set()).add(name)
    for g, n in sorted(mat["groups"].items()):
        names = ", ".join(sorted(by_group.get(g, ())))
        out.append((f"{prefix}:{g} · {g}", f"The {g} defines {n} {label}s: {names}."))
    return out


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
