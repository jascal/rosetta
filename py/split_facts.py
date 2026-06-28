#!/usr/bin/env python3
"""rosetta · split_facts.py — split an inline-fact whole.dl into rules + .facts data modules ("whole.dl in parts").

emit_whole writes weights as inline Datalog facts (`embed_w(0,0,-0.29).`), which souffle re-parses every interpreter
call and inlines into C++ when compiling (a 261k-fact model → a 106MB .cpp that g++ chokes on). This rewrites it as:
  forward.dl       — the rules + .decl + an `.input` for every weight relation   (tiny: ~the 116 rules)
  weights/<rel>.facts — each weight matrix as TSV data, loaded at runtime         (the 99.96% that was facts)
souffle then bulk-loads the weights as data (fast) instead of parsing them, and `souffle -c` compiles only the rules
(tiny binary that loads the .facts at runtime). Same program, same answers — just facts-as-data, not facts-as-code.
Returns (forward_dl_path, weights_dir). Idempotent/cached on mtime. Usage: python3 py/split_facts.py <whole.dl>
"""
import os, re, sys

FACT = re.compile(r"^([a-z_][A-Za-z0-9_]*)\((.*)\)\.\s*$")


def split(whole_dl):
    whole_dl = os.path.abspath(whole_dl)
    base = whole_dl[:-3] if whole_dl.endswith(".dl") else whole_dl
    forward = base + ".forward.dl"
    wdir = base + ".weights"
    if os.path.exists(forward) and os.path.exists(wdir) and os.path.getmtime(forward) >= os.path.getmtime(whole_dl):
        return forward, wdir
    os.makedirs(wdir, exist_ok=True)
    rules, fact_rels, handles = [], set(), {}
    for line in open(whole_dl):
        s = line.rstrip("\n")
        m = FACT.match(s)
        if m and ":-" not in s:                          # an inline weight fact
            rel, args = m.group(1), m.group(2)
            fact_rels.add(rel)
            h = handles.get(rel) or handles.setdefault(rel, open(os.path.join(wdir, rel + ".facts"), "w"))
            h.write("\t".join(a.strip() for a in args.split(",")) + "\n")
        else:
            rules.append(s)
    for h in handles.values():
        h.close()
    # forward.dl = rules, with an `.input <rel>` added for every weight relation (so souffle loads the .facts)
    with open(forward, "w") as f:
        f.write("\n".join(rules) + "\n")
        f.write("\n// weights loaded as data modules (split_facts.py)\n")
        for rel in sorted(fact_rels):
            f.write(f".input {rel}\n")
    return forward, wdir


if __name__ == "__main__":
    fwd, wd = split(sys.argv[1])
    print(f"forward: {fwd} ({sum(1 for _ in open(fwd))} lines)")
    print(f"weights: {wd} ({len(os.listdir(wd))} .facts modules)")
