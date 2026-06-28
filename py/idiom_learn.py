#!/usr/bin/env python3
"""rosetta · idiom_learn.py — UNSUPERVISED idiom learning from a model's behavior (ILP over dl/primitives.dl).

Not hand-coded detectors: the learner mines compact, GENERALIZING rules from behavior and certifies them. The engine here
is the frame-conditioned GATE learner (the 'select' family: output = table[token @ one slot], valid inside a discovered
frame). It is fully unsupervised — nothing about the grammar is given — via a two-phase pipeline:

  Phase 1 (harvest, observational, zero perturbations): anchor on each instance and greedily grow a frame using THAT
          instance's own values until token@k -> output is a clean ≥2-output function; keep the TABLE (the frame greedy
          picks may overfit to a co-occurring content word — discarded).
  Phase 2 (frame invention, observational): EXPAND each table to its full support (all instances consistent with it),
          then DERIVE the frame as the offsets CONSTANT across that support — the true structural frame; offsets that
          vary are the IGNORED slots. Faithfulness = no frame-matching instance violates the table.
  Phase 3 (causal confirm, the only decide() calls): perturb the gate slot within the frame; a REAL gate's output
          FOLLOWS the table (a correlational one doesn't). This is the discriminator between an idiom and a coincidence.

On threx this rediscovers, from behavior alone: the 'select' place gate {hï→fï, fa→bo, dø→sto} (gate slot 4, frame
{wø,·,⟨,⟩}, who@2 correctly ignored, causal 100%) and the conditional retrieved/bigram gates. The 'compose' idiom
(THINGS[i+j], ≥2 operands) is the arith template's job (py/discover.py) — the single-slot gate learner correctly leaves
it alone. select = one causal operand (lookup); compose = two operands (computation). Usage: idiom_learn.py [n] [w] [model_dir]
"""
import os, sys, json
from collections import defaultdict
from minimize import instances, model_refs, ref_source

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINCOV, MAXF = 4, 4


def tableof(insts, refs, sub, k):
    vm = defaultdict(set)
    for i in sub:
        vm[insts[i][-k]].add(refs[i])
    return vm


def anchored_table(insts, refs, idxs, anchor, k, w):
    """Phase 1: grow a frame from the ANCHOR's own values until token@k→output is a clean ≥2-output function; return the
    TABLE (the frame may overfit — re-derived in Phase 2). Anchoring makes even rare local idioms surface."""
    cur, frame = list(idxs), {}
    for _ in range(MAXF + 1):
        vm = tableof(insts, refs, cur, k)
        collided = [t for t, o in vm.items() if len(o) > 1]
        funct = {t: next(iter(o)) for t, o in vm.items() if len(o) == 1}
        if not collided and len(set(funct.values())) >= 2 and len(cur) >= MINCOV:
            return funct
        best = None
        for m in range(1, w + 1):
            if m == k or m in frame:
                continue
            v = insts[anchor][-m]
            sub = [i for i in cur if insts[i][-m] == v]
            if len(sub) < MINCOV:
                continue
            vmm = tableof(insts, refs, sub, k)
            fk = [t for t, o in vmm.items() if len(o) == 1]
            sc = (round(sum(len(vmm[t]) for t in fk) / len(sub), 4),
                  len({next(iter(vmm[t])) for t in fk}), len(sub))
            if best is None or sc > best[0]:
                best = (sc, m, v, sub)
        if not best:
            break
        _, m, v, sub = best
        frame[m] = v
        cur = sub
    return None


def build_gate(insts, refs, idxs, k, table, w):
    """Phase 2: expand the table to its full support, invent the frame (constant offsets), check faithfulness."""
    support = [i for i in idxs if insts[i][-k] in table and refs[i] == table[insts[i][-k]]]
    if len(support) < MINCOV or len(set(table.values())) < 2:
        return None
    frame = {m: next(iter({insts[i][-m] for i in support})) for m in range(1, w + 1)
             if m != k and len({insts[i][-m] for i in support}) == 1}
    fmatch = [i for i in idxs if all(insts[i][-m] == v for m, v in frame.items())]
    viol = [i for i in fmatch if insts[i][-k] in table and refs[i] != table[insts[i][-k]]]
    ignore = [m for m in range(1, w + 1) if m != k and m not in frame]
    return dict(k=k, table=dict(table), frame=frame, support=support, fmatch=fmatch, viol=viol, ignore=ignore)


def confirm(insts, b, decide_fn, ntest=18):
    """Phase 3: causal confirmation — perturb the gate slot within the frame; output must FOLLOW the table."""
    k, ok, tr = b["k"], 0, 0
    for i in b["fmatch"][:ntest]:
        for key in b["table"]:
            p = insts[i][:]; p[-k] = key
            tr += 1; ok += (decide_fn(p) == b["table"][key])
    return ok / tr if tr else 0.0


def learn_gates(insts, refs, idxs, w, decide_fn, n_anchor=240, max_confirm=40):
    anchors = idxs[::max(1, len(idxs) // n_anchor)]
    tables = defaultdict(set)
    for a in anchors:
        for k in range(1, w + 1):
            t = anchored_table(insts, refs, idxs, a, k, w)
            if t:
                tables[k].add(frozenset(t.items()))
    cands = {}
    for k, tabset in tables.items():
        for tab in tabset:
            b = build_gate(insts, refs, idxs, k, dict(tab), w)
            if not b:
                continue
            cands[(k, tuple(sorted(b["frame"].items())), tuple(sorted(b["table"].items())))] = b
    # rank observationally (faithful first, then coverage); causally confirm only the top candidates (bounds decide() calls)
    ranked = sorted(cands.values(), key=lambda b: (not b["viol"], len(b["support"])), reverse=True)
    for b in ranked[:max_confirm]:
        b["causal"] = confirm(insts, b, decide_fn)
    return [b for b in ranked if "causal" in b]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1400
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    md = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "reference", "threx")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    name = os.path.basename(md.rstrip("/"))
    sym = {i: t[0] for i, t in enumerate(json.load(open(os.path.join(md, "lexicon.json")))["tokens"])}
    s = lambda t: (sym.get(t, str(t)).strip() or f"[{t}]")
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    insts = instances(ids, n, w)
    refs = model_refs(md, insts)
    idxs = [i for i in range(len(insts)) if refs[i] is not None and len(insts[i]) >= w]
    decide_fn = ref_source(md)[1]
    print(f"=== idiom_learn · {name} · {len(idxs)} decisions (W={w}) — unsupervised, nothing hand-coded ===\n")

    gates = learn_gates(insts, refs, idxs, w, decide_fn)
    real = [b for b in gates if not b["viol"] and b["causal"] >= 0.8]
    print(f"frame-conditioned GATEs (select family) — {len(real)} REAL (faithful + causally confirmed) of {len(gates)} mined:\n")
    for b in real[:20]:
        fr = " ".join(f"@{m}={s(v)}" for m, v in sorted(b["frame"].items())) or "∅ (unconditional)"
        td = "{" + ", ".join(f"{s(t)}→{s(o)}" for t, o in sorted(b["table"].items())) + "}"
        print(f"  [causal {b['causal']:.0%}] GATE@{b['k']}  support={len(b['support'])}  ignore@{b['ignore']}")
        print(f"       frame[{fr}]")
        print(f"       {td}")
    print("\n(select = one causal operand → lookup table; the 'compose' idiom THINGS[i+j] is ≥2 operands → py/discover.py.)")


if __name__ == "__main__":
    main()
