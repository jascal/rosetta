#!/usr/bin/env python3
"""rosetta · idiom_learn.py — LEARN idioms from a model's behavior (ILP / anti-unification over dl/primitives.dl).

Not hand-coded detectors: search the primitive vocabulary for compact, GENERALIZING rules that are faithful to the model
(checked against its refs — the in-Datalog certificate is dl/equiv.dl). Each template learner anti-unifies a subset of
the residual into a variabilized rule and reports coverage + whether it generalizes (covers contexts that VARY in the
positions the rule ignores — the difference between an idiom and a memorized n-gram).

Templates so far (the rest of the primitive library follows the same shape):
  gate@k  : output = table[token at offset k]   — output decided by ONE earlier position (threx "selected" / place gate;
            n-grams MISS this when k isn't the contiguous suffix). The generalizer: a small table keyed by one slot.
  copy@k  : output = token at offset k           — verbatim copy from a fixed earlier slot.
  induct  : output = token after the most recent earlier occurrence of the last token (induction/copy, dl/induction.dl).

Validation gate: rediscover threx's idioms from behavior alone, nothing hand-coded. Usage: python3 py/idiom_learn.py [n] [w] [model_dir]
"""
import os, sys, json
from collections import defaultdict
from minimize import instances, model_refs, ref_source

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFF = lambda ctx, k: ctx[-k] if len(ctx) >= k else None     # token at offset k (offset 1 = last token)


def learn_gate(insts, refs, idxs, w):
    """For each offset k, the entries token@k→out where the output is a FUNCTION of token@k alone (consistent across all
    contexts with that token@k) AND it generalizes (≥2 distinct contexts vary in the other positions). Returns per-k:
    (table, n_covered, n_entries). A gate is an idiom iff some entry generalizes — it ignores everything but slot k."""
    out = {}
    for k in range(1, w + 1):
        groups = defaultdict(list)
        for i in idxs:
            t = OFF(insts[i], k)
            if t is not None:
                groups[t].append(i)
        table, covered, generalizing = {}, 0, 0
        for t, members in groups.items():
            if len({refs[i] for i in members}) == 1:                      # output is a function of token@k
                table[t] = refs[members[0]]
                covered += len(members)
                if len({tuple(insts[i]) for i in members}) >= 2:          # ≥2 distinct full contexts → generalizes
                    generalizing += 1
        if table:
            out[k] = (table, covered, generalizing)
    return out


def causal_confirm_gate(insts, k, table, members, decide_fn, ntest=6):
    """The discriminator: a gate@k is a REAL idiom (not a grammar correlation) iff the model's output CAUSALLY follows
    token@k. Take sample contexts, perturb token@k to other table values, and check the model's argmax becomes the
    table's entry for the new value. Returns fraction confirmed. (Correlational gates fail: perturbing the slot doesn't
    move the output to the table's prediction because the slot wasn't the cause.)"""
    vals = list(table)
    if len(vals) < 2:
        return 0.0
    ok = tries = 0
    for i in members[:ntest]:
        ctx = insts[i]
        if len(ctx) < k:
            continue
        for t2 in vals:
            if t2 == ctx[-k]:
                continue
            pert = ctx[:]; pert[-k] = t2
            tries += 1
            ok += (decide_fn(pert) == table[t2])     # output causally tracks the gated slot
    return ok / tries if tries else 0.0


def learn_copy(insts, refs, idxs, w):
    """offsets k where output == token@k (verbatim copy) for ALL covered contexts — and how many it covers."""
    out = {}
    for k in range(1, w + 1):
        hit = [i for i in idxs if OFF(insts[i], k) is not None and OFF(insts[i], k) == refs[i]]
        if hit:
            out[k] = len(hit)
    return out


def learn_induction(insts, refs, idxs):
    """output == token after the most recent earlier occurrence of the last token (induction). Coverage = where it holds."""
    hit = applies = 0
    for i in idxs:
        ctx = insts[i]
        last = ctx[-1]
        js = [j for j in range(len(ctx) - 1) if ctx[j] == last]
        if js:
            applies += 1
            j = max(js)
            if j + 1 < len(ctx):
                hit += (ctx[j + 1] == refs[i])
    return hit, applies


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    md = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "reference", "threx")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    name = os.path.basename(md.rstrip("/"))
    sym = {i: t[0] for i, t in enumerate(json.load(open(os.path.join(md, "lexicon.json")))["tokens"])}
    s = lambda t: (sym.get(t, str(t)).strip() or f"[{t}]")
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    insts = instances(ids, n, w)
    refs = model_refs(md, insts)
    idxs = [i for i in range(len(insts)) if refs[i] is not None]
    print(f"=== idiom_learn · {name} · {len(idxs)} decisions (W={w}) — learned from behavior, not hand-coded ===\n")

    decide_fn = ref_source(md)[1]
    by_tok = lambda k: {t: [i for i in idxs if OFF(insts[i], k) == t] for t in {OFF(insts[i], k) for i in idxs}}
    print("GATE@k  (output = table[token @ offset k]; anti-unified, then CAUSALLY CONFIRMED by perturbing the slot):")
    gates = learn_gate(insts, refs, idxs, w)
    confirmed = []
    for k in sorted(gates):
        table, cov, gen = gates[k]
        if gen < 2:
            continue
        members = [i for t in table for i in by_tok(k).get(t, [])]
        causal = causal_confirm_gate(insts, k, table, members, decide_fn)
        real = causal >= 0.8
        print(f"  k={k}: {len(table)} entries, covers {cov}, {gen} generalize, causal={causal:.0%}"
              f"{'  ← REAL idiom (output follows the slot)' if real else '  (correlational — pruned)'}")
        if real:
            confirmed.append((k, table, cov, causal))
    if confirmed:
        k, table, cov, causal = max(confirmed, key=lambda r: r[2])
        ex = sorted(table.items())[:8]
        print(f"  → learned + causally-confirmed GATE@{k}: {{{', '.join(f'{s(t)}→{s(o)}' for t, o in ex)}}}"
              f"{'' if len(table) <= 8 else ' …'}  — a {len(table)}-row table vs memorizing {cov} contexts (causal {causal:.0%})")

    print("\nCOPY@k  (output = token @ offset k, verbatim):")
    for k, c in sorted(learn_copy(insts, refs, idxs, w).items()):
        if c >= 2:
            print(f"  k={k}: covers {c}")
    hit, ap = learn_induction(insts, refs, idxs)
    print(f"\nINDUCTION (copy after previous occurrence of last token): {hit}/{ap} where it applies")
    print("\n(Next: ANTI-UNIFY/certify these into circuits.dl rules + the arith/state template for the composed i+j.)")


if __name__ == "__main__":
    main()
