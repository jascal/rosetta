#!/usr/bin/env python3
"""rosetta · temperature.py — the T>0 path: ONE rule set carrying logits as incidence values, T parameterized at query.

T=0 circuits.dl maps each context → its argmax (no weights). This emits the distributional version: each rule carries the
top-K (token, LOGIT) — the incidence values, which are T-INVARIANT — and the runtime computes softmax(logits/T) IN SOUFFLE
at any query temperature (`.input temp`). It is a semiring lift: T=0 = argmax collapse; T>0 = the probability semiring with
the incidence weights restored.

Pipeline (pure souffle at runtime, fieldrun/whole.dl only at build time):
  1. logits   — the full scoreboard per context (oracle.logits, T-invariant), cached.
  2. cover    — DISTRIBUTIONAL minimal-suffix cover: shortest suffix under which the softmax (at T_max) is consistent
                within ε across the group (stronger than T=0's argmax-consistency → longer suffixes, the honest T cost).
  3. top-K    — per rule keep the top-K logits covering ≥1-ε mass at T_max (K small: threx ~3, max 9).
  4. emit     — circuits.t.dl: gramN_dist facts + softmax-at-T (E^((S-max)/T)) + cdist(inst,token,prob).
  5. certify  — run it in souffle at T_max, compare cdist to the model's full softmax; CERTIFIED iff max TV < ε over the corpus.
Usage: python3 py/temperature.py [n] [w] [model_dir] [T_max] [eps]
"""
import os, sys, json, math, subprocess, tempfile
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimize import instances
from oracle import logits as model_logits, _run

E = "2.718281828459045"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def softmax(ls, T):
    m = max(s for _, s in ls)
    ex = [(v, math.exp((s - m) / T)) for v, s in ls]
    z = sum(e for _, e in ex)
    return {v: e / z for v, e in ex}


def tv(d1, d2):
    return 0.5 * sum(abs(d1.get(t, 0.0) - d2.get(t, 0.0)) for t in set(d1) | set(d2))


def topk(ls, T, eps):
    """smallest set of (token,logit) whose softmax(/T) mass ≥ 1-eps."""
    order = sorted(ls, key=lambda x: -x[1])
    d = softmax(ls, T)
    kept, mass = [], 0.0
    for v, s in order:
        kept.append((v, s)); mass += d[v]
        if mass >= 1 - eps:
            break
    return kept


def dist_cover(insts, logmap, idxs, T_lo, T_hi, eps, w):
    """shortest suffix s.t. the group's softmax is consistent within eps at the COLDEST T_lo (where grouping is hardest —
    low T amplifies within-group logit gaps); store a representative's top-K sized at the HOTTEST T_hi (where truncation
    is hardest — the tail is fattest). Two opposing error sources, so the cover must straddle the whole [T_lo, T_hi] range."""
    rules, order_of, remaining = {}, {}, set(idxs)
    dists = {i: softmax(logmap[i], T_lo) for i in idxs}
    for k in range(1, w + 1):
        groups = defaultdict(list)
        for i in remaining:
            groups[tuple(insts[i][-k:])].append(i)
        for suf, members in groups.items():
            rep = dists[members[0]]
            if all(tv(rep, dists[i]) < eps for i in members):          # consistent at the cold end
                rules[suf] = topk(logmap[members[0]], T_hi, eps)        # K sized at the hot end
                for i in members:
                    order_of[i] = k
                remaining -= set(members)
        if not remaining:
            break
    return rules, remaining


def emit_T(out_path, rules, w):
    L = ["// rosetta · circuits.t.dl — T-PARAMETERIZED: rules carry top-K logits (incidence); softmax(logits/T) at query.",
         "// tok(inst,pos,id) + temp(t) provided by the includer (run.t.dl). cdist(inst,token,prob) is the distribution at T.",
         "", ".decl mp(inst:number,m:number)", "mp(I,M) :- M = max P : { tok(I,P,_) }.",
         ".decl ctxlogit(inst:number,token:number,s:float)"]
    bylen = defaultdict(dict)
    for suf, kept in rules.items():
        bylen[len(suf)][suf] = kept
    lens = sorted(bylen)
    for n in lens:
        N = n + 1
        cols = ",".join(f"c{i}:number" for i in range(n))
        L += [f".decl gram{N}d({cols},token:number,s:float)", f".decl gram{N}d_any(inst:number)"]
        for suf, kept in bylen[n].items():
            L += [f"gram{N}d({','.join(map(str, suf))},{v},{s})." for v, s in kept]
        toks = [f"tok(I,{'P' if i == n - 1 else f'Pm{n-1-i}'},C{i})" for i in range(n)]
        eqs = [f"Pm{j}=P-{j}" for j in range(1, n)]
        key = f"gram{N}d({','.join(f'C{i}' for i in range(n))},_,_)"
        L.append(f"gram{N}d_any(I) :- mp(I,P), {', '.join(toks + eqs + [key])}.")
    for n in lens:                                                     # longest matching suffix supplies the logits
        N = n + 1
        toks = [f"tok(I,{'P' if i == n - 1 else f'Pm{n-1-i}'},C{i})" for i in range(n)]
        eqs = [f"Pm{j}=P-{j}" for j in range(1, n)]
        pull = f"gram{N}d({','.join(f'C{i}' for i in range(n))},Tk,S)"
        guard = "".join(f", !gram{m+1}d_any(I)" for m in lens if m > n)
        L.append(f"ctxlogit(I,Tk,S) :- mp(I,P), {', '.join(toks + eqs + [pull])}{guard}.")
    L += ["", "// --- softmax at the query temperature (max-shift for stability, exactly as whole.dl) ---",
          ".decl lmax(inst:number,m:float)", "lmax(I,M) :- mp(I,_), M = max S : { ctxlogit(I,_,S) }.",
          ".decl wexp(inst:number,token:number,w:float)",
          f"wexp(I,Tk,W) :- ctxlogit(I,Tk,S), lmax(I,M), temp(T), W = {E}^((S-M)/T).",
          ".decl wz(inst:number,z:float)", "wz(I,Z) :- mp(I,_), Z = sum W : { wexp(I,_,W) }.",
          ".decl cdist(inst:number,token:number,prob:float)", "cdist(I,Tk,W/Z) :- wexp(I,Tk,W), wz(I,Z)."]
    open(out_path, "w").write("\n".join(L) + "\n")
    run = os.path.join(os.path.dirname(out_path), "run.t.dl")
    open(run, "w").write(".decl tok(inst:number,pos:number,id:number)\n.input tok\n.decl temp(t:float)\n.input temp\n"
                         f'#include "{os.path.basename(out_path)}"\n.output cdist\n')


def certify_T(out_path, insts, logmap, idxs, T, eps):
    """run circuits.t.dl in souffle at T, compare cdist to the model's full softmax; CERTIFIED iff max TV < eps."""
    with tempfile.TemporaryDirectory() as d:
        ind, outd, inc = os.path.join(d, "in"), os.path.join(d, "out"), os.path.join(d, "inc")
        os.makedirs(ind); os.makedirs(outd); os.makedirs(inc)
        import shutil
        shutil.copyfile(out_path, os.path.join(inc, os.path.basename(out_path)))
        with open(os.path.join(ind, "tok.facts"), "w") as tf:
            for i in idxs:
                for p, t in enumerate(insts[i]):
                    tf.write(f"{i}\t{p}\t{t}\n")
        open(os.path.join(ind, "temp.facts"), "w").write(f"{T}\n")
        harness = os.path.join(d, "h.dl")
        open(harness, "w").write(".decl tok(inst:number,pos:number,id:number)\n.input tok\n.decl temp(t:float)\n.input temp\n"
                                 f'#include "{os.path.basename(out_path)}"\n.output cdist\n')
        _run(harness, ind, outd, includes=[inc])
        got = defaultdict(dict)
        cp = os.path.join(outd, "cdist.csv")
        if os.path.exists(cp):
            for ln in open(cp).read().splitlines():
                i, t, p = ln.split("\t")
                got[int(i)][int(t)] = float(p)
    worst = 0.0
    for i in idxs:
        worst = max(worst, tv(got.get(i, {}), softmax(logmap[i], T)))
    return worst, len(got)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    md = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "reference", "threx")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    T = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0       # T_max (hot end — sizes top-K)
    eps = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
    T_lo = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5    # T_min (cold end — sizes the grouping)
    name = os.path.basename(md.rstrip("/"))
    whole = os.path.join(md, "whole.dl")
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    insts = instances(ids, n, w)
    cache_p = os.path.join(md, "logit_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    key = lambda c: ",".join(map(str, c))
    print(f"=== temperature · {name} · {len(insts)} windows · T∈[{T_lo},{T}] ε={eps} ===")
    for j, ctx in enumerate(insts):
        if key(ctx) not in cache:
            lg = model_logits(whole, ctx)
            if lg:
                cache[key(ctx)] = lg
            if j % 50 == 0:
                json.dump(cache, open(cache_p, "w")); print(f"   …{j}/{len(insts)} logits")
    json.dump(cache, open(cache_p, "w"))
    logmap = {i: [(int(v), float(s)) for v, s in cache[key(insts[i])]] for i in range(len(insts)) if key(insts[i]) in cache}
    idxs = sorted(logmap)
    rules, remaining = dist_cover(insts, logmap, idxs, T_lo, T, eps, w)
    out = os.path.join(md, "circuits.t.dl")
    emit_T(out, rules, w)
    Ks = [len(v) for v in rules.values()]
    print(f"distributional cover: {len(rules)} rules for {len(idxs)} windows (top-K mean {sum(Ks)/len(Ks):.1f}, max {max(Ks)})"
          + (f"; {len(remaining)} uncovered" if remaining else "") + f" → {out}")
    grid = sorted({T_lo, round((T_lo + T) / 2, 3), T})
    ok_all = True
    for q in grid:                                                    # ONE rule set, certified across the whole range
        worst, ngot = certify_T(out, insts, logmap, idxs, q, eps)
        ok = worst < eps and ngot == len(idxs)
        ok_all &= ok
        print(f"  CERTIFY @T={q}: {ngot}/{len(idxs)} contexts, max TV={worst:.4f} {'✓' if ok else '✗'}")
    print("→ " + (f"CERTIFIED across T∈[{T_lo},{T}] — one rule set, softmax(logits/T) in souffle" if ok_all else "NOT certified over the range"))


if __name__ == "__main__":
    main()
