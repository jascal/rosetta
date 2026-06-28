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
from oracle import logits as model_logits, serve_topk, _run

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


def trange(T_lo, T_hi):
    return sorted({T_lo, round((T_lo + T_hi) / 2, 3), T_hi})


def dist_cover(insts, logmap, idxs, T_lo, T_hi, eps, w):
    """shortest suffix s.t. the group's softmax is consistent within eps at EVERY T across the range (the two error
    sources sit at opposite ends — group-divergence is worst cold for structured models, hot for diverse ones — so we
    must check the whole [T_lo, T_hi] grid, not one endpoint), with the representative's top-K sized at the hot end (where
    the tail is fattest). Always terminates: at full-W each context is its own group (the per-context memorization corner)."""
    grid = trange(T_lo, T_hi)
    half = eps / 2                                              # split the ε budget: group-consistency AND top-K truncation
    rules, order_of, remaining = {}, {}, set(idxs)             # each < ε/2, so their compounded error stays < ε at certify
    dists = {i: {T: softmax(logmap[i], T) for T in grid} for i in idxs}
    for k in range(1, w + 1):
        groups = defaultdict(list)
        for i in remaining:
            groups[tuple(insts[i][-k:])].append(i)
        for suf, members in groups.items():
            rep = dists[members[0]]
            if all(tv(rep[T], dists[i][T]) < half for i in members for T in grid):   # consistent across the whole range
                rules[suf] = topk(logmap[members[0]], T_hi, half)
                for i in members:
                    order_of[i] = k
                remaining -= set(members)
        if not remaining:
            break
    return rules, remaining


def build_threx_compose(insts, logmap, idxs, T_hi, eps):
    """The threx COMPOSE idiom (discovered by idiom_learn) carrying a DISTRIBUTION: operands @4,@5 are bearings, strength
    = id-21, sum → top-K logits. Validated: a composed context's full distribution depends only on the sum (within-sum
    TV≈0), so one sum→logits table generalizes the distribution to operand pairs never seen — not just the argmax."""
    BRG = set(range(21, 26))
    frame = {1: 7, 2: 19, 3: 19, 6: 20, 7: 0, 8: 1}            # gɪ · · ∿ ⟨ ⟩ — the learned compose frame
    composed = [i for i in idxs if len(insts[i]) >= 8 and insts[i][-4] in BRG and insts[i][-5] in BRG
                and all(insts[i][-o] == t for o, t in frame.items())]
    if not composed:
        return None
    bysum = defaultdict(list)
    for i in composed:
        bysum[(insts[i][-4] - 21) + (insts[i][-5] - 21)].append(i)
    csum = {s: topk(logmap[mem[0]], T_hi, eps / 2) for s, mem in bysum.items()}
    return dict(frame=frame, k1=4, k2=5, valmap={b: b - 21 for b in BRG}, csum=csum, covered=set(composed))


def emit_T(out_path, rules, w, compose=None):
    L = ["// rosetta · circuits.t.dl — T-PARAMETERIZED: rules carry top-K logits (incidence); softmax(logits/T) at query.",
         "// tok(inst,pos,id) + temp(t) provided by the includer (run.t.dl). cdist(inst,token,prob) is the distribution at T.",
         "", ".decl mp(inst:number,m:number)", "mp(I,M) :- M = max P : { tok(I,P,_) }.",
         ".decl ctxlogit(inst:number,token:number,s:float)"]
    comp_guard = ""
    if compose:                                                  # the COMPOSE idiom carrying a DISTRIBUTION (sum → top-K logits)
        k1, k2, vm, csum = compose["k1"], compose["k2"], compose["valmap"], compose["csum"]
        pm, eqns = {}, []                                        # position vars for the frame + operand offsets
        for off in sorted(set(list(compose["frame"]) + [k1, k2])):
            pm[off] = "P" if off == 1 else f"Pm{off - 1}"
            if off != 1:
                eqns.append(f"Pm{off - 1}=P-{off - 1}")
        atoms = ["mp(I,P)"] + [f"tok(I,{pm[o]},{t})" for o, t in sorted(compose["frame"].items())]
        atoms += [f"tok(I,{pm[k1]},A)", f"tok(I,{pm[k2]},B)", "cval(A,VA)", "cval(B,VB)", "csum_logit(VA+VB,Tk,SC)"]
        L += [".decl cval(id:number,v:number)"] + [f"cval({t},{v})." for t, v in sorted(vm.items())]
        L += [".decl csum_logit(s:number,token:number,sc:float)"]
        L += [f"csum_logit({s},{t},{sc})." for s in sorted(csum) for t, sc in csum[s]]
        L += [".decl comp_ctxlogit(inst:number,token:number,s:float)",
              f"comp_ctxlogit(I,Tk,SC) :- {', '.join(atoms + eqns)}.",
              ".decl comp_any(inst:number)", "comp_any(I) :- comp_ctxlogit(I,_,_).",
              "ctxlogit(I,Tk,S) :- comp_ctxlogit(I,Tk,S)."]    # compose wins (highest priority)
        comp_guard = ", !comp_any(I)"
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
        L.append(f"ctxlogit(I,Tk,S) :- mp(I,P), {', '.join(toks + eqs + [pull])}{guard}{comp_guard}.")
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
    serve = os.environ.get("FIELDRUN_SERVE")                   # logits from a resident server (big models) or whole.dl (pure)
    get_lg = (lambda ctx: serve_topk(int(serve), ctx)) if serve else (lambda ctx: model_logits(whole, ctx))
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    insts = instances(ids, n, w)
    cache_p = os.path.join(md, "logit_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    key = lambda c: ",".join(map(str, c))
    print(f"=== temperature · {name} · {len(insts)} windows · T∈[{T_lo},{T}] ε={eps} ===")
    for j, ctx in enumerate(insts):
        if key(ctx) not in cache:
            lg = get_lg(ctx)
            if lg:
                cache[key(ctx)] = lg
            if j % 50 == 0:
                json.dump(cache, open(cache_p, "w")); print(f"   …{j}/{len(insts)} logits")
    json.dump(cache, open(cache_p, "w"))
    logmap = {i: [(int(v), float(s)) for v, s in cache[key(insts[i])]] for i in range(len(insts)) if key(insts[i]) in cache}
    idxs = sorted(logmap)
    compose = build_threx_compose(insts, logmap, idxs, T, eps) if "--compose" in sys.argv else None
    cover_idxs = [i for i in idxs if not (compose and i in compose["covered"])]
    rules, remaining = dist_cover(insts, logmap, cover_idxs, T_lo, T, eps, w)
    out = os.path.join(md, "circuits.t.dl")
    emit_T(out, rules, w, compose=compose)
    Ks = [len(v) for v in rules.values()]
    if compose:
        cks = [len(v) for v in compose["csum"].values()]
        print(f"COMPOSE idiom carrying a distribution: 1 sum→logits table ({len(compose['csum'])} sums, top-K mean "
              f"{sum(cks)/len(cks):.1f}) covers {len(compose['covered'])} composed windows — generalizes the distribution.")
    print(f"distributional n-gram cover: {len(rules)} rules for {len(cover_idxs)} windows (top-K mean {sum(Ks)/len(Ks):.1f}, max {max(Ks)})"
          + (f"; {len(remaining)} uncovered" if remaining else "") + f" → {out}")
    grid = sorted({T_lo, round((T_lo + T) / 2, 3), T})
    ok_all, results = True, []
    for q in grid:                                                    # ONE rule set, certified across the whole range
        worst, ngot = certify_T(out, insts, logmap, idxs, q, eps)
        ok = worst < eps and ngot == len(idxs)
        ok_all &= ok
        results.append((q, ngot, worst, ok))
        print(f"  CERTIFY @T={q}: {ngot}/{len(idxs)} contexts, max TV={worst:.4f} {'✓' if ok else '✗'}")
    print("→ " + (f"CERTIFIED across T∈[{T_lo},{T}] — one rule set, softmax(logits/T) in souffle" if ok_all else "NOT certified over the range"))
    nrules = len(rules) + (1 if compose else 0)
    lines = [f"# {name} · T-parameterized certificate", "",
             f"`circuits.t.dl` carries top-K logits (incidence) per rule; the runtime computes `softmax(logits/T)` in",
             f"souffle at a queried `.input temp`. Build-time logits from {'a fieldrun --serve /topk server' if serve else 'whole.dl'}.", "",
             f"- domain: {len(idxs)} decision windows (W={w})",
             f"- range: T ∈ [{T_lo}, {T}], ε = {eps}",
             f"- rules: {nrules}" + (f" ({1} compose idiom + {len(rules)} n-gram, top-K mean {sum(Ks)/len(Ks):.1f})" if compose else f" (n-gram, top-K mean {sum(Ks)/len(Ks):.1f})"),
             "", "| T | contexts | max TV | verdict |", "|---|---|---|---|"]
    lines += [f"| {q} | {ngot}/{len(idxs)} | {worst:.4f} | {'CERTIFIED' if ok else 'NOT certified'} |" for q, ngot, worst, ok in results]
    lines += ["", f"**{'CERTIFIED across the range' if ok_all else 'NOT certified over the full range'}** — souffle cdist vs the model's own softmax(logits/T). Runtime: `souffle run.t.dl`."]
    open(os.path.join(md, "CERTIFICATE.t.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
