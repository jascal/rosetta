#!/usr/bin/env python3
"""rosetta · temperature.py — THE canonical emitter: ONE rule set carrying logits as incidence values, T at query.

This is the canonical `circuits.dl` (no `.t` suffix — we always emit T-rules). Each rule carries the top-K
(token, LOGIT) — the incidence values, which are T-INVARIANT — and the runtime computes softmax(logits/T) IN SOUFFLE
at any query temperature (`.input temp`). It is a semiring lift: **T=0 = the argmax (tropical) collapse, recovered by
querying temp→0** (so the crisp argmax cover is just this artifact at T=0, not a separate file); T>0 = the probability
semiring with the incidence weights restored. The legible `circuits.symbols.dl` is the transliteration of THIS
distributional artifact (token strings + the same logits), when a lexicon.json is present.

Pipeline (pure souffle at runtime, fieldrun/whole.dl only at build time):
  1. logits   — the full scoreboard per context (oracle.logits, T-invariant), cached → regenerable cache-only (no oracle).
  2. cover    — DISTRIBUTIONAL minimal-suffix cover: shortest suffix under which the softmax (at T_max) is consistent
                within ε across the group (stronger than T=0's argmax-consistency → longer suffixes, the honest T cost).
  3. top-K    — per rule keep the top-K logits covering ≥1-ε mass at T_max (K small: threx ~3, max 9).
  4. emit     — circuits.dl: gramNd facts + softmax-at-T (E^((S-max)/T)) + cdist(inst,token,prob); + run.dl + symbols twin.
  5. certify  — run it in souffle at T_max, compare cdist to the model's full softmax; CERTIFIED iff max TV < ε over the corpus.
Usage: python3 py/temperature.py [n] [w] [model_dir] [T_max] [eps] [T_min] [--compose]
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


def dist_table(insts, logmap, members, keyfn, T_lo, T_hi, eps):
    """An idiom CARRIES A DISTRIBUTION iff, grouped by its key (compose sum / gate content value), the model's softmax is
    consistent within the group across the T-grid (the same ε/2 test dist_cover uses). Returns ({key: top-K logits},
    inconsistent-members) — a key whose group disagrees can't carry one faithful distribution, so its members fall back to
    the n-gram cover. This is what 'the learned idiom carries + generalizes the distribution' means, made certifiable."""
    grid = trange(T_lo, T_hi); half = eps / 2
    groups = defaultdict(list)
    for i in members:
        groups[keyfn(insts[i])].append(i)
    table, bad = {}, set()
    for key, mem in groups.items():
        rep = {T: softmax(logmap[mem[0]], T) for T in grid}
        if all(tv(rep[T], softmax(logmap[i], T)) < half for i in mem for T in grid):
            table[key] = topk(logmap[mem[0]], T_hi, half)
        else:
            bad.update(mem)
    return table, bad


def _pos(offsets):
    """offset k (1 = last token) → souffle position var relative to mp P (offset1=P, offset k = Pm{k-1}=P-{k-1}).
    Mirrors idiom_learn._positions so learned frames/operands emit at the same positions."""
    pm, eqns = {}, []
    for off in sorted(set(offsets)):
        pm[off] = "P" if off == 1 else f"Pm{off - 1}"
        if off != 1:
            eqns.append(f"Pm{off - 1}=P-{off - 1}")
    return pm, eqns


def _idiom_lines(idiom, q, ktype):
    """Datalog for ONE distributional idiom: its table facts + <name>_ctxlogit(I,token,logit) + <name>_any. q maps a token
    id to its literal (str for ids, a quoting fn for symbols); ktype is 'number' or 'symbol'. Gate is a lookup (symbolizable);
    compose is arithmetic over operand VALUES (ids only — the caller omits it from the symbol twin). Returns (lines, any_name)."""
    nm = idiom["name"]
    if idiom["kind"] == "gate":                                  # frame + content slot k → top-K logits per content value
        frame, k, tab = idiom["frame"], idiom["k"], idiom["tab"]
        pm, eqns = _pos(list(frame) + [k])
        atoms = ["mp(I,P)"] + [f"tok(I,{pm[o]},{q(t)})" for o, t in sorted(frame.items())]
        atoms += [f"tok(I,{pm[k]},K)", f"{nm}_tab(K,Tk,SC)"]
        lines = [f".decl {nm}_tab(k:{ktype},token:{ktype},sc:float)   // select-gate carrying a distribution"]
        lines += [f"{nm}_tab({q(key)},{q(t)},{s})." for key in sorted(tab) for t, s in tab[key]]
        lines += [f".decl {nm}_ctxlogit(inst:number,token:{ktype},s:float)",
                  f"{nm}_ctxlogit(I,Tk,SC) :- {', '.join(atoms + eqns)}.",
                  f".decl {nm}_any(inst:number)", f"{nm}_any(I) :- {nm}_ctxlogit(I,_,_)."]
        return lines, f"{nm}_any"
    # compose: operands @k1,@k2 → values → sum → top-K logits (generalizes the distribution to unseen operand pairs)
    frame, k1, k2, vm, csum = idiom["frame"], idiom["k1"], idiom["k2"], idiom["valmap"], idiom["csum"]
    pm, eqns = _pos(list(frame) + [k1, k2])
    atoms = ["mp(I,P)"] + [f"tok(I,{pm[o]},{t})" for o, t in sorted(frame.items())]
    atoms += [f"tok(I,{pm[k1]},A)", f"tok(I,{pm[k2]},B)", f"{nm}_val(A,VA)", f"{nm}_val(B,VB)", f"{nm}_sum(VA+VB,Tk,SC)"]
    lines = [f".decl {nm}_val(id:number,v:number)   // compose carrying a distribution"] + [f"{nm}_val({t},{v})." for t, v in sorted(vm.items())]
    lines += [f".decl {nm}_sum(s:number,token:number,sc:float)"]
    lines += [f"{nm}_sum({s},{t},{sc})." for s in sorted(csum) for t, sc in csum[s]]
    lines += [f".decl {nm}_ctxlogit(inst:number,token:number,s:float)",
              f"{nm}_ctxlogit(I,Tk,SC) :- {', '.join(atoms + eqns)}.",
              f".decl {nm}_any(inst:number)", f"{nm}_any(I) :- {nm}_ctxlogit(I,_,_)."]
    return lines, f"{nm}_any"


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


def emit_symbols_T(path, rules, sym, name, idioms=None, induction=False):
    """The legible, RUNNABLE twin of circuits.dl: the SAME distributional rules carrying top-K logits, with token STRINGS
    instead of ids — a self-contained, symbol-typed souffle program that computes softmax(logits/T) at a queried temp.
    A transliteration via the lexicon, so it inherits circuits.dl's T-certificate. Select-GATE idioms symbolize (they are
    token lookups); COMPOSE (arithmetic over operand values) and induction (a structural pointer) can't, so they stay in
    circuits.dl and their contexts are omitted here (noted)."""
    safe = lambda s: "".join(c if (0x20 <= ord(c) != 0x7f) else f"<0x{ord(c):02X}>" for c in s)  # control chars → visible,
    esc = lambda s: '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'                       # TSV-safe & roundtrippable
    q = lambda t: esc(safe(sym[t])) if sym.get(t) else esc(f"id{t}")
    idioms = idioms or []
    symbolizable = [i for i in idioms if i["kind"] == "gate"]
    omitted = [i["name"] for i in idioms if i["kind"] != "gate"] + (["induction"] if induction else [])
    L = [f"// {name} — circuits.symbols.dl: the legible, runnable twin of circuits.dl (token STRINGS, not ids).",
         "// Same DISTRIBUTIONAL rules carrying top-K logits (incidence); the runtime computes softmax(logits/T) at a",
         "// queried temp. A transliteration via the lexicon, so it inherits circuits.dl's T-certificate. Run on symbol input:",
         "//   souffle circuits.symbols.dl -F <dir: tok.facts (inst<TAB>pos<TAB>token) + temp.facts (T)> -D out  →  cdist.csv",
         "//   (control-char tokens — tab/newline — render as <0xNN> so the symbol stays TSV-safe; tokenize input the same way.)", "",
         ".decl tok(inst:number, pos:number, sym:symbol)", ".input tok",
         ".decl temp(t:float)", ".input temp",
         ".decl mp(inst:number, m:number)", "mp(I,M) :- M = max P : { tok(I,P,_) }.",
         ".decl ctxlogit(inst:number, token:symbol, s:float)"]
    if omitted:
        L.append(f"// NOTE: {', '.join(omitted)} compute over operand VALUES / are structural pointers, not token lookups —")
        L.append("// not representable in symbol form; see circuits.dl (their contexts are covered there, omitted here).")
    anys = []
    for idiom in symbolizable:                                    # select-gate idioms carry distributions AND symbolize
        lines, anm = _idiom_lines(idiom, q, "symbol")
        L += [""] + lines
        guard = "".join(f", !{a}(I)" for a in anys)
        L.append(f"ctxlogit(I,Tk,S) :- {idiom['name']}_ctxlogit(I,Tk,S){guard}.")
        anys.append(anm)
    idiom_guard = "".join(f", !{a}(I)" for a in anys)
    bylen = defaultdict(dict)
    for suf, kept in rules.items():
        bylen[len(suf)][suf] = kept
    lens = sorted(bylen)
    L.append("")
    for n in lens:
        N = n + 1
        cols = ",".join(f"c{i}:symbol" for i in range(n))
        L += [f".decl gram{N}d({cols},token:symbol,s:float)   // {N}-gram, distributional", f".decl gram{N}d_any(inst:number)"]
        for suf, kept in bylen[n].items():
            L += [f"gram{N}d({','.join(q(t) for t in suf)},{q(v)},{s})." for v, s in kept]
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
        L.append(f"ctxlogit(I,Tk,S) :- mp(I,P), {', '.join(toks + eqs + [pull])}{guard}{idiom_guard}.")
    L += ["", "// --- softmax at the query temperature (max-shift for stability, exactly as whole.dl) ---",
          ".decl lmax(inst:number,m:float)", "lmax(I,M) :- mp(I,_), M = max S : { ctxlogit(I,_,S) }.",
          ".decl wexp(inst:number,token:symbol,w:float)",
          f"wexp(I,Tk,W) :- ctxlogit(I,Tk,S), lmax(I,M), temp(T), W = {E}^((S-M)/T).",
          ".decl wz(inst:number,z:float)", "wz(I,Z) :- mp(I,_), Z = sum W : { wexp(I,_,W) }.",
          ".decl cdist(inst:number,token:symbol,prob:float)", ".output cdist", "cdist(I,Tk,W/Z) :- wexp(I,Tk,W), wz(I,Z)."]
    open(path, "w").write("\n".join(L) + "\n")


def emit_T(out_path, rules, w, idioms=None, induction=None, sym=None, name=""):
    """Canonical emit: circuits.dl (distributional, ids) + run.dl + — as the FINAL STEP — circuits.symbols.dl (legible twin)
    when a lexicon (sym) is given. Routing (priority via negation guards): the LEARNED idioms (compose/gate carrying top-K
    logits) in order > longest n-gram > induction (OOD point-mass) > abstain; each fires its full distribution into ctxlogit,
    then softmax(logits/T) runs uniformly. T=0 is the argmax collapse of this same artifact (query temp→0)."""
    idioms = idioms or []
    order = " > ".join([i["name"] for i in idioms] + ["longest-ngram"] + (["induction(OOD)"] if induction else []) + ["abstain"])
    L = ["// rosetta · circuits.dl — the model as next-token rules carrying top-K logits (incidence); softmax(logits/T) at",
         "// query temp. CANONICAL: we always emit T-rules; T=0 is the argmax (tropical) collapse, recovered by querying temp→0.",
         f"// Routing: {order}.  tok(inst,pos,id) + temp(t) provided by the includer (run.dl). cdist(inst,token,prob) = the dist at T.",
         "", ".decl mp(inst:number,m:number)", "mp(I,M) :- M = max P : { tok(I,P,_) }.",
         ".decl ctxlogit(inst:number,token:number,s:float)"]
    anys = []
    for idiom in idioms:                                          # LEARNED idioms carrying distributions, in priority order
        lines, anm = _idiom_lines(idiom, str, "number")
        L += [""] + lines
        guard = "".join(f", !{a}(I)" for a in anys)              # guarded by all higher-priority idioms
        L.append(f"ctxlogit(I,Tk,S) :- {idiom['name']}_ctxlogit(I,Tk,S){guard}.")
        anys.append(anm)
    idiom_guard = "".join(f", !{a}(I)" for a in anys)
    bylen = defaultdict(dict)
    for suf, kept in rules.items():
        bylen[len(suf)][suf] = kept
    lens = sorted(bylen)
    L.append("")
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
    for n in lens:                                                     # longest matching suffix supplies the logits (below idioms)
        N = n + 1
        toks = [f"tok(I,{'P' if i == n - 1 else f'Pm{n-1-i}'},C{i})" for i in range(n)]
        eqs = [f"Pm{j}=P-{j}" for j in range(1, n)]
        pull = f"gram{N}d({','.join(f'C{i}' for i in range(n))},Tk,S)"
        guard = "".join(f", !gram{m+1}d_any(I)" for m in lens if m > n)
        L.append(f"ctxlogit(I,Tk,S) :- mp(I,P), {', '.join(toks + eqs + [pull])}{guard}{idiom_guard}.")
    if induction:                                                     # copy/induction OOD fallback — structural pointer
        gram_guard = "".join(f", !gram{m+1}d_any(I)" for m in lens)
        L += ["", "// copy/induction OOD fallback: a structural pointer, NOT a calibrated distribution → POINT-MASS on the",
              "// copied token. Fires only where no idiom/n-gram matches, so it never affects the in-domain certificate.",
              ".decl ind_pj(inst:number,j:number)", "ind_pj(I,J) :- mp(I,P), tok(I,P,X), tok(I,J,X), J<P.",
              ".decl ind_last(inst:number,j:number)", "ind_last(I,J) :- ind_pj(I,_), J = max JJ : { ind_pj(I,JJ) }.",
              ".decl ind_ctxlogit(inst:number,token:number,s:float)",
              f"ind_ctxlogit(I,OUT,0.0) :- ind_last(I,J), tok(I,J+1,OUT){idiom_guard}{gram_guard}.",
              ".decl ind_any(inst:number)", "ind_any(I) :- ind_ctxlogit(I,_,_).",
              "ctxlogit(I,Tk,S) :- ind_ctxlogit(I,Tk,S)."]
    L += ["", "// --- softmax at the query temperature (max-shift for stability, exactly as whole.dl) ---",
          ".decl lmax(inst:number,m:float)", "lmax(I,M) :- mp(I,_), M = max S : { ctxlogit(I,_,S) }.",
          ".decl wexp(inst:number,token:number,w:float)",
          f"wexp(I,Tk,W) :- ctxlogit(I,Tk,S), lmax(I,M), temp(T), W = {E}^((S-M)/T).",
          ".decl wz(inst:number,z:float)", "wz(I,Z) :- mp(I,_), Z = sum W : { wexp(I,_,W) }.",
          ".decl cdist(inst:number,token:number,prob:float)", "cdist(I,Tk,W/Z) :- wexp(I,Tk,W), wz(I,Z)."]
    open(out_path, "w").write("\n".join(L) + "\n")
    run = os.path.join(os.path.dirname(out_path), "run.dl")
    open(run, "w").write("// standalone runtime harness for circuits.dl — souffle only, no fieldrun/whole.dl/weights.\n"
                         "// souffle run.dl -F <dir: tok.facts (inst<TAB>pos<TAB>id) + temp.facts (T)> -D <out>  →  cdist.csv\n"
                         ".decl tok(inst:number,pos:number,id:number)\n.input tok\n.decl temp(t:float)\n.input temp\n"
                         f'#include "{os.path.basename(out_path)}"\n.output cdist\n')
    if sym:                                                       # FINAL STEP of extraction: the legible token-string twin
        emit_symbols_T(os.path.join(os.path.dirname(out_path), "circuits.symbols.dl"), rules, sym, name, idioms=idioms, induction=bool(induction))


def certify_T(out_path, insts, logmap, idxs, T, eps):
    """run circuits.dl in souffle at T, compare cdist to the model's full softmax; CERTIFIED iff max TV < eps."""
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


def finalize(md, insts, logmap, idxs, idioms, rules, remaining, induction, w, sym, name, T, eps, T_lo, src):
    """Emit the canonical circuits.dl (+ run.dl + symbols twin) from idioms + n-gram cover (+ induction OOD), certify across
    the T-grid, write CERTIFICATE.md. Shared by temperature.main (n-gram + threx-compose) and idiom_learn (full learned
    idioms carrying distributions). Returns the across-range verdict."""
    out = os.path.join(md, "circuits.dl")
    emit_T(out, rules, w, idioms=idioms, induction=induction, sym=sym, name=name)
    Ks = [len(v) for v in rules.values()] or [0]
    nc = sum(1 for i in idioms if i["kind"] == "compose"); ng = sum(1 for i in idioms if i["kind"] == "gate")
    idiom_desc = ", ".join(f"{x} {k}" for x, k in [(nc, "compose"), (ng, "select-gate")] if x) or "no idioms"
    print(f"canonical emit: {idiom_desc} carrying distributions + {len(rules)} n-gram (top-K mean {sum(Ks)/len(Ks):.1f}, max {max(Ks)})"
          + (" + induction OOD" if induction else "") + (f"; {len(remaining)} uncovered" if remaining else "")
          + f" → {out}" + (" + circuits.symbols.dl (legible twin)" if sym else ""))
    grid = sorted({T_lo, round((T_lo + T) / 2, 3), T})
    ok_all, results = True, []
    for q in grid:                                                    # ONE rule set, certified across the whole range
        worst, ngot = certify_T(out, insts, logmap, idxs, q, eps)
        ok = worst < eps and ngot == len(idxs)
        ok_all &= ok
        results.append((q, ngot, worst, ok))
        print(f"  CERTIFY @T={q}: {ngot}/{len(idxs)} contexts, max TV={worst:.4f} {'✓' if ok else '✗'}")
    print("→ " + (f"CERTIFIED across T∈[{T_lo},{T}] — learned idioms + n-gram, softmax(logits/T) in souffle" if ok_all else "NOT certified over the range"))
    nrules = len(rules) + len(idioms)
    lines = [f"# {name} · certificate (T-parameterized — the canonical artifact)", "",
             "`circuits.dl` carries top-K logits (incidence) per rule; the runtime computes `softmax(logits/T)` in souffle",
             f"at a queried `.input temp` (T=0 = the argmax collapse). Build-time logits from {src}.",
             (f"`circuits.symbols.dl` is the legible token-string twin (inherits this certificate)." if sym else None), "",
             f"- domain: {len(idxs)} decision windows (W={w})",
             f"- range: T ∈ [{T_lo}, {T}], ε = {eps}",
             f"- rules: {nrules} ({idiom_desc} + {len(rules)} n-gram" + (", induction OOD" if induction else "") + f", top-K mean {sum(Ks)/len(Ks):.1f})",
             "", "| T | contexts | max TV | verdict |", "|---|---|---|---|"]
    lines += [f"| {q} | {ngot}/{len(idxs)} | {worst:.4f} | {'CERTIFIED' if ok else 'NOT certified'} |" for q, ngot, worst, ok in results]
    lines += ["", f"**{'CERTIFIED across the range' if ok_all else 'NOT certified over the full range'}** — souffle cdist vs the model's own softmax(logits/T). Runtime: `souffle run.dl`."]
    open(os.path.join(md, "CERTIFICATE.md"), "w").write("\n".join([x for x in lines if x is not None]) + "\n")
    return ok_all


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
    lexp = os.path.join(md, "lexicon.json")                    # optional — if present, emit the legible symbols twin
    sym = {i: t[0] for i, t in enumerate(json.load(open(lexp))["tokens"])} if os.path.exists(lexp) else {}
    serve = os.environ.get("FIELDRUN_SERVE")                   # logits from a resident server (big models) or whole.dl (pure)
    if serve:
        get_lg, src = (lambda ctx: serve_topk(int(serve), ctx)), "a fieldrun --serve /topk server"
    elif os.path.exists(whole):
        get_lg, src = (lambda ctx: model_logits(whole, ctx)), "whole.dl"
    else:
        get_lg, src = (lambda ctx: None), "the cached logits (cache-only — no oracle)"   # regenerate from logit_cache.json
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
    idioms = [{**compose, "kind": "compose", "name": "comp0"}] if compose else []
    cover_idxs = [i for i in idxs if not (compose and i in compose["covered"])]
    rules, remaining = dist_cover(insts, logmap, cover_idxs, T_lo, T, eps, w)
    finalize(md, insts, logmap, idxs, idioms, rules, remaining, False, w, sym, name, T, eps, T_lo, src)


if __name__ == "__main__":
    main()
