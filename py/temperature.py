#!/usr/bin/env python3
"""rosetta · temperature.py — THE canonical emitter: ONE rule set carrying logits as incidence values, T at query.

This is the canonical `circuits.dl` (no `.t` suffix — we always emit T-rules). Each rule carries the top-K
(token, LOGIT) — the incidence values, which are T-INVARIANT — and the runtime computes softmax(logits/T) IN SOUFFLE
at a positive query temperature (`.input temp`). T=0 is not supported by this runtime; use the crisp emitter for
argmax. The optional `circuits.symbols.dl` is an uncertified rendering that may omit arithmetic/structural idioms.

Pipeline (pure souffle at runtime, fieldrun/whole.dl only at build time):
  1. logits   — the full scoreboard per context (oracle.logits, T-invariant), cached → regenerable cache-only (no oracle).
  2. cover    — DISTRIBUTIONAL minimal-suffix cover: shortest suffix under which the softmax (at T_max) is consistent
                within ε across the group (stronger than T=0's argmax-consistency → longer suffixes, the honest T cost).
  3. top-K    — per rule keep the top-K logits covering ≥1-ε mass at T_max (K small: threx ~3, max 9).
  4. emit     — circuits.dl: gramNd facts + softmax-at-T (E^((S-max)/T)) + cdist(inst,token,prob); + run.dl + symbols twin.
  5. certify  — equiv_dist.dl computes TV and the verdict at the finite temperature grid against supplied logits.
                Complete domain and replayable evidence are retained. No interval or unbounded-tail claim.
Usage: python3 py/temperature.py [n] [w] [model_dir] [T_max] [eps] [T_min] [--compose]
"""
import os, sys, json, math
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimize import instances
from oracle import logits as model_logits, serve_topk

E = "2.718281828459045"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_sym(md):
    """id→glyph for the legible symbols twin: from lexicon.json if present, else decode via the model's bundle tokenizer
    (so lexicon-less real models — llama/qwen/pythia — still get a twin; byte-fallback / invalid-UTF8 tokens fall back to
    their raw token form, then control chars render <0xNN> in the emitter). Returns {} if neither is available → no twin."""
    lex = os.path.join(md, "lexicon.json")
    if os.path.exists(lex):
        return {i: t[0] for i, t in enumerate(json.load(open(lex))["tokens"])}
    tokp = os.path.join(md, "bundle.tokenizer.json")
    if not os.path.exists(tokp):
        return {}
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tokp)

    class _TokSym(dict):                                          # decode lazily + cache; the emitter only hits ids in rules
        def __bool__(self):                                       # truthy even when unpopulated (a tokenizer IS available) →
            return True                                           # so emit_T's `if sym:` fires the symbols twin

        def get(self, i, default=None):
            if i not in self:
                v = tk.decode([i])
                if not v or "�" in v:
                    v = tk.id_to_token(i) or ""
                dict.__setitem__(self, i, v)
            return dict.get(self, i) or default

        def __getitem__(self, i):
            return self.get(i) or f"id{i}"

    return _TokSym()


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
    return sorted({T_lo, (T_lo + T_hi) / 2, T_hi})


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
    """shortest suffix s.t. the group's softmax is consistent within eps at every tested grid temperature (the two error
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
            if all(tv(rep[T], dists[i][T]) < half for i in members for T in grid):   # finite-grid candidate search
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
    This rendering has no inherited certificate. Select-GATE idioms symbolize (they are
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
         "// queried positive temp. Uncertified rendering; no inherited certificate. Run on symbol input:",
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
    then softmax(logits/T) runs uniformly for T > 0; T=0 requires the crisp emitter."""
    idioms = idioms or []
    order = " > ".join([i["name"] for i in idioms] + ["longest-ngram"] + (["induction(OOD)"] if induction else []) + ["abstain"])
    L = ["// rosetta · circuits.dl — the model as next-token rules carrying top-K logits (incidence); softmax(logits/T) at",
         "// query temp > 0. T=0 is unsupported here; use the crisp emitter for argmax.",
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
                         f'#include "{os.path.basename(out_path)}"\n.output cdist\n'
                         '.decl abstain(inst:number)\n.output abstain\nabstain(I) :- tok(I,_,_), !cdist(I,_,_).\n')
    if sym:                                                       # FINAL STEP of extraction: the legible token-string twin
        emit_symbols_T(os.path.join(os.path.dirname(out_path), "circuits.symbols.dl"), rules, sym, name, idioms=idioms, induction=bool(induction))


REFERENCE_SCOPE = (
    "softmax of the supplied reference logits only; omitted model probability mass is not bounded"
)


def certify_T(out_path, insts, logmap, idxs, T, eps, *, evidence_dir=None, provenance=None):
    """Read equiv_dist.dl's verdict at ONE temperature over the entire requested domain.

    Missing logits remain missing obligations. Python serializes logits; Datalog computes both softmax and TV.
    """
    from certificate import check
    if len(set(idxs)) != len(idxs) or any(i < 0 or i >= len(insts) for i in idxs):
        raise ValueError("domain indices must be unique and refer to supplied instances")
    if not math.isfinite(T) or not math.isfinite(eps):
        raise ValueError("temperature and epsilon must be finite")
    for i in idxs:
        if any(not math.isfinite(s) for _, s in logmap.get(i, [])):
            raise ValueError(f"non-finite reference logit for instance {i}")
    facts = {
        "domain": "".join(f"{i}\n" for i in idxs),
        "tok": "".join(f"{i}\t{p}\t{t}\n" for i in idxs for p, t in enumerate(insts[i])),
        "temp": f"{T}\n",
        "epsilon": f"{eps}\n",
        "ref_logit": "".join(f"{i}\t{t}\t{s}\n" for i in idxs for t, s in logmap.get(i, [])),
    }
    scalars = {name: int for name in ("ndomain", "ncover", "nmiss", "nuncov", "nmissing", "ninvalid")}
    scalars["worst"] = float
    return check(os.path.join(HERE, "dl", "equiv_dist.dl"), out_path, facts, scalars,
                 evidence_dir=evidence_dir, provenance=provenance)


def finalize(md, insts, logmap, idxs, idioms, rules, remaining, induction, w, sym, name, T, eps, T_lo, src):
    """Emit and retain a finite-temperature-grid Datalog certificate against supplied logits.

    Each evidence directory is independently replayable. No interval, full-model-tail, symbol-twin or OOD claim.
    """
    from certificate import sha256
    if not (math.isfinite(T_lo) and math.isfinite(T) and 0 < T_lo <= T):
        raise ValueError("temperature grid requires 0 < T_min <= T_max")
    if not math.isfinite(eps) or not 0 < eps <= 1:
        raise ValueError("epsilon must be finite and in (0, 1]")
    out = os.path.join(md, "circuits.dl")
    emit_T(out, rules, w, idioms=idioms, induction=induction, sym=sym, name=name)
    grid = trange(T_lo, T)
    provenance = {"source": src, "reference_scope": REFERENCE_SCOPE,
                  "source_files_sha256": {f: sha256(os.path.join(md, f))
                      for f in ("whole.dl", "corpus.json", "logit_cache.json") if os.path.isfile(os.path.join(md, f))}}
    results = []
    for q in grid:
        result = certify_T(out, insts, logmap, idxs, q, eps,
                           evidence_dir=os.path.join(md, "certificate-evidence"), provenance=provenance)
        result.pop("relations", None)
        result["temperature"] = q
        result["evidence"] = os.path.relpath(result["evidence"], md)
        results.append(result)
        print(f"  Datalog check @T={q}: {result.get('ncover', '?')}/{len(idxs)} contexts, "
              f"max TV={result.get('worst', '?')} → {'CERTIFIED' if result['certified'] else 'NOT certified'}")
        if "error" in result:
            print(result["error"])
    ok_all = all(r["certified"] for r in results)
    report = {
        "schema_version": 1, "tag": "proved" if ok_all else "open",
        "scope": REFERENCE_SCOPE, "numeric_semantics": "Souffle floating-point arithmetic",
        "temperatures": grid, "epsilon": eps, "domain": idxs, "window": w,
        "artifact_sha256": sha256(out), "provenance": provenance, "checks": results,
        "interval_certified": False, "symbol_twin_certified": False,
        "residual_policy": "no matching rule: no output (abstain); no model fallback",
    }
    with open(os.path.join(md, "certificate.json"), "w") as f:
        json.dump(report, f, indent=2); f.write("\n")
    lines = [f"# {name} · finite-grid distributional certificate", "",
             f"**{'proved' if ok_all else 'open'}** — {'CERTIFIED' if ok_all else 'NOT certified'} against {REFERENCE_SCOPE}.",
             "The verifier is `dl/equiv_dist.dl`; arithmetic follows Souffle floating-point semantics.", "",
             f"- domain: {len(idxs)} requested decision windows (W={w}); exact inputs retained in evidence",
             f"- temperatures checked: {grid}; epsilon = {eps}",
             f"- rules: {len(rules)} n-gram + {len(idioms)} idioms",
             f"- artifact SHA-256: `{report['artifact_sha256']}`", "",
             "| T | contexts producing output | max TV | verdict | evidence |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['temperature']} | {r.get('ncover', '?')}/{len(idxs)} | {r.get('worst', '?')} | "
                     f"{'CERTIFIED' if r['certified'] else 'NOT certified'} | [{r['evidence']}]({r['evidence']}/certificate.json) |")
    lines += ["", "No claim is made between the checked temperatures, outside the stated domain, or about omitted model mass.",
              "The symbol rendering is an uncertified view; it does not inherit this certificate.",
              "Runtime: `souffle run.dl` with positive `temp` and token facts only. No matching rule means abstention.",
              "Replay a check: `python3 py/certificate.py <model-dir>/<evidence-directory>`.",
              "`certificate.json` records the grid and artifact identity; each evidence directory retains the exact",
              "candidate, verifier, facts, output relations, reference provenance, and SHA-256 hashes."]
    with open(os.path.join(md, "CERTIFICATE.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
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
    sym = build_sym(md)                                        # legible symbols twin: lexicon.json, else the bundle tokenizer
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
    ok = finalize(md, insts, logmap, list(range(len(insts))), idioms, rules, remaining, False, w, sym, name, T, eps, T_lo, src)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
