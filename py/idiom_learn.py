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
import os, sys, json, itertools
from collections import defaultdict, Counter
from minimize import instances, model_refs, ref_source, minimal_suffix_cover

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
    if max(Counter(table.values()).values()) / len(table) > 0.7:   # near-constant table → the slot doesn't SELECT
        return None                                                # (kills n-gram boilerplate like "<s> …→upon", not a gate)
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


def learn_gates(insts, refs, idxs, w, decide_fn, n_anchor=240, max_confirm=40, ntest=14, fill=None):
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
    top = ranked[:max_confirm]
    if fill:                                                  # batch the perturbation oracle calls so they run in parallel
        fill([(*insts[i][:-b["k"]], key, *insts[i][len(insts[i]) - b["k"] + 1:])
              for b in top for i in b["fmatch"][:ntest] for key in b["table"]])
    for b in top:
        b["causal"] = confirm(insts, b, decide_fn, ntest)
    return top


# ---------------------------------------------------------------------------------------------------------------------
# COMPOSE family (the 'compute' idioms): output = h(value(slot k1) + value(slot k2)) — TWO operands combined by a binary
# op, then indexed. Generalizes py/discover.py to discovered frames. select = one operand (lookup); compose = two
# operands (computation) — distinct families, told apart by how many slots are causally load-bearing.

def pair_table(insts, refs, sub, k1, k2):
    vm = defaultdict(set)
    for i in sub:
        vm[(insts[i][-k1], insts[i][-k2])].add(refs[i])
    return vm


def single_functional(insts, refs, sub, k):
    vm = defaultdict(set)
    for i in sub:
        vm[insts[i][-k]].add(refs[i])
    return all(len(o) == 1 for o in vm.values()) and len({next(iter(o)) for o in vm.values()}) >= 2


def additive(T):
    """discover.py's search: a labeling of the operand alphabet under which the output depends only on the SUM."""
    alpha = sorted(set(a for a, _ in T) | set(b for _, b in T))
    if not 3 <= len(alpha) <= 7:
        return None
    for perm in itertools.permutations(range(len(alpha))):
        lab, bysum, ok = dict(zip(alpha, perm)), {}, True
        for (a, b), o in T.items():
            if bysum.setdefault(lab[a] + lab[b], o) != o:
                ok = False; break
        if ok and len(set(bysum.values())) >= 3:
            return lab, bysum
    return None


def confirm_compose(insts, k1, k2, region, lab, bysum, decide_fn):
    """Causal confirmation over the FULL operand grid (not just corpus pairs): output must follow bysum[lab[a]+lab[b]].
    Agreement across the whole grid proves the model computes the sum (same-sum pairs → same output), not a pair table.
    Also reports extrapolation: agreement on grid pairs the corpus never showed."""
    alpha, base = sorted(lab), region[0]
    seen = {(insts[i][-k1], insts[i][-k2]) for i in region}
    ok = tr = ext = etr = 0
    for a in alpha:
        for b in alpha:
            pred = bysum.get(lab[a] + lab[b])
            if pred is None:
                continue
            p = insts[base][:]; p[-k1] = a; p[-k2] = b
            hit = decide_fn(p) == pred
            tr += 1; ok += hit
            if (a, b) not in seen:
                etr += 1; ext += hit
    return (ok / tr if tr else 0.0), (ext / etr if etr else None)


def learn_compose(insts, refs, idxs, w, decide_fn, guard_cap=80, max_confirm=30, fill=None):
    """Frame-first mining: iterate single-guard frame regions, find an operand PAIR with a clean additive 2D table where
    neither slot alone determines the output, then causally confirm."""
    offval = defaultdict(Counter)
    for i in idxs:
        for m in range(1, w + 1):
            offval[m][insts[i][-m]] += 1
    seen, cands = set(), []
    for m in range(1, w + 1):
        for v, c in offval[m].most_common(guard_cap):
            if c < MINCOV:
                continue
            region = [i for i in idxs if insts[i][-m] == v]
            for k1, k2 in itertools.combinations([x for x in range(1, w + 1) if x != m], 2):
                vm = pair_table(insts, refs, region, k1, k2)
                if any(len(o) > 1 for o in vm.values()):
                    continue
                a1 = len({insts[i][-k1] for i in region}); a2 = len({insts[i][-k2] for i in region})
                if len({next(iter(o)) for o in vm.values()}) < 3 or a1 < 3 or a2 < 3:
                    continue
                if single_functional(insts, refs, region, k1) or single_functional(insts, refs, region, k2):
                    continue
                add = additive({p: next(iter(o)) for p, o in vm.items()})
                if not add:
                    continue
                frame = {mm: next(iter({insts[i][-mm] for i in region})) for mm in range(1, w + 1)
                         if mm not in (k1, k2) and len({insts[i][-mm] for i in region}) == 1}
                key = (k1, k2, tuple(sorted(frame.items())))
                if key in seen:
                    continue
                seen.add(key)
                cands.append(dict(k1=k1, k2=k2, frame=frame, region=region, lab=add[0], bysum=add[1]))
    cands.sort(key=lambda b: len(b["region"]), reverse=True)
    top = cands[:max_confirm]
    if fill:                                                  # batch the operand-grid oracle calls for parallelism
        grid = []
        for b in top:
            base = insts[b["region"][0]]
            for a in b["lab"]:
                for bb in b["lab"]:
                    p = base[:]; p[-b["k1"]] = a; p[-b["k2"]] = bb
                    grid.append(p)
        fill(grid)
    for b in top:
        b["causal"], b["extrap"] = confirm_compose(insts, b["k1"], b["k2"], b["region"], b["lab"], b["bysum"], decide_fn)
    return top


# ---------------------------------------------------------------------------------------------------------------------
# COPY / INDUCTION family (content-relative, not offset-relative): output = ctx[ prev_occ(last-L suffix) + L ] — the
# token following the most-recent earlier occurrence of the current suffix (the prev_occ primitive). CRITICAL: the
# OBSERVATIONAL rate is confounded by n-gram determinism (a recurring suffix yields the same next token for n-gram
# reasons, not copying), so the CAUSAL test — perturb the pointed-to token, output must follow — is the real signal.
# On pythia-160m: greedy natural text observational 59-91% but causal 11% (n-gram); repeated NOVEL tokens causal 85-90%
# (true induction). select/compose are offset-relative; this family is content-relative — the other axis of idiom.

def learn_relational(insts, refs, idxs, decide_fn, fill=None, maxL=3, ntest=80):
    out = []
    lo = min(min(c) for c in insts if c); hi = max(max(c) for c in insts if c)
    for L in range(1, maxL + 1):
        app, hit = [], 0
        for i in idxs:
            ctx = insts[i]
            if len(ctx) <= L:
                continue
            suf = tuple(ctx[-L:])
            js = [j for j in range(len(ctx) - L) if tuple(ctx[j:j + L]) == suf]
            if js:
                ptr = max(js) + L
                if ptr < len(ctx):
                    app.append((i, ptr)); hit += (ctx[ptr] == refs[i])
        if not app:
            continue
        import random as _r
        rng = _r.Random(1)
        tests = []
        for i, ptr in app[:ntest]:
            ctx = insts[i]; xp = rng.randint(lo, hi)
            while xp == ctx[ptr]:
                xp = rng.randint(lo, hi)
            p = ctx[:]; p[ptr] = xp; tests.append((p, xp))
        if fill:
            fill([p for p, _ in tests])
        causal = sum(1 for p, xp in tests if decide_fn(p) == xp) / len(tests) if tests else 0.0
        out.append(dict(L=L, applicable=len(app), obs=hit / len(app), causal=causal))
    return out


def idiom_coverage(insts, refs, idxs, gates, comps, rels):
    """Which instances the CONFIRMED idioms predict correctly (the generalizing rules). The residual is everything left —
    what the n-gram backfill must memoize. Idioms first, n-grams last: an n-gram is the cache of a rule, so we only
    backfill what no learned idiom captured."""
    covered, by = set(), {}
    for b in gates:
        c = {i for i in idxs if all(len(insts[i]) >= m and insts[i][-m] == v for m, v in b["frame"].items())
             and len(insts[i]) >= b["k"] and insts[i][-b["k"]] in b["table"]
             and refs[i] == b["table"][insts[i][-b["k"]]]}
        by[f"select gate@{b['k']}"] = len(c); covered |= c
    for b in comps:
        c = set()
        for i in idxs:
            ctx = insts[i]
            if all(len(ctx) >= m and ctx[-m] == v for m, v in b["frame"].items()) and len(ctx) >= max(b["k1"], b["k2"]):
                a, bb = ctx[-b["k1"]], ctx[-b["k2"]]
                if a in b["lab"] and bb in b["lab"] and refs[i] == b["bysum"].get(b["lab"][a] + b["lab"][bb]):
                    c.add(i)
        by[f"compose@{b['k1']}+{b['k2']}"] = len(c); covered |= c
    for r in rels:
        L, c = r["L"], set()
        for i in idxs:
            ctx = insts[i]
            if len(ctx) > L:
                suf = tuple(ctx[-L:]); js = [j for j in range(len(ctx) - L) if tuple(ctx[j:j + L]) == suf]
                if js and max(js) + L < len(ctx) and ctx[max(js) + L] == refs[i]:
                    c.add(i)
        by[f"induction L={L}"] = len(c); covered |= c
    return covered, by


def _positions(offsets):
    """offset k (1=last token) → souffle position var relative to mp P: offset1=P, offset k=Pm{k-1} with Pm{k-1}=P-{k-1}.
    Returns ({offset: var}, [eqns])."""
    pm, eqns = {}, []
    for off in sorted(set(offsets)):
        if off == 1:
            pm[off] = "P"
        else:
            pm[off] = f"Pm{off - 1}"; eqns.append(f"Pm{off - 1}=P-{off - 1}")
    return pm, eqns


def emit_circuits(out_path, gates, comps, rels, ngram_rules, w, name=""):
    """Emit a runtime-independent circuits.dl (souffle only) from LEARNED idioms + n-gram backfill. Cover-ordering:
    faithful idioms (compose, then select gate) WIN; else the longest n-gram suffix; else copy/induction as the OOD
    fallback (fires only where no n-gram matches — generalizes off-corpus without breaking the in-domain certificate)."""
    L = ["// rosetta · circuits.dl — LEARNED idioms (unsupervised, causally confirmed) + n-gram backfill. Numbers = token ids.",
         f"// model: {name}.  Routing: compose > select-gate > longest n-gram > copy/induction (OOD fallback) > abstain.",
         "// Runtime: souffle only (see run.dl). tok(inst,pos,id) is provided by the includer (run.dl / equiv.dl).", "",
         ".decl mp(inst:number,m:number)", "mp(I,M) :- M = max P : { tok(I,P,_) }.",
         ".decl cdecide(inst:number,out:number)", ""]
    fired, anys = [], []                                       # (predicate, _any) in priority order — for negation guards

    for j, b in enumerate(comps):                              # ---- compose (highest priority) ----
        nm = f"comp{j}"
        pm, eqns = _positions(list(b["frame"]) + [b["k1"], b["k2"]])
        atoms = ["mp(I,P)"] + [f"tok(I,{pm[m]},{v})" for m, v in sorted(b["frame"].items())]
        atoms += [f"tok(I,{pm[b['k1']]},A)", f"tok(I,{pm[b['k2']]},B)",
                  f"{nm}_val(A,VA)", f"{nm}_val(B,VB)", f"{nm}_sum(VA+VB,OUT)"]
        L += [f".decl {nm}_val(id:number,v:number)"] + [f"{nm}_val({t},{v})." for t, v in sorted(b["lab"].items())]
        L += [f".decl {nm}_sum(s:number,o:number)"] + [f"{nm}_sum({s},{o})." for s, o in sorted(b["bysum"].items())]
        L += [f".decl {nm}(inst:number,out:number)", f"{nm}(I,OUT) :- {', '.join(atoms + eqns)}.",
              f".decl {nm}_any(inst:number)", f"{nm}_any(I) :- {nm}(I,_).", ""]
        fired.append(nm); anys.append(f"{nm}_any")

    for j, b in enumerate(gates):                              # ---- select gates ----
        nm = f"gate{j}"
        pm, eqns = _positions(list(b["frame"]) + [b["k"]])
        atoms = ["mp(I,P)"] + [f"tok(I,{pm[m]},{v})" for m, v in sorted(b["frame"].items())]
        atoms += [f"tok(I,{pm[b['k']]},K)", f"{nm}_tab(K,OUT)"]
        L += [f".decl {nm}_tab(k:number,out:number)"] + [f"{nm}_tab({k},{o})." for k, o in sorted(b["table"].items())]
        L += [f".decl {nm}(inst:number,out:number)", f"{nm}(I,OUT) :- {', '.join(atoms + eqns)}.",
              f".decl {nm}_any(inst:number)", f"{nm}_any(I) :- {nm}(I,_).", ""]
        fired.append(nm); anys.append(f"{nm}_any")

    bylen = defaultdict(dict)                                  # ---- n-gram cover ----
    for suf, o in ngram_rules.items():
        bylen[len(suf)][suf] = o
    lens = sorted(bylen)
    for n in lens:
        N = n + 1
        pm, _ = _positions(range(1, n + 1))
        cols = ",".join(f"c{i}:number" for i in range(n))
        L += [f".decl gram{N}({cols},t:number)", f".decl gram{N}_hit(inst:number,t:number)", f".decl gram{N}_any(inst:number)"]
        L += [f"gram{N}({','.join(map(str, suf))},{o})." for suf, o in bylen[n].items()]
        atoms = ["mp(I,P)"] + [f"tok(I,{pm[n - i]},C{i})" for i in range(n)] + [f"gram{N}({','.join(f'C{i}' for i in range(n))},T)"]
        eqns = [f"Pm{k}=P-{k}" for k in range(1, n)]
        L += [f"gram{N}_hit(I,T) :- {', '.join(atoms + eqns)}.", f"gram{N}_any(I) :- gram{N}_hit(I,_).", ""]

    ind = []                                                   # ---- copy/induction OOD fallback (L=1) ----
    for r in rels:
        if r["L"] != 1:
            continue
        L += [".decl ind1_pj(inst:number,j:number)", "ind1_pj(I,J) :- mp(I,P), tok(I,P,X), tok(I,J,X), J<P.",
              ".decl ind1_last(inst:number,j:number)", "ind1_last(I,J) :- ind1_pj(I,_), J = max JJ : { ind1_pj(I,JJ) }.",
              ".decl ind1(inst:number,out:number)", "ind1(I,OUT) :- ind1_last(I,J), tok(I,J+1,OUT).",
              ".decl ind1_any(inst:number)", "ind1_any(I) :- ind1(I,_).", ""]
        ind.append("ind1")
        break

    L.append("// --- routing (priority via negation guards) ---")
    for i, nm in enumerate(fired):                             # idioms: each guarded by all higher-priority idioms
        guard = "".join(f", !{anys[k]}(I)" for k in range(i))
        L.append(f"cdecide(I,T) :- {nm}(I,T){guard}.")
    no_idiom = "".join(f", !{a}(I)" for a in anys)
    for n in lens:                                             # n-grams: longest wins, below all idioms
        g = no_idiom + "".join(f", !gram{m + 1}_any(I)" for m in lens if m > n)
        L.append(f"cdecide(I,T) :- gram{n + 1}_hit(I,T){g}.")
    no_gram = "".join(f", !gram{m + 1}_any(I)" for m in lens)
    for nm in ind:                                             # induction: OOD fallback, below idioms AND n-grams
        L.append(f"cdecide(I,T) :- {nm}(I,T){no_idiom}{no_gram}.")

    open(out_path, "w").write("\n".join(L) + "\n")
    run = os.path.join(os.path.dirname(out_path), "run.dl")
    open(run, "w").write(
        "// standalone runtime harness — souffle only, no fieldrun/weights.\n"
        "// souffle run.dl -F <dir with tok.facts: inst<TAB>pos<TAB>id> -D <out>  →  cdecide.csv\n"
        ".decl tok(inst:number, pos:number, id:number)\n.input tok\n"
        f'#include "{os.path.basename(out_path)}"\n.output cdecide\n')


def main():
    backfill = "--backfill" in sys.argv                       # report idiom coverage + n-gram backfill stats
    emit = "--emit" in sys.argv                               # write circuits.dl + run.dl (idioms + n-gram cover, souffle-only)
    cert = "--certify" in sys.argv                            # prove the emitted circuits.dl == model via equiv.dl (cached refs)
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    n = int(a[0]) if len(a) > 0 else 1400
    w = int(a[1]) if len(a) > 1 else 8
    md = a[2] if len(a) > 2 else os.path.join(HERE, "reference", "threx")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    name = os.path.basename(md.rstrip("/"))
    lexp = os.path.join(md, "lexicon.json")                    # optional — large LMs (Llama) decode to token ids
    sym = {i: t[0] for i, t in enumerate(json.load(open(lexp))["tokens"])} if os.path.exists(lexp) else {}
    s = lambda t: (sym.get(t, str(t)).strip() or f"[{t}]")
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    insts = instances(ids, n, w)
    refs = model_refs(md, insts)
    idxs = [i for i in range(len(insts)) if refs[i] is not None and len(insts[i]) >= w]
    _raw, _cache = ref_source(md)[1], {}                      # memoize the oracle — fieldrun calls are ~0.1-0.4s each
    def decide_fn(ctx):
        key = tuple(ctx)
        if key not in _cache:
            _cache[key] = _raw(ctx)
        return _cache[key]
    def fill(ctxs, workers=8):                                # batch-compute perturbations in parallel (fieldrun subprocess releases the GIL)
        miss = list({tuple(c) for c in ctxs} - set(_cache))
        if not miss:
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for c, o in zip(miss, ex.map(lambda t: _raw(list(t)), miss)):
                _cache[c] = o
    print(f"=== idiom_learn · {name} · {len(idxs)} decisions (W={w}) — unsupervised, nothing hand-coded ===\n")

    gates = learn_gates(insts, refs, idxs, w, decide_fn, fill=fill)
    real = [b for b in gates if not b["viol"] and b["causal"] >= 0.8]
    print(f"frame-conditioned GATEs (select family) — {len(real)} REAL (faithful + causally confirmed) of {len(gates)} mined:\n")
    for b in real[:20]:
        fr = " ".join(f"@{m}={s(v)}" for m, v in sorted(b["frame"].items())) or "∅ (unconditional)"
        td = "{" + ", ".join(f"{s(t)}→{s(o)}" for t, o in sorted(b["table"].items())) + "}"
        print(f"  [causal {b['causal']:.0%}] GATE@{b['k']}  support={len(b['support'])}  ignore@{b['ignore']}")
        print(f"       frame[{fr}]")
        print(f"       {td}")
    comps = learn_compose(insts, refs, idxs, w, decide_fn, fill=fill)
    real_c = [b for b in comps if b["causal"] >= 0.8]
    print(f"\n2-operand COMPOSE idioms (compute family) — {len(real_c)} REAL (causally confirmed) of {len(comps)} mined:\n")
    for b in real_c[:10]:
        lab, bysum = b["lab"], b["bysum"]
        fr = " ".join(f"@{m}={s(v)}" for m, v in sorted(b["frame"].items())) or "∅"
        ex = "" if b["extrap"] is None else f"  extrapolate {b['extrap']:.0%}"
        vals = ", ".join(f"{s(t)}:{lab[t]}" for t in sorted(lab))
        sums = ", ".join(f"{n}:{s(o)}" for n, o in sorted(bysum.items()))
        print(f"  [causal {b['causal']:.0%}{ex}] COMPOSE @{b['k1']}+@{b['k2']}  support={len(b['region'])}")
        print(f"       frame[{fr}]   values {{{vals}}}")
        print(f"       sum→out {{{sums}}}")

    rels = learn_relational(insts, refs, idxs, decide_fn, fill=fill)
    print("\nCOPY/INDUCTION family (content-relative; observational is n-gram-CONFOUNDED — causal is the real signal):")
    for r in rels:
        if r["causal"] >= 0.8 and r["obs"] >= 0.5:
            tag = "  ← REAL copy/induction (causally confirmed)"
        elif r["obs"] >= 0.5:
            tag = "  (n-gram determinism, NOT copy — causal too low)"
        else:
            tag = ""
        print(f"  induction L={r['L']}: applicable {r['applicable']}, observational {r['obs']:.0%}, causal {r['causal']:.0%}{tag}")

    if backfill:
        rels_real = [r for r in rels if r["causal"] >= 0.8 and r["obs"] >= 0.5]
        covered, by = idiom_coverage(insts, refs, idxs, real, real_c, rels_real)
        residual = [i for i in idxs if i not in covered]
        rules, remaining, _ = minimal_suffix_cover(insts, refs, residual, w)
        print(f"\n=== n-gram BACKFILL (idioms first, memoize the residual last) ===")
        print(f"  idioms (generalizing rules) cover {len(covered)}/{len(idxs)} = {len(covered)/len(idxs):.0%}"
              + (f"  [{', '.join(f'{k}:{v}' for k, v in by.items() if v)}]" if any(by.values()) else "  [none — pure n-gram model]"))
        print(f"  n-gram suffix cover memoizes the residual: {len(rules)} rules for {len(residual)} instances"
              + (f", {len(remaining)} UNCOVERED (raise w)" if remaining else " (complete)"))
        tot = len(real) + len(real_c) + len(rels_real) + len(rules)
        print(f"  full circuit = {len(real)+len(real_c)+len(rels_real)} idioms + {len(rules)} n-gram rules = {tot} rules for {len(idxs)} decisions")

    if emit or cert:
        rels_real = [r for r in rels if r["causal"] >= 0.8 and r["obs"] >= 0.5]   # induction → OOD fallback (below n-grams)
        covered, _ = idiom_coverage(insts, refs, idxs, real, real_c, [])          # faithful idioms only (gate/compose)
        residual = [i for i in idxs if i not in covered]
        rules, remaining, _ = minimal_suffix_cover(insts, refs, residual, w)
        out = os.path.join(md, "circuits.dl")
        emit_circuits(out, real, real_c, rels_real, rules, w, name)
        print(f"\n=== EMIT → {out} ===")
        print(f"  {len(real_c)} compose + {len(real)} select-gate idioms (faithful, in-domain) cover {len(covered)}/{len(idxs)};"
              f" {len(rules)} n-gram rules memoize the residual" + (f"; {len(remaining)} uncovered" if remaining else "")
              + (f"; {len(rels_real)} induction OOD fallback" if rels_real else "") + ".  + run.dl (souffle-only harness).")
        if cert:
            from oracle import run_equiv
            r = run_equiv(out, [insts[i] for i in idxs], [refs[i] for i in idxs])
            ok = r.get("nmiss", 1) == 0 and r.get("nuncov", 1) == 0
            print(f"  CERTIFY (equiv.dl): ncover={r.get('ncover')} nmiss={r.get('nmiss')} nuncov={r.get('nuncov')} → "
                  + ("CERTIFIED — circuits.dl == model over the corpus (souffle-only)" if ok else "NOT certified"))

    print("\nselect = one operand → lookup · compose = two operands → computation · copy/induction = content-relative pointer. "
          "All learned from behavior, nothing hand-coded; the CAUSAL test is the universal discriminator.")


if __name__ == "__main__":
    main()
