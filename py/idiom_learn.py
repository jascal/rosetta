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


class OracleUnavailable(RuntimeError):
    """Raised when the causal oracle can't reproduce known refs — we refuse to emit a false 'no idioms'."""


def assert_oracle_live(decide_fn, insts, refs, idxs, label="", nprobe=8, thresh=0.75):
    """Preflight guard for the CAUSAL phase. Confirmation compares decide_fn(perturbed) against a table; if the
    oracle is dead/misconfigured it returns None for every call, so every idiom scores causal=0 and the run
    SILENTLY reports 'no idioms' — byte-indistinguishable from a genuinely n-gram model. This probes the live
    oracle on instances whose refs are already known (from cache) and ABORTS LOUDLY unless it reproduces them,
    so a reported zero means the model, not the plumbing. (emit-only paths take pre-confirmed idioms and need
    no oracle, so they are not guarded here.)"""
    tag = f" · {label}" if label else ""
    known = [i for i in idxs if refs[i] is not None][:nprobe]
    if not known:
        raise OracleUnavailable(
            f"[oracle preflight{tag}] no known refs to probe — the refs oracle produced nothing, so causal "
            f"confirmation cannot run. Set FIELDRUN_BIN=<fieldrun binary>, or bring up `fieldrun --serve <port>` "
            f"and export FIELDRUN_SERVE=<port>. Refusing to run (would emit a false 'no idioms').")
    got = [decide_fn(insts[i]) for i in known]
    agree = sum(g is not None and g == refs[i] for g, i in zip(got, known))
    if agree < thresh * len(known):
        nnone = sum(g is None for g in got)
        raise OracleUnavailable(
            f"[oracle preflight{tag}] CAUSAL ORACLE NOT LIVE: reproduces only {agree}/{len(known)} known refs "
            f"({nnone} returned no answer). Every idiom would be silently rejected (causal=0) → a FALSE 'no idioms'. "
            f"Refusing to run. Fix the oracle: FIELDRUN_BIN=<binary>, or `fieldrun --serve <port>` + FIELDRUN_SERVE=<port>.")
    print(f"[oracle preflight{tag}] live — reproduced {agree}/{len(known)} known refs; causal confirmation trustworthy.")
    return agree, len(known)


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


# ---------------------------------------------------------------------------------------------------------------------
# GRAMMAR / closed-class SKELETON family (the cross-content generalizer): keep the most-frequent tokens (function words /
# punctuation / structural), collapse content tokens to a sentinel O, and predict from the SKELETON — one rule for every
# context sharing a syntactic frame, regardless of content. Causally confirmed by CONTENT-INVARIANCE: perturbing the
# content (O) positions must NOT change the output (proof the skeleton, not the content, drives it). Ported from fieldrun's
# `explain` GRAMMAR idiom (closed-class skeleton → successor), which lexical n-grams miss because they memorize content.

OCONTENT = -1   # the content sentinel in a skeleton


def learn_skeleton(insts, refs, idxs, w, decide_fn, closed_n=40, fill=None, max_confirm=24, ntest=8):
    freq = Counter(t for i in idxs for t in insts[i])
    closed = {t for t, _ in freq.most_common(closed_n)}            # closed class = the most frequent tokens (unsupervised)
    skel = lambda ctx, k: tuple(t if t in closed else OCONTENT for t in ctx[-k:])
    pool = list({t for i in idxs for t in insts[i]} - closed)      # content tokens, for the invariance perturbation
    cands = {}
    for k in range(1, w + 1):
        groups = defaultdict(list)
        for i in idxs:
            if len(insts[i]) >= k:
                groups[skel(insts[i], k)].append(i)
        for sk, members in groups.items():
            if OCONTENT not in sk:                                 # all function words = a literal n-gram (cover handles it)
                continue
            if len({refs[i] for i in members}) == 1 and len({tuple(insts[i][-k:]) for i in members}) >= 2:
                cands[(k, sk)] = members                           # skeleton determines output across ≥2 distinct contents
    ranked = sorted(cands.items(), key=lambda kv: -len(kv[1]))[:max_confirm]
    import random as _r
    rng = _r.Random(0)
    out = []
    for (k, sk), members in ranked:
        tgt = refs[members[0]]
        opos = [j for j in range(k) if sk[j] == OCONTENT]
        tests = []
        for i in members[:ntest]:
            c = insts[i][:]
            for j in opos:                                         # replace every content slot with random content
                if pool:
                    c[len(c) - k + j] = rng.choice(pool)
            tests.append(c)
        if fill:
            fill(tests)
        inv = sum(1 for c in tests if decide_fn(c) == tgt) / len(tests) if tests else 0.0
        out.append(dict(k=k, sk=sk, members=members, out=tgt, inv=inv,
                        ncontent=len({tuple(insts[i][-k:]) for i in members})))
    return out, closed


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


def emit_canonical_T(md, insts, idxs, gates, comps, rels, w, sym, name, get_lg, T=1.0, eps=0.02, T_lo=0.7):
    """FINAL STEP of extraction: emit the CANONICAL distributional circuits.dl (+ run.dl + symbols twin) from the LEARNED
    idioms carrying DISTRIBUTIONS. compose (sum→top-K logits) and select-gate (content→top-K logits) each keep a key only
    where the model's softmax is CONSISTENT within that key's group across the T-range (temperature.dist_table) — so the
    idiom genuinely carries+generalizes the distribution, certifiably; inconsistent contexts fall back to the n-gram cover.
    Routing compose > gate > n-gram > induction(OOD). This is the distributional twin of emit_circuits, and the canonical
    artifact (vs equiv.dl's exact-argmax cert on the crisp emit). Returns the across-range verdict."""
    from temperature import dist_table, dist_cover, finalize          # the distributional machinery lives in temperature
    cache_p = os.path.join(md, "logit_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    kf = lambda c: ",".join(map(str, c))
    for j, i in enumerate(idxs):                                       # gather the T-invariant logit scoreboards (cached)
        if kf(insts[i]) not in cache:
            lg = get_lg(insts[i])
            if lg:
                cache[kf(insts[i])] = lg
            if j % 50 == 0:
                json.dump(cache, open(cache_p, "w")); print(f"   …{j}/{len(idxs)} logits")
    json.dump(cache, open(cache_p, "w"))
    logmap = {i: [(int(v), float(s)) for v, s in cache[kf(insts[i])]] for i in idxs if kf(insts[i]) in cache}
    lidx = sorted(logmap)
    idioms, covered = [], set()
    for j, b in enumerate(comps):                                      # compose (highest priority) carrying a distribution
        mem = [i for i in lidx if all(len(insts[i]) >= m and insts[i][-m] == v for m, v in b["frame"].items())
               and len(insts[i]) >= max(b["k1"], b["k2"]) and insts[i][-b["k1"]] in b["lab"] and insts[i][-b["k2"]] in b["lab"]]
        csum, bad = dist_table(insts, logmap, mem, lambda c, b=b: b["lab"][c[-b["k1"]]] + b["lab"][c[-b["k2"]]], T_lo, T, eps)
        if csum:
            idioms.append(dict(kind="compose", name=f"comp{j}", frame=b["frame"], k1=b["k1"], k2=b["k2"], valmap=b["lab"], csum=csum))
            covered |= {i for i in mem if i not in bad}
    for j, b in enumerate(gates):                                      # select-gate carrying a distribution
        mem = [i for i in lidx if i not in covered and all(len(insts[i]) >= m and insts[i][-m] == v for m, v in b["frame"].items())
               and len(insts[i]) >= b["k"] and insts[i][-b["k"]] in b["table"]]
        tab, bad = dist_table(insts, logmap, mem, lambda c, b=b: c[-b["k"]], T_lo, T, eps)
        if tab:
            idioms.append(dict(kind="gate", name=f"gate{j}", frame=b["frame"], k=b["k"], tab=tab))
            covered |= {i for i in mem if i not in bad}
    residual = [i for i in lidx if i not in covered]                  # n-gram distributional backfill on the rest
    rules, remaining = dist_cover(insts, logmap, residual, T_lo, T, eps, w)
    induction = any(r["L"] == 1 for r in rels)                        # copy/induction OOD fallback (structural point-mass)
    src = "a fieldrun --serve /topk server" if os.environ.get("FIELDRUN_SERVE") else ("whole.dl" if os.path.exists(os.path.join(md, "whole.dl")) else "the cached logits")
    print(f"\n=== EMIT (canonical, distributional — the FINAL STEP of extraction) ===")
    print(f"  learned idioms carrying distributions cover {len(covered)}/{len(lidx)} windows (the rest → n-gram backfill)")
    return finalize(md, insts, logmap, lidx, idioms, rules, remaining, induction, w, sym, name, T, eps, T_lo, src)


def emit_expert_package(md, insts, refs, idxs, real, real_c, rels_real, w, name, minsupp=3, mindet=1.0):
    """The bounded-EXPERT package (rosetta→sgiandubh convergence): CAUSALLY-CONFIRMED idioms as the TRUSTED (ungated) tier
    + a GATED n-gram backfill (support/determinism → abstain on weak suffixes) + a manifest that distinguishes causal
    idioms from observational n-grams (with provenance). The strengthening over the corpus-only abstain_emit path: the
    bounded expert is built from CONFIRMED computation (the causal idioms), not just corpus correlation. Reuses
    emit_circuits (idiom > gated-ngram > induction(OOD) > abstain — abstain is automatic where no rule fires). Writes the
    package to md/package/ (circuits.expert.dl + run.dl + manifest.json)."""
    from abstain_emit import build_tab, confident_rules
    pkg = os.path.join(md, "package"); os.makedirs(pkg, exist_ok=True)

    def gate_cov(b):
        return [i for i in idxs if all(len(insts[i]) >= m and insts[i][-m] == v for m, v in b["frame"].items())
                and len(insts[i]) >= b["k"] and insts[i][-b["k"]] in b["table"] and refs[i] == b["table"][insts[i][-b["k"]]]]

    def comp_cov(b):
        out = []
        for i in idxs:
            ctx = insts[i]
            if all(len(ctx) >= m and ctx[-m] == v for m, v in b["frame"].items()) and len(ctx) >= max(b["k1"], b["k2"]):
                a, bb = ctx[-b["k1"]], ctx[-b["k2"]]
                if a in b["lab"] and bb in b["lab"] and refs[i] == b["bysum"].get(b["lab"][a] + b["lab"][bb]):
                    out.append(i)
        return out

    covered, man, rid = set(), [], 0
    for b in real_c:                                                   # compose idioms — TRUSTED (causal)
        c = comp_cov(b); covered |= set(c)
        man.append({"id": rid, "kind": "compose", "tier": "trusted", "basis": "causal", "causal": round(b.get("causal", 0), 3),
                    "extrapolate": (None if b.get("extrap") is None else round(b["extrap"], 3)), "support": len(c),
                    "frame": {int(k): int(v) for k, v in b["frame"].items()}, "operands": [b["k1"], b["k2"]],
                    "valmap": {int(t): int(v) for t, v in b["lab"].items()},      # token → operand value
                    "sum": {int(s): int(o) for s, o in b["bysum"].items()},        # value-sum → output token (so a consumer can evaluate it)
                    "cite": c[:5]}); rid += 1
    for b in real:                                                     # select-gate idioms — TRUSTED (causal)
        c = gate_cov(b); covered |= set(c)
        man.append({"id": rid, "kind": "gate", "tier": "trusted", "basis": "causal", "causal": round(b.get("causal", 0), 3),
                    "support": len(c), "frame": {int(k): int(v) for k, v in b["frame"].items()}, "slot": b["k"],
                    "table": {int(k): int(v) for k, v in b["table"].items()}, "cite": c[:5]}); rid += 1
    residual = [i for i in idxs if i not in covered]                  # GATED n-gram backfill on what idioms didn't cover
    tab, cites = build_tab([(tuple(insts[i]), refs[i], i) for i in residual], w)
    conf = confident_rules(tab, cites, w, minsupp, mindet)
    ngram_rules = {}
    for k in conf:
        for s, (o, sup, det, cite) in conf[k].items():
            ngram_rules[s] = o
            man.append({"id": rid, "kind": "ngram", "tier": "gated", "basis": "observational", "ctx": list(s),
                        "out": o, "support": sup, "determinism": round(det, 3), "cite": cite}); rid += 1
    out = os.path.join(pkg, "circuits.expert.dl")
    emit_circuits(out, real, real_c, rels_real, ngram_rules, w, name)  # idiom > gated-ngram > induction(OOD) > abstain
    json.dump({"model": name, "trusted_idioms": len(real) + len(real_c), "gated_ngrams": len(ngram_rules),
               "induction_ood": len(rels_real), "minsupp": minsupp, "mindet": mindet, "rules": man},
              open(os.path.join(pkg, "manifest.json"), "w"))
    ngcov = sum(1 for i in residual if any(tuple(insts[i][-k:]) in conf.get(k, {}) for k in range(1, w + 1)))
    H = len(idxs)
    print(f"\n=== EXPERT PACKAGE (bounded, causal-confirmed) → {pkg}/ ===")
    print(f"  TRUSTED idioms (causal): {len(real_c)} compose + {len(real)} gate → cover {len(covered)}/{H} = {len(covered)/H:.0%}")
    print(f"  GATED n-grams (supp>={minsupp},det>={mindet}): {len(ngram_rules)} rules → cover {ngcov}/{H} = {ngcov/H:.0%} of residual")
    cov = (len(covered) + ngcov) / H
    print(f"  bounded-expert composition: trusted {len(covered)/H:.0%} + gated {ngcov/H:.0%} = answer {cov:.0%}, abstain {1-cov:.0%}")
    print(f"  manifest.json: {len(man)} rules tagged causal(trusted) vs observational(gated) + provenance — the audit of what the expert knows")
    return out, os.path.join(pkg, "manifest.json")


def select_cover(insts, refs, idxs, w, decide_fn, fill=None, hold=0.3, s=str):
    """The reframed learner: learn every family on TRAIN (causal-soundness = the gate), then GREEDILY admit the family
    with the best Δcorrect-holdout ÷ Δrules and stop when nothing pays (minimize holdout loss, bias to fewer rules — the
    IDIOM_LEARNER objective). Reports the residual floor (what no family captures = the model, not us)."""
    assert_oracle_live(decide_fn, insts, refs, idxs, label="select_cover")   # a dead oracle → false 'no idioms'
    import random as _r
    sh = idxs[:]; _r.Random(0).shuffle(sh)
    cut = int(len(sh) * (1 - hold))
    train, holdout = sh[:cut], sh[cut:]
    gates = [b for b in learn_gates(insts, refs, train, w, decide_fn, fill=fill) if not b["viol"] and b["causal"] >= 0.8]
    comps = [b for b in learn_compose(insts, refs, train, w, decide_fn, fill=fill) if b["causal"] >= 0.8]
    rels = [r for r in learn_relational(insts, refs, train, decide_fn, fill=fill) if r["causal"] >= 0.8 and r["obs"] >= 0.5]
    sks, closed = learn_skeleton(insts, refs, train, w, decide_fn, fill=fill)
    sks = [b for b in sks if b["inv"] >= 0.8 and b["ncontent"] >= 2]
    ng = minimal_suffix_cover(insts, refs, train, w)[0]
    skr, OC = {}, OCONTENT
    skl = lambda c, k: tuple(t if t in closed else OC for t in c[-k:])
    for b in sks:
        skr[b["sk"]] = b["out"]

    def ng_pred(ctx):
        for k in range(min(len(ctx), w), 0, -1):
            o = ng.get(tuple(ctx[-k:]))
            if o is not None:
                return o
        return None

    def gate_pred(ctx):
        for b in gates:
            if all(len(ctx) >= m and ctx[-m] == v for m, v in b["frame"].items()) and len(ctx) >= b["k"] and ctx[-b["k"]] in b["table"]:
                return b["table"][ctx[-b["k"]]]
        return None

    def comp_pred(ctx):
        for b in comps:
            if all(len(ctx) >= m and ctx[-m] == v for m, v in b["frame"].items()) and len(ctx) >= max(b["k1"], b["k2"]):
                a, bb = ctx[-b["k1"]], ctx[-b["k2"]]
                if a in b["lab"] and bb in b["lab"]:
                    o = b["bysum"].get(b["lab"][a] + b["lab"][bb])
                    if o is not None:
                        return o
        return None

    def ind_pred(ctx):
        for L in range(min(3, len(ctx) - 1), 0, -1):
            suf = tuple(ctx[-L:]); js = [j for j in range(len(ctx) - L) if tuple(ctx[j:j + L]) == suf]
            if js and max(js) + L < len(ctx):
                return ctx[max(js) + L]
        return None

    def sk_pred(ctx):
        for k in range(min(len(ctx), w), 0, -1):
            o = skr.get(skl(ctx, k))
            if o is not None:
                return o
        return None

    H = len(holdout)

    def score(fns):
        n = 0
        for i in holdout:
            for f in fns:
                p = f(insts[i])
                if p is not None:
                    n += (p == refs[i]); break
        return n

    FAM = {"compose": (comp_pred, sum(len(b["lab"]) + len(b["bysum"]) for b in comps)),
           "select": (gate_pred, sum(len(b["table"]) for b in gates)),
           "induction": (ind_pred, 0), "skeleton": (sk_pred, len(skr))}
    FAM = {k: v for k, v in FAM.items() if (gates if k == "select" else comps if k == "compose"
           else sks if k == "skeleton" else True)}
    covt = {nm: {i for i in train if fn(insts[i]) == refs[i]} for nm, (fn, _) in FAM.items()}   # train coverage per family

    def build(adm):
        """re-derive the n-gram cover on the residual AFTER the admitted families — so a family pays by SHRINKING the
        n-gram cover (extrapolation/compression credit: a generalizer that covers many contexts with few rules wins MDL
        even at 0 marginal holdout, because it replaces the n-grams it made redundant)."""
        covered = set().union(*[covt[nm] for nm in adm]) if adm else set()
        ng2 = minimal_suffix_cover(insts, refs, [i for i in train if i not in covered], w)[0]
        rules = sum(FAM[nm][1] for nm in adm) + len(ng2)

        def pred(ctx):
            for nm in adm:
                p = FAM[nm][0](ctx)
                if p is not None:
                    return p
            for k in range(min(len(ctx), w), 0, -1):
                o = ng2.get(tuple(ctx[-k:]))
                if o is not None:
                    return o
            return None
        return rules, sum(pred(insts[i]) == refs[i] for i in holdout)

    lam = 0.5                                                          # rule cost in holdout-window units (MDL knob)
    obj = lambda r, k: (H - k) + lam * r
    R0, k0 = build([])
    print(f"  baseline n-gram: {R0} rules, holdout {k0}/{H}={k0/H:.0%}  obj {obj(R0, k0):.1f}")
    adm, cur_obj, pend = [], obj(R0, k0), set(FAM)
    while pend:
        (r, k), nm = min(((build(adm + [nm]), nm) for nm in pend), key=lambda o: obj(*o[0]))
        if obj(r, k) >= cur_obj - 1e-9:
            print(f"  — not admitted (no holdout+MDL gain): {', '.join(sorted(pend))}")
            break
        adm.append(nm); pend.discard(nm); cur_obj = obj(r, k)
        print(f"  + {nm}: {r} rules (Δ{r - R0:+d}), holdout {k}/{H}={k/H:.0%}, obj {obj(r, k):.1f}")
    rr, kk = build(adm)
    print(f"  ⇒ n-gram" + "".join(f" + {nm}" for nm in adm) + f"  · holdout {kk/H:.0%} · {rr} rules (baseline {R0}) · residual {1-kk/H:.0%}")
    return adm, kk, H


def main():
    backfill = "--backfill" in sys.argv                       # report idiom coverage + n-gram backfill stats
    emit = "--emit" in sys.argv                               # write circuits.dl + run.dl (idioms + n-gram cover, souffle-only)
    cert = "--certify" in sys.argv                            # prove the emitted circuits.dl == model via equiv.dl (cached refs)
    cb = next((int(x.split("=")[1]) for x in sys.argv if x.startswith("--confirm=")), None)   # confirmation budget per family
    a = [x for x in sys.argv[1:] if not x.startswith("--")]   # (cap candidates causally confirmed — big/slow models need a small one)
    n = int(a[0]) if len(a) > 0 else 1400
    w = int(a[1]) if len(a) > 1 else 8
    md = a[2] if len(a) > 2 else os.path.join(HERE, "reference", "threx")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    name = os.path.basename(md.rstrip("/"))
    from temperature import build_sym
    sym = build_sym(md)                                         # lexicon.json, else decode via the bundle tokenizer
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

    if "--select" in sys.argv:                                 # holdout+MDL admission over the full family set (the objective)
        print("=== holdout+MDL family selection — minimize holdout loss, bias to fewer rules (causal-soundness = gate) ===")
        select_cover(insts, refs, idxs, w, decide_fn, fill=fill, s=s)
        return

    assert_oracle_live(decide_fn, insts, refs, idxs, label=name)   # abort loudly if causal can't reproduce known refs
    gates = learn_gates(insts, refs, idxs, w, decide_fn, max_confirm=(cb or 40), ntest=(6 if cb else 14), fill=fill)
    real = [b for b in gates if not b["viol"] and b["causal"] >= 0.8]
    print(f"frame-conditioned GATEs (select family) — {len(real)} REAL (faithful + causally confirmed) of {len(gates)} mined:\n")
    for b in real[:20]:
        fr = " ".join(f"@{m}={s(v)}" for m, v in sorted(b["frame"].items())) or "∅ (unconditional)"
        td = "{" + ", ".join(f"{s(t)}→{s(o)}" for t, o in sorted(b["table"].items())) + "}"
        print(f"  [causal {b['causal']:.0%}] GATE@{b['k']}  support={len(b['support'])}  ignore@{b['ignore']}")
        print(f"       frame[{fr}]")
        print(f"       {td}")
    comps = learn_compose(insts, refs, idxs, w, decide_fn, max_confirm=(cb or 30), fill=fill)
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

    sks, closed = learn_skeleton(insts, refs, idxs, w, decide_fn, fill=fill)
    real_sk = [b for b in sks if b["inv"] >= 0.8 and b["ncontent"] >= 2]
    print(f"\nGRAMMAR/SKELETON family (closed-class frame, content→O; causal = CONTENT-INVARIANCE) — {len(real_sk)} real:")
    for b in sorted(real_sk, key=lambda b: -len(b["members"]))[:8]:
        skd = " ".join("O" if t == OCONTENT else s(t) for t in b["sk"])
        print(f"  [content-invariant {b['inv']:.0%}] covers {len(b['members'])} ({b['ncontent']} contents)  [{skd}] → {s(b['out'])}")

    if "--package" in sys.argv:                                   # bounded-EXPERT package: causal idioms (trusted) + gated n-grams
        rels_real = [r for r in rels if r["causal"] >= 0.8 and r["obs"] >= 0.5]
        ms = next((int(x.split("=")[1]) for x in sys.argv if x.startswith("--minsupp=")), 3)
        mdv = next((float(x.split("=")[1]) for x in sys.argv if x.startswith("--mindet=")), 1.0)
        emit_expert_package(md, insts, refs, idxs, real, real_c, rels_real, w, name, ms, mdv)

    if emit or cert:
        rels_real = [r for r in rels if r["causal"] >= 0.8 and r["obs"] >= 0.5]   # induction → OOD fallback (below n-grams)
        if "--crisp" in sys.argv:                                 # legacy: crisp argmax circuits.dl + the EXACT equiv.dl cert
            covered, _ = idiom_coverage(insts, refs, idxs, real, real_c, [])      # (research path; canonical artifact is the T one)
            residual = [i for i in idxs if i not in covered]
            rules, remaining, _ = minimal_suffix_cover(insts, refs, residual, w)
            out = os.path.join(md, "circuits.dl")
            emit_circuits(out, real, real_c, rels_real, rules, w, name)
            print(f"\n=== EMIT (crisp/argmax) → {out} ===")
            print(f"  {len(real_c)} compose + {len(real)} select-gate idioms (faithful, in-domain) cover {len(covered)}/{len(idxs)};"
                  f" {len(rules)} n-gram rules memoize the residual" + (f"; {len(remaining)} uncovered" if remaining else "")
                  + (f"; {len(rels_real)} induction OOD fallback" if rels_real else "") + ".  + run.dl (souffle-only harness).")
            from oracle import run_equiv
            r = run_equiv(out, [insts[i] for i in idxs], [refs[i] for i in idxs])
            ok = r.get("nmiss", 1) == 0 and r.get("nuncov", 1) == 0
            print(f"  CERTIFY (equiv.dl, EXACT argmax): ncover={r.get('ncover')} nmiss={r.get('nmiss')} nuncov={r.get('nuncov')} → "
                  + ("CERTIFIED — circuits.dl == model over the corpus (souffle-only)" if ok else "NOT certified"))
        else:                                                     # CANONICAL (default): the distributional T circuits.dl + symbols,
            from oracle import logits as model_logits, serve_topk  # learned idioms carrying distributions, as the FINAL STEP
            serve = os.environ.get("FIELDRUN_SERVE"); whole = os.path.join(md, "whole.dl")
            if serve:
                get_lg = lambda c: serve_topk(int(serve), c)
            elif os.path.exists(whole):
                get_lg = lambda c: model_logits(whole, c)
            else:
                get_lg = lambda c: None                           # cache-only: regenerate from logit_cache.json
            emit_canonical_T(md, insts, idxs, real, real_c, rels_real, w, sym, name, get_lg)

    print("\nselect = one operand → lookup · compose = two operands → computation · copy/induction = content-relative pointer. "
          "All learned from behavior, nothing hand-coded; the CAUSAL test is the universal discriminator.")


if __name__ == "__main__":
    main()
