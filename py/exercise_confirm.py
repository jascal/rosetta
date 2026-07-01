#!/usr/bin/env python3
"""rosetta · exercise_confirm.py — the EXERCISE-then-CONFIRM regression for idiom-finding.

The learner recovers a circuit only when the corpus EXERCISES it (makes the circuit the *sole* way to predict). A
model's own (greedy/natural) corpus masks its circuits behind n-gram statistics — so "no idioms on the natural corpus"
is NOT evidence the circuit is absent. This harness proves that per circuit, on a REAL model, with two asserted bars:

  RECOVERY   (detect + CAUSAL under exercising stimuli): the model runs the circuit, and perturbing its operand makes
             the output follow — the circuit is genuinely present (not a correlation). Induction adds the EXERCISE-vs-
             NATURAL contrast (recovered when exposed, masked on the model's own corpus — the gap is the mask).
  ADMISSION  (holdout generalization vs n-grams): on a corpus that exercises the circuit with a HELD-OUT region (novel
             content the n-gram cover could never memorize), the circuit RULE matches the model where the memorized
             n-gram cover can't. Beating n-grams on holdout is what earns a place in circuits.dl — the "did we capture
             the algorithm, or just memorize" bar (cover_structured, generalized).

Circuits: induction/copy · succession/ordinal · IOI/name-mover. Registry is extensible. Recovery under exercise is the
rebuttal to "only works on threx"; a KNOWN circuit failing recovery means the oracle/harness is broken (complements
assert_oracle_live). Oracle: FIELDRUN_SERVE=<port> (resident) · a fieldrun bundle · or a whole.dl — via ref_source;
succession/IOI additionally need bundle.tokenizer.json.
Usage:  FIELDRUN_SERVE=<port> .venv/bin/python py/exercise_confirm.py <model_dir> [--only=induction,succession,ioi] [--json=OUT]
"""
import sys, os, json, random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimize import ref_source, instances, minimal_suffix_cover
from idiom_learn import learn_relational

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU = 0.8              # recovery / causal threshold (matches the learner's gate)
OBS_MIN = 0.5         # observational/detect floor (a real circuit must also predict, not just follow perturbations)
LETTERS = [f" {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]


def vocab_of(md):
    """Model vocab — to sample novel tokens inside a valid id range. vocab is the largest model dimension in these
    bundles (pythia160m config=[12,12,64,768,3072,50304,16,1] → 50304), so take max(config); else the lexicon length;
    else a default. (Bundle config field ORDER varies by arch — max is order-robust where an index is not.)"""
    import glob
    for p in glob.glob(os.path.join(md, "*.fieldrun.json")):
        cfg = json.load(open(p)).get("config")
        if isinstance(cfg, list) and cfg:
            return int(max(cfg))
    lex = os.path.join(md, "lexicon.json")
    if os.path.exists(lex):
        return len(json.load(open(lex))["tokens"])
    return 40000


def make_oracle(md):
    """(dec, fill, label) over the model's build-time oracle (serve / fieldrun / whole.dl) with cache + parallel batcher."""
    label, raw = ref_source(md)
    cache = {}
    serve = os.environ.get("FIELDRUN_SERVE")

    def dec(ctx):
        k = tuple(ctx)
        if k not in cache:
            cache[k] = raw(list(ctx))
        return cache[k]

    def fill(ctxs):
        miss = list({tuple(c) for c in ctxs} - set(cache))
        if not miss:
            return
        workers = 2 if serve else 8          # a resident server is sequential; few clients suffice
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for c, o in zip(miss, ex.map(lambda t: raw(list(t)), miss)):
                cache[c] = o

    return dec, fill, label


def holdout_vs_ngram(insts, refs, train_idx, hold_idx, rule_pred, w):
    """The ADMISSION bar. Build a minimal-suffix n-gram cover on TRAIN; on the HELD-OUT region (structure the cover never
    saw) compare the circuit RULE vs the n-gram cover — BOTH scored against the model's refs. The circuit admits iff it
    matches the model on held-out contexts the memorized cover can't. (rule_pred(ctx) -> id or None; None = abstain.)"""
    ng = minimal_suffix_cover(insts, refs, [i for i in train_idx if refs[i] is not None], w)[0]

    def ng_pred(ctx):
        for k in range(min(len(ctx), w), 0, -1):
            o = ng.get(tuple(ctx[-k:]))
            if o is not None:
                return o
        return None

    valid = [i for i in hold_idx if refs[i] is not None]
    cm = sum(1 for i in valid if rule_pred(insts[i]) == refs[i])
    nm = sum(1 for i in valid if ng_pred(insts[i]) == refs[i])
    return dict(circuit_match=cm, ngram_match=nm, n_hold=len(valid), delta=cm - nm,
                admits=cm > nm, ngram_rules=len(ng))


# ==== INDUCTION / COPY ==============================================================================================
def induction_exercise(vocab, n_seqs=40, seqlen=22):
    """Repeated NOVEL sequences S + S[:k]: predicting S[k] is reachable ONLY by copying the first occurrence."""
    lo, hi = max(5, vocab // 50), max(10, vocab - 1)
    seqs = []
    for seed in range(n_seqs):
        r = random.Random(seed)
        seqs.append([r.choice(range(lo, hi)) for _ in range(seqlen)])
    insts, seedof = [], []
    for si, S in enumerate(seqs):
        for k in range(1, seqlen - 1):
            insts.append(S + S[:k]); seedof.append(si)
    return insts, seedof


def circuit_induction(md, dec, fill, tok, vocab):
    insts, seedof = induction_exercise(vocab)
    fill(insts)
    refs = [dec(c) for c in insts]
    idxs = [i for i in range(len(insts)) if refs[i] is not None]
    rels = learn_relational(insts, refs, idxs, dec, fill=fill, maxL=3)
    rows = [{"label": f"L{r['L']}", "applicable": r["applicable"], "detect": r["obs"], "causal": r["causal"],
             "recovered": r["causal"] >= TAU and r["obs"] >= OBS_MIN} for r in rels]
    # NATURAL contrast — same learner measure on the model's own corpus (the masked regime)
    nat = instances(json.load(open(os.path.join(md, "corpus.json")))["ids"], 600, 16)
    fill(nat)
    nrefs = [dec(c) for c in nat]
    nidx = [i for i in range(len(nat)) if nrefs[i] is not None]
    nrel = learn_relational(nat, nrefs, nidx, dec, fill=fill, maxL=3)
    natrows = {r["L"]: r["causal"] for r in nrel}
    for row in rows:
        row["natural_causal"] = natrows.get(int(row["label"][1:]), 0.0)
    # ADMISSION — copy rule (L=1) vs n-gram cover on a NOVEL-seed holdout (train seeds even, holdout seeds odd)
    train = [i for i in idxs if seedof[i] % 2 == 0]
    hold = [i for i in idxs if seedof[i] % 2 == 1]

    def copy_rule(ctx):
        last = ctx[-1]
        js = [j for j in range(len(ctx) - 1) if ctx[j] == last]
        return ctx[max(js) + 1] if js and max(js) + 1 < len(ctx) else None

    ho = holdout_vs_ngram(insts, refs, train, hold, copy_rule, w=8)
    return {"rows": rows, "holdout": ho, "has_natural": True}


# ==== SUCCESSION / ORDINAL ==========================================================================================
def _single(tok, words):
    out = {}
    for wd in words:
        ids = tok.encode(wd, add_special_tokens=False).ids
        if len(ids) == 1:
            out[wd] = ids[0]
    return out


def circuit_succession(md, dec, fill, tok, vocab):
    idmap = _single(tok, LETTERS)
    letters = [c for c in LETTERS if c in idmap]
    if len(letters) < 8:
        return None
    pos = {c: i for i, c in enumerate(letters)}
    runs = [(letters[i], letters[i + 1], letters[i + 2], letters[i + 3]) for i in range(len(letters) - 3)]
    # FORMAT-ROBUST (CROSS_ARCH.md: a 0% is often the probe, not the model — a code model reads bare ' A B C' as a token
    # list but 'A, B, C,' as a sequence). Try both joins; keep the one the model actually continues (highest detect).
    fmts = [lambda a, b, c: f"{a}{b}{c}", lambda a, b, c: f"{a},{b},{c},"]
    best = None
    for fmt in fmts:
        ins = [tok.encode(fmt(a, b, c)).ids for a, b, c, _ in runs]
        fill(ins)
        rf = [dec(x) for x in ins]
        tr = [idmap[d] for _, _, _, d in runs]
        det = sum(rf[i] == tr[i] for i in range(len(runs))) / len(runs)
        if best is None or det > best[0]:
            best = (det, fmt, ins, rf, tr)
    detect, fmt, insts, refs, truth = best
    n = len(insts)
    # CAUSAL — shift the whole run one letter up; the predicted successor must shift with it (tracks ordinal position,
    # not a memorized token). Confirmed where the model does the base run at all.
    shifted, sh_truth, base = [], [], []
    for i in range(len(letters) - 4):
        shifted.append(tok.encode(fmt(letters[i + 1], letters[i + 2], letters[i + 3])).ids)
        sh_truth.append(idmap[letters[i + 4]]); base.append(i)
    fill(shifted)
    did = [i for i in base if refs[i] == truth[i]]             # runs the model actually continues
    causal = (sum(1 for k, i in enumerate(base) if i in did and dec(shifted[k]) == sh_truth[k]) / len(did)) if did else 0.0
    rows = [{"label": "succ", "applicable": n, "detect": detect, "causal": causal,
             "recovered": detect >= TAU and causal >= TAU}]
    # ADMISSION — HELD-OUT region: train = early-letter runs, holdout = late-letter runs (n-gram never saw the late transitions)
    cut = int(n * 0.6)
    train, hold = list(range(cut)), list(range(cut, n))

    def succ_rule(ctx):
        seq = []                                               # extract the alphabetic letters in order (ignore commas/spaces)
        for t in ctx:
            d = tok.decode([t]).strip()
            if len(d) == 1 and d.isalpha():
                p = pos.get(" " + d.upper())
                if p is not None:
                    seq.append(p)
        if len(seq) >= 3 and seq[-2] == seq[-1] - 1 and seq[-3] == seq[-1] - 2 and seq[-1] + 1 < len(letters):
            return idmap[letters[seq[-1] + 1]]
        return None

    ho = holdout_vs_ngram(insts, refs, train, hold, succ_rule, w=6)
    return {"rows": rows, "holdout": ho, "has_natural": False, "stimuli": (insts, refs)}


# ==== IOI / NAME-MOVER =============================================================================================
NAMES = [" John", " Mary", " Tom", " Sara", " Paul", " Anna", " Mark", " Lucy", " Mike", " Emma", " David", " Kate",
         " James", " Laura", " Peter", " Alice", " Henry", " Julia", " Robert", " Nancy"]


def circuit_ioi(md, dec, fill, tok, vocab, n=60):
    nm = _single(tok, NAMES)
    names = list(nm)
    if len(names) < 6:
        return None
    name_ids = set(nm.values())
    rng = random.Random(0)
    cut = len(names) * 2 // 3
    train_names, hold_names = names[:cut], names[cut:]         # HOLDOUT uses names never seen in training

    def build(pool, count):
        insts, truth, swp, swp_truth = [], [], [], []
        for _ in range(count):
            a, b = rng.sample(pool, 2)
            insts.append(tok.encode(f"When{a} and{b} went to the store,{b} gave a drink to").ids); truth.append(nm[a])
            swp.append(tok.encode(f"When{b} and{a} went to the store,{a} gave a drink to").ids); swp_truth.append(nm[b])
        return insts, truth, swp, swp_truth

    tr_i, tr_t, tr_s, tr_st = build(train_names, n)
    ho_i, ho_t, ho_s, ho_st = build(hold_names, n)
    insts = tr_i + ho_i
    truth = tr_t + ho_t
    fill(insts + tr_s + ho_s)
    refs = [dec(c) for c in insts]
    detect = sum(refs[i] == truth[i] for i in range(len(insts))) / len(insts)
    swp, swp_truth = tr_s + ho_s, tr_st + ho_st
    causal = sum(dec(swp[i]) == swp_truth[i] for i in range(len(swp))) / len(swp)   # swap duplicated name → answer follows
    rows = [{"label": "ioi", "applicable": len(insts), "detect": detect, "causal": causal,
             "recovered": detect >= TAU and causal >= TAU}]
    # ADMISSION — the name-mover RULE (the once-appearing name) vs the n-gram cover, on UNSEEN-name holdout
    train, hold = list(range(len(tr_i))), list(range(len(tr_i), len(insts)))

    def ioi_rule(ctx):
        present = [t for t in ctx if t in name_ids]
        once = [t for t, c in Counter(present).items() if c == 1]
        return once[0] if len(once) == 1 else None

    ho = holdout_vs_ngram(insts, refs, train, hold, ioi_rule, w=8)
    return {"rows": rows, "holdout": ho, "has_natural": False, "stimuli": (insts, refs)}


# ==== the exhaustive catalog ========================================================================================
# Two mechanistic classes (the taxonomy IS the finding):
#   STRUCTURAL (admittable): the answer is an in-context token computable by a token rule — copy(induction),
#     ordinal(succession), or the once-appearing entity (IOI + the stated-consequent reasoning families). These reduce
#     to a cover rule → we measure the holdout ADMISSION (beat n-grams on unseen content) and can EMIT them.
#   SEMANTIC / RECALL (recovery-only): the answer needs weights (antonym, capital, analogy) or semantic binding
#     (coreference, set, defeasible, causal). No structural token rule → admission N/A → not emittable. (This is the
#     SAE-bridge negative concretized: composition/recall "doesn't factor through a sparse rule".)
NOUNS = [" dog", " cat", " bird", " fish", " tree", " rock", " star", " car", " boat", " house",
         " king", " wolf", " lion", " bear", " duck", " frog", " horse", " sheep", " goat", " mouse"]
ADJ = [" red", " big", " hot", " soft", " loud", " green", " tall", " round", " sharp", " heavy",
       " blue", " small", " cold", " hard", " quiet", " short", " flat", " light", " smooth", " warm"]
CAPITALS = [(" France", " Paris"), (" Germany", " Berlin"), (" Japan", " Tokyo"), (" Italy", " Rome"),
            (" Spain", " Madrid"), (" Russia", " Moscow"), (" China", " Beijing"), (" Egypt", " Cairo")]
ANTONYMS = [(" hot", " cold"), (" big", " small"), (" up", " down"), (" fast", " slow"), (" happy", " sad"),
            (" open", " closed"), (" black", " white"), (" good", " bad"), (" rich", " poor"), (" hard", " soft"),
            (" light", " dark"), (" high", " low"), (" old", " young"), (" wet", " dry")]


def once_appearing_rule(name_ids):
    """The name-mover / binding rule as a token function: among the class tokens present, the answer is the one that
    appears exactly once (the indirect object / the stated consequent). Shared by IOI + the reasoning families."""
    def rule(ctx):
        present = [t for t in ctx if t in name_ids]
        once = [t for t, c in Counter(present).items() if c == 1]
        return once[0] if len(once) == 1 else None
    return rule


def measure_structural(md, dec, fill, tok, words, build, w=8, n=50):
    """Generic STRUCTURAL family: answer = a once-appearing in-context member of `words`. build(rng, tok, ids) ->
    (ctx_ids, ans_id, causal_ctx_ids, causal_ans_id) or None. Recovery = detect(argmax==ans) + causal(argmax follows a
    changed consequent). Admission = the once-appearing rule vs n-grams on an UNSEEN-word holdout (train words / holdout words)."""
    ids = _single(tok, words)
    pool = list(ids)
    if len(pool) < 6:
        return None
    name_ids = set(ids.values())
    rng = random.Random(1234)
    cut = len(pool) * 2 // 3
    train_ids = {w: ids[w] for w in pool[:cut]}
    hold_ids = {w: ids[w] for w in pool[cut:]}

    def mk(sub, count):
        I, T, C, CT = [], [], [], []
        for _ in range(count):
            r = build(rng, tok, sub)
            if r:
                I.append(r[0]); T.append(r[1]); C.append(r[2]); CT.append(r[3])
        return I, T, C, CT

    trI, trT, trC, trCT = mk(train_ids, n)
    hoI, hoT, _, _ = mk(hold_ids, n)
    if not trI or not hoI:
        return None
    fill(trI + trC + hoI)
    detect = sum(dec(trI[i]) == trT[i] for i in range(len(trI))) / len(trI)
    causal = sum(dec(trC[i]) == trCT[i] for i in range(len(trC))) / len(trC)
    rows = [{"label": "struct", "applicable": len(trI), "detect": detect, "causal": causal,
             "recovered": detect >= TAU and causal >= TAU}]
    insts = trI + hoI
    refs = [dec(x) for x in insts]
    train, hold = list(range(len(trI))), list(range(len(trI), len(insts)))
    ho = holdout_vs_ngram(insts, refs, train, hold, once_appearing_rule(name_ids), w)
    return {"rows": rows, "holdout": ho, "has_natural": False, "rule": "once_appearing", "class": words,
            "stimuli": (insts, refs)}


def _b_transitivity(rng, tok, ids):
    pool = list(ids)
    if len(pool) < 4:
        return None
    x, y, z, z2 = rng.sample(pool, 4)
    return (tok.encode(f"A{x} is a{y}. A{y} is a{z}. So a{x} is a").ids, ids[z],
            tok.encode(f"A{x} is a{y}. A{y} is a{z2}. So a{x} is a").ids, ids[z2])


def _b_mp(rng, tok, ids):
    pool = list(ids)
    if len(pool) < 3:
        return None
    a, b, b2 = rng.sample(pool, 3)
    return (tok.encode(f"If something is{a}, it is{b}. This is{a}. So this is").ids, ids[b],
            tok.encode(f"If something is{a}, it is{b2}. This is{a}. So this is").ids, ids[b2])


def _b_temporal(rng, tok, ids):
    pool = list(ids)
    if len(pool) < 4:
        return None
    x, y, z, z2 = rng.sample(pool, 4)
    return (tok.encode(f"{x} comes before{y}.{y} comes before{z}. So{x} comes before").ids, ids[z],
            tok.encode(f"{x} comes before{y}.{y} comes before{z2}. So{x} comes before").ids, ids[z2])


def _b_spatial(rng, tok, ids):
    pool = list(ids)
    if len(pool) < 4:
        return None
    x, y, z, z2 = rng.sample(pool, 4)
    return (tok.encode(f"The{x} is left of the{y}. The{y} is left of the{z}. So the{x} is left of the").ids, ids[z],
            tok.encode(f"The{x} is left of the{y}. The{y} is left of the{z2}. So the{x} is left of the").ids, ids[z2])


def circuit_transitivity(md, dec, fill, tok, vocab):
    return measure_structural(md, dec, fill, tok, NOUNS, _b_transitivity)


def circuit_mp(md, dec, fill, tok, vocab):
    return measure_structural(md, dec, fill, tok, ADJ, _b_mp)


def circuit_temporal(md, dec, fill, tok, vocab):
    return measure_structural(md, dec, fill, tok, NOUNS, _b_temporal)


def circuit_spatial(md, dec, fill, tok, vocab):
    return measure_structural(md, dec, fill, tok, NOUNS, _b_spatial)


def circuit_syllogism(md, dec, fill, tok, vocab):
    sn, nn = _single(tok, NAMES), list(_single(tok, NOUNS))
    if len(sn) < 1 or len(nn) < 1:
        return None
    names = list(sn)

    def build(rng, tok, ids):                                          # answer = the once-appearing ADJ (the stated predicate)
        pool = list(ids)
        if len(pool) < 2:
            return None
        z, z2 = rng.sample(pool, 2); x, y = rng.choice(names), rng.choice(nn)
        return (tok.encode(f"Every{y} is{z}.{x} is a{y}. So{x} is").ids, ids[z],
                tok.encode(f"Every{y} is{z2}.{x} is a{y}. So{x} is").ids, ids[z2])
    return measure_structural(md, dec, fill, tok, ADJ, build)


# ---- SEMANTIC / RECALL (recovery-only) --------------------------------------------------------------------------
def _recovery_only(rows, na):
    return {"rows": rows, "holdout": None, "has_natural": False, "admission_na": na}


def _detcaus(dec, fill, dctx, dans, cctx, cans):
    fill(dctx + cctx)
    detect = sum(dec(dctx[i]) == dans[i] for i in range(len(dctx))) / max(1, len(dctx))
    causal = sum(dec(cctx[i]) == cans[i] for i in range(len(cctx))) / max(1, len(cctx))
    return detect, causal


def circuit_coreference(md, dec, fill, tok, vocab, n=60):
    nm = _single(tok, NAMES); names = list(nm)
    if len(names) < 4:
        return None
    rng = random.Random(2); dctx, dans, cctx, cans = [], [], [], []
    for _ in range(n):
        a, b = rng.sample(names, 2)
        dctx.append(tok.encode(f"{a} is a girl.{b} is a boy. The girl is named").ids); dans.append(nm[a])
        cctx.append(tok.encode(f"{a} is a boy.{b} is a girl. The girl is named").ids); cans.append(nm[b])
    det, caus = _detcaus(dec, fill, dctx, dans, cctx, cans)
    return _recovery_only([{"label": "coref", "applicable": n, "detect": det, "causal": caus,
                            "recovered": det >= TAU and caus >= TAU}], "semantic gender-binding — not a structural token rule")


def circuit_antonym(md, dec, fill, tok, vocab):
    aid = lambda w: tok.encode(w, add_special_tokens=False).ids
    pairs = [(a, b) for a, b in ANTONYMS if len(aid(a)) == 1 and len(aid(b)) == 1]
    if len(pairs) < 4:
        return None
    fwd = [tok.encode(f"The opposite of{a} is").ids for a, _ in pairs]
    rev = [tok.encode(f"The opposite of{b} is").ids for _, b in pairs]
    det, caus = _detcaus(dec, fill, fwd, [aid(b)[0] for _, b in pairs], rev, [aid(a)[0] for a, _ in pairs])
    return _recovery_only([{"label": "antonym", "applicable": len(pairs), "detect": det, "causal": caus,
                            "recovered": det >= TAU and caus >= TAU}], "semantic recall — antonym looked up from weights, not the tokens")


def circuit_capital(md, dec, fill, tok, vocab):
    aid = lambda w: tok.encode(w, add_special_tokens=False).ids
    pairs = [(c, cap) for c, cap in CAPITALS if len(aid(c)) == 1 and len(aid(cap)) == 1]
    if len(pairs) < 4:
        return None
    ctx = [tok.encode(f"The capital of{c} is").ids for c, _ in pairs]
    fill(ctx)
    det = sum(dec(ctx[i]) == aid(pairs[i][1])[0] for i in range(len(pairs))) / len(pairs)
    return _recovery_only([{"label": "capital", "applicable": len(pairs), "detect": det, "causal": det,     # cue-tracked recall
                            "recovered": det >= TAU}], "world-knowledge recall (cue-tracked) — not a structural rule")


def circuit_analogy(md, dec, fill, tok, vocab):
    aid = lambda w: tok.encode(w, add_special_tokens=False).ids
    p = {a: b for a, b in CAPITALS if len(aid(a)) == 1 and len(aid(b)) == 1}
    keys = list(p)
    if len(keys) < 4:
        return None
    rng = random.Random(12); dctx, dans, cctx, cans = [], [], [], []
    for _ in range(50):
        c1, c2, c3 = rng.sample(keys, 3)
        dctx.append(tok.encode(f"{c1} is to{p[c1]} as{c2} is to").ids); dans.append(aid(p[c2])[0])
        cctx.append(tok.encode(f"{c1} is to{p[c1]} as{c3} is to").ids); cans.append(aid(p[c3])[0])
    det, caus = _detcaus(dec, fill, dctx, dans, cctx, cans)
    return _recovery_only([{"label": "analogy", "applicable": 50, "detect": det, "causal": caus,
                            "recovered": det >= TAU and caus >= TAU}], "relation-mapping recall — not a structural rule")


def circuit_set(md, dec, fill, tok, vocab, n=50):
    sn, sN = _single(tok, NAMES), _single(tok, NOUNS)
    if len(sn) < 3 or len(sN) < 3:
        return None
    names, nouns = list(sn), list(sN)
    rng = random.Random(10); dctx, dans, cctx, cans = [], [], [], []
    for _ in range(n):
        x, y = rng.sample(names, 2); pp, q = rng.sample(nouns, 2)
        dctx.append(tok.encode(f"{x} has a{pp} and a{q}.{y} has a{pp}. The one with the{q} is").ids); dans.append(sn[x])
        cctx.append(tok.encode(f"{y} has a{pp} and a{q}.{x} has a{pp}. The one with the{q} is").ids); cans.append(sn[y])
    det, caus = _detcaus(dec, fill, dctx, dans, cctx, cans)
    return _recovery_only([{"label": "set", "applicable": n, "detect": det, "causal": caus,
                            "recovered": det >= TAU and caus >= TAU}], "set-membership binding — hardest; model-dependent (CROSS_ARCH)")


def circuit_defeasible(md, dec, fill, tok, vocab, n=50):
    yes, no = _single(tok, [" Yes"]), _single(tok, [" No"])
    s = _single(tok, NOUNS)
    if not yes or not no or len(s) < 2:
        return None
    yid, nid = yes[" Yes"], no[" No"]; nn = list(s)
    rng = random.Random(11); dctx, dans, cctx, cans = [], [], [], []
    for _ in range(n):
        x, y = rng.sample(nn, 2)
        dctx.append(tok.encode(f"Most{y} can swim. A{x} is a{y} that cannot swim. Can a{x} swim? Answer:").ids); dans.append(nid)
        cctx.append(tok.encode(f"Most{y} can swim. A{x} is a{y}. Can a{x} swim? Answer:").ids); cans.append(yid)
    det, caus = _detcaus(dec, fill, dctx, dans, cctx, cans)
    return _recovery_only([{"label": "defeas", "applicable": n, "detect": det, "causal": caus,
                            "recovered": det >= TAU and caus >= TAU}], "exception-override (Yes/No) — not a structural token rule")


def circuit_causal(md, dec, fill, tok, vocab, n=50):
    yes, no = _single(tok, [" Yes"]), _single(tok, [" No"])
    s = _single(tok, NOUNS)
    if not yes or not no or len(s) < 3:
        return None
    yid, nid = yes[" Yes"], no[" No"]; nn = list(s)
    rng = random.Random(13); ictx, ians, cctx, cans = [], [], [], []
    for _ in range(n):
        a, b, c = rng.sample(nn, 3)
        ictx.append(tok.encode(f"The{a} turns on the{b}. The{b} turns on the{c}. We unplug the{b}. The{a} is on. Is the{c} on? Answer:").ids); ians.append(nid)
        cctx.append(tok.encode(f"The{a} turns on the{b}. The{b} turns on the{c}. The{a} is on. Is the{c} on? Answer:").ids); cans.append(yid)
    interv, ctrl = _detcaus(dec, fill, ictx, ians, cctx, cans)          # do-asymmetry: intervention→No AND control→Yes
    return _recovery_only([{"label": "do/see", "applicable": n, "detect": ctrl, "causal": interv,
                            "recovered": interv >= TAU and ctrl >= TAU}], "do-calculus asymmetry (Yes/No) — not a structural token rule")


CIRCUITS = {
    # STRUCTURAL — admittable (holdout vs n-grams) + emittable
    "induction": dict(fn=circuit_induction, needs=(), emit="induction", known="copy / induction heads (pythia≥160m, llama/qwen ~1B)"),
    "succession": dict(fn=circuit_succession, needs=("tok",), emit="succession", known="ordinal succession over single-token letters"),
    "ioi": dict(fn=circuit_ioi, needs=("tok",), emit="once_appearing", known="name-mover / IOI copy (strong ~1B; weak small NeoX)"),
    "transitivity": dict(fn=circuit_transitivity, needs=("tok",), emit="once_appearing", known="is-a transitivity over stated premises"),
    "modus_ponens": dict(fn=circuit_mp, needs=("tok",), emit="once_appearing", known="modus ponens (stated conditional)"),
    "temporal": dict(fn=circuit_temporal, needs=("tok",), emit="once_appearing", known="temporal ordering transitivity"),
    "spatial": dict(fn=circuit_spatial, needs=("tok",), emit="once_appearing", known="spatial relational composition"),
    "syllogism": dict(fn=circuit_syllogism, needs=("tok",), emit="once_appearing", known="universal instantiation"),
    # SEMANTIC / RECALL — recovery-only (no structural rule → not emittable)
    "coreference": dict(fn=circuit_coreference, needs=("tok",), emit=None, known="gender-binding — never robustly confirmed (CROSS_ARCH)"),
    "antonym": dict(fn=circuit_antonym, needs=("tok",), emit=None, known="antonym recall (strong ~1B; not structural)"),
    "capital": dict(fn=circuit_capital, needs=("tok",), emit=None, known="capital-of world-knowledge recall"),
    "analogy": dict(fn=circuit_analogy, needs=("tok",), emit=None, known="analogy / structure-mapping recall"),
    "set": dict(fn=circuit_set, needs=("tok",), emit=None, known="set membership (∩) — hardest reasoning family"),
    "defeasible": dict(fn=circuit_defeasible, needs=("tok",), emit=None, known="defeasible exception override"),
    "causal": dict(fn=circuit_causal, needs=("tok",), emit=None, known="causal do-vs-see (intervention asymmetry)"),
}


def run(md, only=None, json_out=None):
    name = os.path.basename(md.rstrip("/"))
    vocab = vocab_of(md)
    dec, fill, label = make_oracle(md)
    tokpath = os.path.join(md, "bundle.tokenizer.json")
    tok = None
    if os.path.exists(tokpath):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokpath)
    print(f"=== exercise_confirm · {name} · oracle={label} · vocab={vocab} === (τ={TAU})\n")
    report = {"model": name, "oracle": label, "vocab": vocab, "tau": TAU, "circuits": {}}

    for cname, spec in CIRCUITS.items():
        if only and cname not in only:
            continue
        if "tok" in spec["needs"] and tok is None:
            print(f"[{cname}] skipped — needs bundle.tokenizer.json\n"); continue
        res = spec["fn"](md, dec, fill, tok, vocab)
        if res is None:
            print(f"[{cname}] skipped — stimuli not single-token in this tokenizer\n"); continue
        print(f"[{cname}]  ({spec['known']})")
        # recovery rows
        for row in res["rows"]:
            flag = "✓ recovered" if row["recovered"] else "· not recovered"
            nat = f"   natural causal {row['natural_causal']:>4.0%}  → gap {row['causal'] - row['natural_causal']:+.0%}" \
                  if res.get("has_natural") else ""
            print(f"  {row['label']:5} detect {row['detect']:>4.0%}  causal {row['causal']:>4.0%}   {flag}{nat}")
        # admission (structural circuits only; semantic/recall have no token rule → N/A)
        ho = res.get("holdout")
        if ho is None:
            print(f"  admission: N/A — {res.get('admission_na', 'recovery-only (no structural token rule)')}\n")
        else:
            H = max(1, ho["n_hold"])
            verdict = "ADMITS (generalizes past memorization)" if ho["admits"] else "does not beat n-grams on holdout"
            print(f"  holdout admission ({ho['n_hold']} unseen; {ho['ngram_rules']} n-gram rules on train):")
            print(f"    circuit rule : {ho['circuit_match']:>4}/{H} = {ho['circuit_match']/H:>4.0%} match model")
            print(f"    n-gram cover : {ho['ngram_match']:>4}/{H} = {ho['ngram_match']/H:>4.0%} match model")
            print(f"    ⇒ Δ {ho['delta']:+d} ({ho['delta']/H:+.0%}) — {verdict}\n")
        report["circuits"][cname] = res

    if json_out:
        json.dump(report, open(json_out, "w"), indent=2)
        print(f"wrote scorecard → {json_out}")
    return report


# ====================================================================================================================
# WIRE ALL ADMITTED CIRCUITS INTO THE COVER (souffle). The 8 admittable families reduce to THREE souffle rule-types:
# copy(induction), once-appearing(IOI + the reasoning families), ordinal(succession). We emit them as OOD fallbacks
# BELOW the natural-corpus n-gram cover (so on natural text the n-gram wins; on the exercising stimuli — novel content
# the cover can't memorize — the circuit fires), then CERTIFY the whole thing with equiv.dl over the stated domain
# (natural windows ∪ the stimulus instances the model actually runs the circuit on). A clean certificate = the admitted
# circuits are genuinely wired in, not just measured.


def py_ind1(ctx):
    last = ctx[-1]
    js = [j for j in range(len(ctx) - 1) if ctx[j] == last]
    return ctx[max(js) + 1] if js and max(js) + 1 < len(ctx) else None


def py_once(entity_ids):
    def r(ctx):
        pres = [t for t in ctx if t in entity_ids]
        once = [t for t, c in Counter(pres).items() if c == 1]
        return once[0] if len(once) == 1 else None
    return r


def py_succ(lord, lat):
    def r(ctx):
        present = {lord[t] for t in ctx if t in lord}
        if not present:
            return None
        o = max(present)
        return lat.get(o + 1) if (o - 1) in present and (o - 2) in present and (o + 1) in lat else None
    return r


def _emit_full_souffle(ngram_rules, entity_ids, lord, lat):
    from collections import defaultdict
    L = ["// rosetta · circuits.full.dl — natural-corpus n-gram cover + admitted STRUCTURAL circuits as souffle rules:",
         "//   copy/induction · once-appearing/name-mover (IOI + reasoning) · ordinal succession.",
         "// Routing: longest n-gram wins; else the circuits fire as OOD fallbacks (novel content the cover can't memorize).",
         "// Runtime: souffle only. tok(inst,pos,id) is provided by the includer (run.dl / equiv.dl).", "",
         ".decl mp(inst:number,m:number)", "mp(I,M) :- tok(I,_,_), M = max P : { tok(I,P,_) }.",
         ".decl cdecide(inst:number,out:number)", ""]
    bylen = defaultdict(dict)
    for suf, o in ngram_rules.items():
        bylen[len(suf)][suf] = o
    lens = sorted(bylen)
    for n in lens:                                                     # n-gram cover (token ids)
        N = n + 1
        cols = ",".join(f"c{i}:number" for i in range(n))
        L += [f".decl gram{N}({cols},t:number)", f".decl gram{N}_hit(inst:number,t:number)", f".decl gram{N}_any(inst:number)"]
        L += [f"gram{N}({','.join(map(str, suf))},{o})." for suf, o in bylen[n].items()]
        atoms = ["mp(I,P)"] + [f"tok(I,{'P' if i == n - 1 else f'P{n-1-i}'},C{i})" for i in range(n)]
        atoms += [f"P{k}=P-{k}" for k in range(1, n)] + [f"gram{N}({','.join(f'C{i}' for i in range(n))},T)"]
        L += [f"gram{N}_hit(I,T) :- {', '.join(atoms)}.", f"gram{N}_any(I) :- gram{N}_hit(I,_)."]
    L.append("")
    L += [".decl ind1_pj(inst:number,j:number)", "ind1_pj(I,J) :- mp(I,P), tok(I,P,X), tok(I,J,X), J<P.",
          ".decl ind1_last(inst:number,j:number)", "ind1_last(I,J) :- ind1_pj(I,_), J = max JJ : { ind1_pj(I,JJ) }.",
          ".decl ind1(inst:number,out:number)", "ind1(I,OUT) :- ind1_last(I,J), tok(I,J+1,OUT).",
          ".decl ind1_any(inst:number)", "ind1_any(I) :- ind1(I,_).", ""]
    if entity_ids:                                                    # once-appearing entity (name-mover / stated consequent)
        L += [".decl entity(id:number)"] + [f"entity({e})." for e in sorted(entity_ids)]
        L += [".decl ent_present(inst:number,id:number)", "ent_present(I,X) :- tok(I,_,X), entity(X).",
              ".decl ent_count(inst:number,id:number,n:number)", "ent_count(I,X,N) :- ent_present(I,X), N = count : { tok(I,P,X) }.",
              ".decl once_ent(inst:number,id:number)", "once_ent(I,X) :- ent_count(I,X,1).",
              ".decl once_ct(inst:number,n:number)", "once_ct(I,N) :- mp(I,_), N = count : { once_ent(I,X) }.",
              ".decl once_app(inst:number,out:number)", "once_app(I,OUT) :- once_ent(I,OUT), once_ct(I,1).",
              ".decl once_app_any(inst:number)", "once_app_any(I) :- once_app(I,_).", ""]
    if lord:                                                          # ordinal succession
        L += [".decl lord(id:number,ordv:number)"] + [f"lord({i},{o})." for i, o in sorted(lord.items())]
        L += [".decl lat(ordv:number,id:number)"] + [f"lat({o},{i})." for o, i in sorted(lat.items())]
        L += [".decl lpres(inst:number,ordv:number)", "lpres(I,O) :- tok(I,_,X), lord(X,O).",
              ".decl lmax(inst:number,ordv:number)", "lmax(I,O) :- lpres(I,_), O = max OO : { lpres(I,OO) }.",
              ".decl succ(inst:number,out:number)",
              "succ(I,OUT) :- lmax(I,O), Om1=O-1, lpres(I,Om1), Om2=O-2, lpres(I,Om2), Op1=O+1, lat(Op1,OUT).",
              ".decl succ_any(inst:number)", "succ_any(I) :- succ(I,_).", ""]
    # routing: longest n-gram > once-appearing > succession > induction > abstain. Succession is ABOVE induction because
    # a comma-separated run ends in a repeated punctuation token, on which the copy head would fire spuriously.
    L.append("// routing: longest n-gram > once-appearing > succession > induction > abstain")
    for n in lens:
        guard = "".join(f", !gram{m+1}_any(I)" for m in lens if m > n)
        L.append(f"cdecide(I,T) :- gram{n+1}_hit(I,T){guard}.")
    nog = "".join(f", !gram{m+1}_any(I)" for m in lens)
    if entity_ids:
        L.append(f"cdecide(I,T) :- once_app(I,T){nog}.")
    og = nog + (", !once_app_any(I)" if entity_ids else "")
    if lord:
        L.append(f"cdecide(I,T) :- succ(I,T){og}.")
    sg = og + (", !succ_any(I)" if lord else "")
    L.append(f"cdecide(I,T) :- ind1(I,T){sg}.")
    return "\n".join(L) + "\n", lens


def _emit_full_symbols(ngram_rules, sym, admitted):
    """The legible twin: n-gram rules with token STRINGS; the circuits are computations (not lookups) so they are
    described, and carried in circuits.full.dl (same as the composed-circuit note in minimize.emit_symbols)."""
    from collections import defaultdict
    esc = lambda s: '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t") + '"'
    q = lambda t: esc(sym.get(t, f"id{t}"))
    bylen = defaultdict(dict)
    for suf, o in ngram_rules.items():
        bylen[len(suf)][suf] = o
    lens = sorted(bylen)
    L = ["// rosetta · circuits.full.symbols.dl — legible twin (token STRINGS) of the n-gram cover; inherits the certificate.",
         f"// NOTE: this model also carries STRUCTURAL circuits ({', '.join(admitted)}) — computations over token identity/",
         "// order/ordinal, not token lookups, so they are not representable as symbol facts; see circuits.full.dl.", "",
         ".decl tok(inst:number, pos:number, sym:symbol)", ".input tok",
         ".decl mp(inst:number, m:number)", "mp(I,M) :- tok(I,_,_), M = max P : { tok(I,P,_) }.",
         ".decl cdecide(inst:number, out:symbol)", ".output cdecide"]
    for n in lens:
        N = n + 1
        cols = ",".join(f"c{i}:symbol" for i in range(n))
        L += [f".decl gram{N}({cols},t:symbol)", f".decl gram{N}_hit(inst:number,t:symbol)", f".decl gram{N}_any(inst:number)"]
        L += [f"gram{N}({','.join(q(t) for t in suf)},{q(o)})." for suf, o in bylen[n].items()]
        atoms = ["mp(I,P)"] + [f"tok(I,{'P' if i == n - 1 else f'P{n-1-i}'},C{i})" for i in range(n)]
        atoms += [f"P{k}=P-{k}" for k in range(1, n)] + [f"gram{N}({','.join(f'C{i}' for i in range(n))},T)"]
        L += [f"gram{N}_hit(I,T) :- {', '.join(atoms)}.", f"gram{N}_any(I) :- gram{N}_hit(I,_)."]
    for n in lens:
        guard = "".join(f", !gram{m+1}_any(I)" for m in lens if m > n)
        L.append(f"cdecide(I,T) :- gram{n+1}_hit(I,T){guard}.")
    return "\n".join(L) + "\n"


def emit_full_cover(md, dec, fill, tok, vocab, nat_n=300, nat_w=8):
    """Wire ALL admitted circuits into circuits.full.dl (+ symbols twin), certify over natural ∪ circuit-behavior stimuli."""
    from oracle import run_equiv
    name = os.path.basename(md.rstrip("/"))
    # 1. natural-corpus n-gram cover (the memoization backstop the circuits sit above)
    nat = instances(json.load(open(os.path.join(md, "corpus.json")))["ids"], nat_n, nat_w)
    fill(nat)
    nat_refs = [dec(c) for c in nat]
    nat_ok = [i for i in range(len(nat)) if nat_refs[i] is not None]
    ng = minimal_suffix_cover(nat, nat_refs, nat_ok, nat_w)[0]

    def ng_pred(ctx):
        for k in range(min(len(ctx), nat_w), 0, -1):
            o = ng.get(tuple(ctx[-k:]))
            if o is not None:
                return o
        return None

    # 2. facts for the structural rules
    entity_ids = set()
    for w in NAMES + NOUNS + ADJ:
        ids = tok.encode(w, add_special_tokens=False).ids
        if len(ids) == 1:
            entity_ids.add(ids[0])
    lord, lat = {}, {}
    letters = [c for c in LETTERS if len(tok.encode(c, add_special_tokens=False).ids) == 1]
    for o, c in enumerate(letters):
        i = tok.encode(c, add_special_tokens=False).ids[0]
        lord[i] = o; lat[o] = i

    # 3. gather each admittable circuit's stimuli; keep the instances the COVER (routing-aware, not the isolated rule)
    #    decides correctly AND the n-gram cover abstains on (novel content) — the stated domain the circuits certify.
    once_r, succ_r = py_once(entity_ids), py_succ(lord, lat)

    def cover_decide(ctx):                                              # mirror the souffle routing exactly
        p = ng_pred(ctx)
        if p is not None:
            return p
        p = once_r(ctx)
        if p is not None:
            return p
        p = succ_r(ctx)
        if p is not None:
            return p
        return py_ind1(ctx)

    dom_ins, dom_ref, per = list(nat), list(nat_refs), {}
    ind_ins, _s = induction_exercise(vocab, n_seqs=12, seqlen=16)       # induction stimuli inline (skip the costly natural re-measure)
    fill(ind_ins)
    emit_stim = {"induction": ("induction", ind_ins, [dec(c) for c in ind_ins])}
    for cname, spec in CIRCUITS.items():
        if not spec["emit"] or cname == "induction":
            continue
        res = spec["fn"](md, dec, fill, tok, vocab)
        if res and res.get("stimuli"):
            emit_stim[cname] = (spec["emit"], res["stimuli"][0], res["stimuli"][1])
    for cname, (kind, ins, refs) in emit_stim.items():
        keep = [k for k in range(len(ins)) if refs[k] is not None and ng_pred(ins[k]) is None and cover_decide(ins[k]) == refs[k]]
        for k in keep:
            dom_ins.append(ins[k]); dom_ref.append(refs[k])
        per[cname] = {"emit": kind, "stimuli": len(ins), "certified_instances": len(keep)}

    admitted_kinds = sorted({per[c]["emit"] for c in per if per[c].get("certified_instances", 0) > 0})
    # 4. emit + certify
    dl, _lens = _emit_full_souffle(ng, entity_ids if "once_appearing" in admitted_kinds else set(),
                                   lord if "succession" in admitted_kinds else {},
                                   lat if "succession" in admitted_kinds else {})
    out = os.path.join(md, "circuits.full.dl")
    open(out, "w").write(dl)
    sym = {}
    for t in {x for c in ng for x in c} | set(ng.values()):
        s = tok.id_to_token(t) if hasattr(tok, "id_to_token") else None
        sym[t] = (s or tok.decode([t]) or f"id{t}")
    open(os.path.join(md, "circuits.full.symbols.dl"), "w").write(_emit_full_symbols(ng, sym, admitted_kinds or ["none"]))
    r = run_equiv(out, dom_ins, dom_ref)
    certified = r.get("nmiss", 1) == 0 and r.get("nuncov", 1) == 0 and r.get("ncover", 0) == len([x for x in dom_ref if x is not None])
    score = {"model": name, "n_ngram_rules": len(ng), "natural_instances": len(nat_ok),
             "circuit_instances": len(dom_ins) - len(nat), "domain": len([x for x in dom_ref if x is not None]),
             "ncover": r.get("ncover"), "nmiss": r.get("nmiss"), "nuncov": r.get("nuncov"),
             "certified": certified, "per_circuit": per, "emitted_kinds": admitted_kinds}
    json.dump(score, open(os.path.join(md, "circuits.full.CERT.json"), "w"), indent=2)
    print(f"\n=== emit_full_cover · {name} ===")
    print(f"  n-gram cover (natural): {len(ng)} rules over {len(nat_ok)} windows")
    for c, d in per.items():
        if d.get("certified_instances", 0) or d.get("admitted"):
            print(f"  + {c:13} [{d.get('emit','-')}]: {d.get('certified_instances',0)} certified circuit instances "
                  f"(admits={d.get('admitted')})")
    print(f"  emitted structural kinds: {admitted_kinds or '(none)'}")
    print(f"  CERTIFY (equiv.dl) over natural ∪ circuit stimuli ({score['domain']} inst): "
          f"ncover={r.get('ncover')} nmiss={r.get('nmiss')} nuncov={r.get('nuncov')} → "
          + ("CERTIFIED — the wired circuits equal the model over the stated domain" if certified else "NOT certified"))
    print("  wrote circuits.full.dl + circuits.full.symbols.dl + circuits.full.CERT.json")
    return score


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    md = args[0] if args else "models/pythia160m"
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    only = next((set(a.split("=", 1)[1].split(",")) for a in sys.argv if a.startswith("--only=")), None)
    json_out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--json=")), None)
    if "--emit-cover" in sys.argv:
        dec, fill, _label = make_oracle(md)
        tokpath = os.path.join(md, "bundle.tokenizer.json")
        from tokenizers import Tokenizer
        emit_full_cover(md, dec, fill, Tokenizer.from_file(tokpath), vocab_of(md))
        return
    run(md, only, json_out)


if __name__ == "__main__":
    main()
