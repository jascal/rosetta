#!/usr/bin/env python3
"""rosetta · probe_families.py — probe-driven symbolic-family detection (the semantic tier of the toolkit).

Some families (copy/name-mover, coreference, succession, the ergo reasoning families) are entangled with content in a
natural corpus, so we ISOLATE them with templated/nonce stimuli + a foil, the way fieldrun's IOI work and ergo's probes
do. Toolkit-inclusion rule (per user): a family is IN the toolkit if we can DETECT it (the model follows the rule) AND
show it's CAUSAL (perturb the operand → the output follows) — whether or not any given model/dataset strictly needs it.
(Per-model COVER admission is the separate holdout+MDL question.)

Uses the model's tokenizer (rosetta .venv) + a resident `fieldrun --serve` server (FIELDRUN_SERVE).
Usage: FIELDRUN_SERVE=<port> python3 py/probe_families.py <model_dir> [n]
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import serve_decide, serve_topk
from tokenizers import Tokenizer


def in_topk(port, ctx, target, k=5):
    """set-based detect for NON-UNIQUE families (antonym: hot→cold/cool; relation): is the answer in the model's top-K,
    and does it shift causally with the cue? Fairer than argmax-exact when several answers are valid."""
    return target in [t for t, _ in serve_topk(port, list(ctx), k)]

NAMES = [" John", " Mary", " Tom", " Sara", " Paul", " Anna", " Mark", " Lucy", " Mike", " Emma", " David", " Kate",
         " James", " Laura", " Peter", " Alice", " Henry", " Julia", " Robert", " Nancy"]
DAYS = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"]
MONTHS = [" January", " February", " March", " April", " May", " June", " July", " August", " September", " October"]


def single(tok, words):
    """keep only words that are a single CONTENT token (no special/BOS), so the next-token prediction is the whole word."""
    out = {}
    for wd in words:
        ids = tok.encode(wd, add_special_tokens=False).ids
        if len(ids) == 1:
            out[wd] = ids[0]
    return out


def family_ioi(tok, port, n=40):
    """copy / name-mover (IOI): 'When A and B went…, B gave a drink to' → A (the once-mentioned name). Causal: swap the
    repeated name → the answer follows to the other name (the structure, not a fixed position, drives the copy)."""
    nm = single(tok, NAMES)
    names = list(nm)
    if len(names) < 4:
        return None
    rng = random.Random(0)
    det = cfollow = trials = 0
    for _ in range(n):
        a, b = rng.sample(names, 2)
        o1 = serve_decide(port, tok.encode(f"When{a} and{b} went to the store,{b} gave a drink to").ids)
        o2 = serve_decide(port, tok.encode(f"When{b} and{a} went to the store,{a} gave a drink to").ids)
        det += (o1 == nm[a]); cfollow += (o2 == nm[b]); trials += 1
    return det / trials, cfollow / trials


def family_succession(tok, port, seq, label, n=30):
    """succession / greater-than: an ordered run 'X Y Z' → the next item. Causal: shift the window's start → the
    predicted successor shifts with it (the model tracks ordinal position, not a memorized token). FORMAT-ROBUST: models
    differ in list-format prior (a code model reads bare-space ' Mon Tue Wed' as a token list → predicts a number/comma,
    but reads 'Mon, Tue, Wed,' as a sequence → the successor), so try both joins and report the best — the CIRCUIT is what
    we test, not one surface format; a 0% under one format is a probe artifact, not an architectural absence."""
    s = single(tok, seq)
    items = [w for w in seq if w in s]
    if len(items) < 5:
        return None
    fmts = [lambda a, b, c: f"{a}{b}{c}", lambda a, b, c: f"{a},{b},{c},"]   # bare-space vs comma-separated
    best = (0.0, 0.0)
    for fmt in fmts:
        rng = random.Random(1)
        det = cfollow = trials = 0
        for _ in range(n):
            i = rng.randint(0, len(items) - 4)
            det += (serve_decide(port, tok.encode(fmt(items[i], items[i + 1], items[i + 2])).ids) == s[items[i + 3]])
            j = (i + 1) % (len(items) - 3)                          # shift the window → successor must shift
            cfollow += (serve_decide(port, tok.encode(fmt(items[j], items[j + 1], items[j + 2])).ids) == s[items[j + 3]])
            trials += 1
        best = max(best, (det / trials, cfollow / trials), key=lambda t: t[0])
    return best


CAPITALS = [(" France", " Paris"), (" Germany", " Berlin"), (" Japan", " Tokyo"), (" Italy", " Rome"),
            (" Spain", " Madrid"), (" Russia", " Moscow"), (" China", " Beijing"), (" Egypt", " Cairo"),
            (" Canada", " Ottawa"), (" Greece", " Athens")]
ANTONYMS = [(" hot", " cold"), (" big", " small"), (" up", " down"), (" fast", " slow"), (" happy", " sad"),
            (" open", " closed"), (" black", " white"), (" good", " bad"), (" rich", " poor"), (" hard", " soft"),
            (" light", " dark"), (" high", " low"), (" true", " false"), (" old", " young"), (" wet", " dry")]


def family_coreference(tok, port, n=40):
    """coreference: '{A} is a girl. {B} is a boy. The girl is named' → A. Causal: swap the genders → the answer follows
    the role, not the position (the binding tracks the referent, not a slot)."""
    nm = single(tok, NAMES)
    names = list(nm)
    if len(names) < 4:
        return None
    rng = random.Random(2)
    det = cfollow = trials = 0
    for _ in range(n):
        a, b = rng.sample(names, 2)
        det += in_topk(port, tok.encode(f"{a} is a girl.{b} is a boy. The girl is named").ids, nm[a])
        cfollow += in_topk(port, tok.encode(f"{a} is a boy.{b} is a girl. The girl is named").ids, nm[b])
        trials += 1
    return det / trials, cfollow / trials


def family_pairs(tok, port, pairs, template, n=40):
    """relational lookup (analogy / antonym): one-shot relation by example or a named relation → the related token.
    Causal: change the cue → the related token follows the RELATION (not a fixed output)."""
    p = {a: b for a, b in pairs if a in single(tok, [a]) and b in single(tok, [b])}
    if len(p) < 4:
        return None
    sb = single(tok, [b for _, b in pairs])
    keys = list(p)
    rng = random.Random(3)
    det = cfollow = trials = 0
    for _ in range(n):
        a = rng.choice(keys)
        det += in_topk(port, tok.encode(template(a)).ids, sb[p[a]])
        a2 = rng.choice([k for k in keys if k != a])
        cfollow += in_topk(port, tok.encode(template(a2)).ids, sb[p[a2]]); trials += 1
    return det / trials, cfollow / trials


NOUNS = [" dog", " cat", " bird", " fish", " tree", " rock", " star", " car", " boat", " house",
         " king", " wolf", " lion", " bear", " duck", " frog", " horse", " sheep", " goat", " mouse"]
ADJ = [" red", " big", " hot", " soft", " loud", " green", " tall", " round", " sharp", " heavy",
       " blue", " small", " cold", " hard", " quiet", " short", " flat", " light", " smooth", " warm"]


def family_transitive(tok, port, n=40):
    """is-a / inheritance transitivity over STATED (arbitrary) facts: 'A X is a Y. A Y is a Z. So a X is a' → Z. Nonce
    chains so world knowledge can't shortcut it; causal: change the 2nd premise's Z → the conclusion follows (it's
    reasoning over the stated chain, not recalling)."""
    s = single(tok, NOUNS)
    nn = list(s)
    if len(nn) < 6:
        return None
    rng = random.Random(4)
    det = cfollow = trials = 0
    for _ in range(n):
        x, y, z, z2 = rng.sample(nn, 4)
        det += in_topk(port, tok.encode(f"A{x} is a{y}. A{y} is a{z}. So a{x} is a").ids, s[z])
        cfollow += in_topk(port, tok.encode(f"A{x} is a{y}. A{y} is a{z2}. So a{x} is a").ids, s[z2])
        trials += 1
    return det / trials, cfollow / trials


def family_mp(tok, port, n=40):
    """modus ponens: 'If something is A, it is B. This is A. So this is' → B. Causal: change the consequent B → the
    conclusion follows (the model applies the stated conditional, given the antecedent holds)."""
    s = single(tok, ADJ)
    aa = list(s)
    if len(aa) < 4:
        return None
    rng = random.Random(5)
    det = cfollow = trials = 0
    for _ in range(n):
        a, b, b2 = rng.sample(aa, 3)
        det += in_topk(port, tok.encode(f"If something is{a}, it is{b}. This is{a}. So this is").ids, s[b])
        cfollow += in_topk(port, tok.encode(f"If something is{a}, it is{b2}. This is{a}. So this is").ids, s[b2])
        trials += 1
    return det / trials, cfollow / trials


def family_temporal(tok, port, n=40):
    """temporal/ordering transitivity: 'X before Y. Y before Z. So X before' → Z. Causal: change Z → conclusion follows."""
    s = single(tok, NOUNS)
    nn = list(s)
    if len(nn) < 6:
        return None
    rng = random.Random(6)
    det = cfollow = trials = 0
    for _ in range(n):
        x, y, z, z2 = rng.sample(nn, 4)
        det += in_topk(port, tok.encode(f"{x} comes before{y}.{y} comes before{z}. So{x} comes before").ids, s[z])
        cfollow += in_topk(port, tok.encode(f"{x} comes before{y}.{y} comes before{z2}. So{x} comes before").ids, s[z2])
        trials += 1
    return det / trials, cfollow / trials


def family_syllogism(tok, port, n=40):
    """universal instantiation: 'Every Y is Z. name is a Y. So name is' → Z. Causal: change Z → conclusion follows."""
    sn, sa, sN = single(tok, NAMES), single(tok, ADJ), single(tok, NOUNS)
    if len(sn) < 3 or len(sa) < 3 or len(sN) < 2:
        return None
    names, adjs, nouns = list(sn), list(sa), list(sN)
    rng = random.Random(7)
    det = cf = tr = 0
    for _ in range(n):
        x, y, (z, z2) = rng.choice(names), rng.choice(nouns), rng.sample(adjs, 2)
        det += in_topk(port, tok.encode(f"Every{y} is{z}.{x} is a{y}. So{x} is").ids, sa[z])
        cf += in_topk(port, tok.encode(f"Every{y} is{z2}.{x} is a{y}. So{x} is").ids, sa[z2]); tr += 1
    return det / tr, cf / tr


def family_spatial(tok, port, n=40):
    """spatial relational composition: 'X left of Y. Y left of Z. So X left of' → Z. Causal: change Z."""
    s = single(tok, NOUNS)
    nn = list(s)
    if len(nn) < 6:
        return None
    rng = random.Random(8)
    det = cf = tr = 0
    for _ in range(n):
        x, y, z, z2 = rng.sample(nn, 4)
        det += in_topk(port, tok.encode(f"The{x} is left of the{y}. The{y} is left of the{z}. So the{x} is left of the").ids, s[z])
        cf += in_topk(port, tok.encode(f"The{x} is left of the{y}. The{y} is left of the{z2}. So the{x} is left of the").ids, s[z2]); tr += 1
    return det / tr, cf / tr


def family_set(tok, port, n=40):
    """set membership / intersection: 'A has a P and a Q. B has a P. The one with the Q is' → A. Causal: swap who has Q."""
    sn, sN = single(tok, NAMES), single(tok, NOUNS)
    if len(sn) < 3 or len(sN) < 3:
        return None
    names, nouns = list(sn), list(sN)
    rng = random.Random(10)
    det = cf = tr = 0
    for _ in range(n):
        x, y = rng.sample(names, 2)
        p, q = rng.sample(nouns, 2)
        det += in_topk(port, tok.encode(f"{x} has a{p} and a{q}.{y} has a{p}. The one with the{q} is").ids, sn[x])
        cf += in_topk(port, tok.encode(f"{y} has a{p} and a{q}.{x} has a{p}. The one with the{q} is").ids, sn[y]); tr += 1
    return det / tr, cf / tr


def family_defeasible(tok, port, n=40):
    """defeasible / exception override: 'Most Y can V. An X is a Y that cannot V. Can an X V? Answer:' → No (exception
    beats the default). Causal: drop the exception → Yes (default applies)."""
    yes, no = single(tok, [" Yes"]), single(tok, [" No"])
    s = single(tok, NOUNS)
    if not yes or not no or len(s) < 4:
        return None
    yid, nid = yes[" Yes"], no[" No"]
    nn = list(s)
    rng = random.Random(11)
    det = cf = tr = 0
    for _ in range(n):
        x, y = rng.sample(nn, 2)
        det += in_topk(port, tok.encode(f"Most{y} can swim. A{x} is a{y} that cannot swim. Can a{x} swim? Answer:").ids, nid)
        cf += in_topk(port, tok.encode(f"Most{y} can swim. A{x} is a{y}. Can a{x} swim? Answer:").ids, yid); tr += 1
    return det / tr, cf / tr


def family_analogy(tok, port, n=40):
    """analogy / structure-mapping: 'c1 is to cap1 as c2 is to' → cap2 (the relation is INFERRED from the first pair,
    not named). Causal: change c2 → the mapped target follows the relation."""
    p = {a: b for a, b in CAPITALS if a in single(tok, [a]) and b in single(tok, [b])}
    sb = single(tok, [b for _, b in CAPITALS])
    keys = list(p)
    if len(keys) < 4:
        return None
    rng = random.Random(12)
    det = cf = tr = 0
    for _ in range(n):
        c1, c2, c3 = rng.sample(keys, 3)
        det += in_topk(port, tok.encode(f"{c1} is to{p[c1]} as{c2} is to").ids, sb[p[c2]])
        cf += in_topk(port, tok.encode(f"{c1} is to{p[c1]} as{c3} is to").ids, sb[p[c3]]); tr += 1
    return det / tr, cf / tr


def family_causal(tok, port, n=40):
    """causal do-vs-see (intervention): 'A turns on B. B turns on C. We unplug B. Is C on?' → No (intervention breaks the
    chain), vs the observational control 'A is on. Is C on?' → Yes. The family is real iff the model distinguishes do
    from see — both the intervention(No) AND the control(Yes) must hold (the do-calculus asymmetry)."""
    yes, no = single(tok, [" Yes"]), single(tok, [" No"])
    s = single(tok, NOUNS)
    if not yes or not no or len(s) < 4:
        return None
    yid, nid = yes[" Yes"], no[" No"]
    nn = list(s)
    rng = random.Random(13)
    interv = ctrl = tr = 0
    for _ in range(n):
        a, b, c = rng.sample(nn, 3)
        interv += in_topk(port, tok.encode(f"The{a} turns on the{b}. The{b} turns on the{c}. We unplug the{b}. The{a} is on. Is the{c} on? Answer:").ids, nid)
        ctrl += in_topk(port, tok.encode(f"The{a} turns on the{b}. The{b} turns on the{c}. The{a} is on. Is the{c} on? Answer:").ids, yid); tr += 1
    return interv / tr, ctrl / tr


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else "models/llama32_1b"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    port = int(os.environ["FIELDRUN_SERVE"])
    tok = Tokenizer.from_file(os.path.join(md, "bundle.tokenizer.json"))
    name = os.path.basename(md.rstrip("/"))
    print(f"=== probe_families · {name} === (detect ≥80% AND causal ≥80% ⇒ in the toolkit)")
    probes = [("copy/name-mover (IOI)", lambda: family_ioi(tok, port, n)),
              ("succession (days)", lambda: family_succession(tok, port, DAYS, "days", n)),
              ("succession (months)", lambda: family_succession(tok, port, MONTHS, "months", n)),
              ("coreference (gender)", lambda: family_coreference(tok, port, n)),
              ("relation (capital-of)", lambda: family_pairs(tok, port, CAPITALS, lambda a: f"The capital of{a} is", n)),
              ("antonym (opposite-of)", lambda: family_pairs(tok, port, ANTONYMS, lambda a: f"The opposite of{a} is", n)),
              ("is-a transitivity", lambda: family_transitive(tok, port, n)),
              ("modus ponens", lambda: family_mp(tok, port, n)),
              ("temporal ordering", lambda: family_temporal(tok, port, n)),
              ("syllogism (instantiation)", lambda: family_syllogism(tok, port, n)),
              ("spatial (left-of)", lambda: family_spatial(tok, port, n)),
              ("set membership (∩)", lambda: family_set(tok, port, n)),
              ("defeasible (exception)", lambda: family_defeasible(tok, port, n)),
              ("analogy (a:b::c:?)", lambda: family_analogy(tok, port, n)),
              ("causal do-vs-see", lambda: family_causal(tok, port, n))]
    for label, fn in probes:
        r = fn()
        if r is None:
            print(f"  {label:28} — skipped (stimuli not single-token in this tokenizer)")
            continue
        det, caus = r
        verdict = "  ← IN TOOLKIT" if det >= 0.8 and caus >= 0.8 else ("  (causal but weak detect)" if caus >= 0.8 else "")
        print(f"  {label:28} detect {det:.0%}  causal {caus:.0%}{verdict}")


if __name__ == "__main__":
    main()
