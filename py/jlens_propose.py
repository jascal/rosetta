#!/usr/bin/env python3
"""rosetta · jlens_propose.py — the J-Lens proposer shim (UNTRUSTED generator; `empirical`).

Reads a J-Lens *prior* JSONL (produced on the pil side from a `fieldrun --source-dump` +
`--jlens-export`; see JLENS_BRIDGE.md) and turns it into a per-context **family / candidate ordering**
that `idiom_learn.select_cover` consults to spend its causal-confirmation budget on likely-right
candidates first. It NEVER removes a candidate and NEVER touches `dl/` — the certificate (`dl/equiv.dl`)
is unchanged. If the prior file is absent, malformed, or carries no signal (the expected null-by-scale
case on a sub-24-layer model), every entry point degrades to `None`/identity and the blind miner runs
exactly as today.

The prior gives DEPTH, not POSITION. J-Lens attributes the decode to a layer's attn/mlp *blocks*; it does
NOT say which context offset is the operand. Position localization stays with `discover.py` flip-counts.
The shim's contribution is orthogonal: which *family* (n-gram / select-gate / compose / induction /
skeleton) and how *deep* the computation is, so the enumerator tries the right shape first.

Prior JSONL schema (one record per corpus position — see producer in JLENS_BRIDGE.md):
    {"pos": int,                    # corpus position p: context = corpus_ids[p-w+1 : p+1] (from --source-dump)
     "decode": int,                 # model next-token argmax at this position (== cands[0])
     "resolve": float,              # fraction-of-depth where the cumulative J-corrected read first
                                    #   locks to `decode` (lower = earlier); the pil sweep's `resolve`
     "block_inc": [[label, inc]],   # J-corrected per-block incidence toward `decode`, sorted desc |inc|
     "n_layer": int}
"""
from __future__ import annotations

import json
import re
from collections import Counter

_BLOCK = re.compile(r"^L(\d+)\.(attn|mlp)$")   # matches pil.fieldrun_io._JLENS_BLOCK label form

# Family names MUST match idiom_learn.select_cover's FAM keys (+ "ngram" for the suffix cover).
FAMILIES = ("ngram", "select", "compose", "induction", "skeleton")


# ── load + align ────────────────────────────────────────────────────────────────────────────────────
def load_prior(path):
    """Read the prior JSONL in corpus order. Returns [] on any read/parse failure (→ blind fallback)."""
    try:
        with open(path) as fh:
            recs = [json.loads(ln) for ln in fh if ln.strip()]
    except (OSError, ValueError):
        return []
    return recs


def align_to_contexts(prior, corpus_ids, w, decide_fn=None, whole=None, check=32):
    """Map each length-`w` corpus window → its prior record, keyed by the window token tuple.

    Each `--source-dump` record carries its corpus position `pos` (context = `corpus_ids[pos-w+1 : pos+1]`);
    absent a `pos` field we fall back to enumerate order. Keying by the context tuple survives idiom_learn's
    window dedup (identical windows share one prior). If `decide_fn`/`whole` are given, ABORT LOUDLY (rosetta
    discipline — no vacuous success) when the prior's `decode` disagrees with the oracle on a sample, i.e. the
    alignment is off (wrong corpus, off-by-one, wrong bundle).

    Returns {ctx_tuple: prior_rec}. Empty dict ⇒ callers fall back to blind order.
    """
    if not prior:
        return {}
    out, checked, bad = {}, 0, 0
    for i, rec in enumerate(prior):
        p = int(rec.get("pos", i))                       # prefer the dump's explicit position
        if p < w - 1 or p >= len(corpus_ids):
            continue
        ctx = tuple(corpus_ids[p - w + 1: p + 1])
        out.setdefault(ctx, rec)
        if decide_fn is not None and whole is not None and checked < check:
            checked += 1
            if decide_fn(whole, list(ctx)) != rec.get("decode"):
                bad += 1
    if checked and bad:
        raise SystemExit(
            f"✗ ABORT: J-Lens prior misaligned — {bad}/{checked} decode mismatches vs the oracle. "
            f"The --source-dump is not positionally aligned to {len(corpus_ids)} corpus ids "
            f"(wrong corpus, off-by-one, or the dump needs a `pos` field — see JLENS_BRIDGE.md). "
            f"No vacuous success.")
    return out


# ── depth/family prior ────────────────────────────────────────────────────────────────────────────
def _summarize(rec, top_frac=0.5):
    """Collapse a prior record into (depth01, attn_frac, n_load, resolve01): the shape signal.

    depth01   normalized mean layer of the load-bearing blocks (blocks within `top_frac` of the top |inc|)
    attn_frac share of load-bearing |inc| carried by attn (vs mlp) blocks
    n_load    number of load-bearing blocks (1 ≈ single-operand lookup; ≥2 ≈ composed/distributed)
    resolve01 the sweep's fraction-of-depth resolve (lower = decode fixed earlier in the stack)
    """
    inc = rec.get("block_inc") or []
    n_layer = max(int(rec.get("n_layer", 1)), 1)
    if not inc:
        return None
    top = abs(inc[0][1]) or 1.0
    load = [(lbl, a) for lbl, a in inc if abs(a) >= top_frac * top]
    layers, attn_mass, mlp_mass = [], 0.0, 0.0
    for lbl, a in load:
        m = _BLOCK.match(lbl)
        if not m:                                    # 'embed' / unmapped → depth 0, counts as mlp-ish
            layers.append(0); mlp_mass += abs(a); continue
        layers.append(int(m.group(1)))
        if m.group(2) == "attn":
            attn_mass += abs(a)
        else:
            mlp_mass += abs(a)
    depth01 = (sum(layers) / len(layers)) / max(n_layer - 1, 1)
    tot = attn_mass + mlp_mass or 1.0
    return depth01, attn_mass / tot, len(load), float(rec.get("resolve", 1.0))


def family_prior(rec):
    """Ranked family names (best first) for one context — a SOFT ordering, never a filter.

    Heuristic mapping (itself `open` — the experiment measures whether it improves survival-per-search):
      shallow + early-resolve + one block          → n-gram / select-gate  (surface lookup)
      attn-dominated load-bearing block            → induction / copy       (attention = the copy path)
      ≥2 load-bearing blocks across depth, mlp-heavy, late resolve → compose (multi-operand computation)
    Returns the full FAMILIES tuple reordered; ties fall back to the current blind priority
    (compose > select > induction > skeleton > ngram) so nothing is ever dropped.
    """
    s = _summarize(rec)
    if s is None:
        return list(FAMILIES)
    depth01, attn_frac, n_load, resolve01 = s
    score = {f: 0.0 for f in FAMILIES}
    # shallow & early & concentrated → surface families
    if resolve01 < 0.4 and n_load <= 1 and depth01 < 0.4:
        score["ngram"] += 2.0; score["select"] += 1.5
    # attention-carried → content-relative copy
    if attn_frac > 0.6:
        score["induction"] += 2.0
    # deep, multi-block, mlp-heavy → composition (mlp-dominance required, else attn wins induction above)
    if n_load >= 2 and depth01 > 0.5 and attn_frac < 0.5:
        score["compose"] += 2.0
    if resolve01 > 0.6:
        score["compose"] += 0.5; score["skeleton"] += 0.5
    blind = {"compose": 4, "select": 3, "induction": 2, "skeleton": 1, "ngram": 0}   # tie-break = today's order
    return sorted(FAMILIES, key=lambda f: (-score[f], -blind[f]))


# ── corpus-level ordering + budget split (what select_cover consults) ─────────────────────────────
def family_order(ctx_prior, insts, idxs, min_discrim=0.9):
    """Aggregate per-context family priors over the working set → a corpus-level family admission order
    and a normalized budget weight per family. Returns (order, weight), or (None, None) → select_cover
    keeps its blind order + uniform budget, when either:
      * no prior covers the set, OR
      * the prior does not DISCRIMINATE — one family is top-choice for ≥ `min_discrim` of covered contexts
        (a flat/degenerate signal, e.g. a below-the-gate or under-fit J-Lens; the expected null case). A
        prior that says "everything is the same family" gives no ordering benefit over blind. `weight`
        sums to 1.0."""
    if not ctx_prior:
        return None, None
    votes, tops, n = Counter(), Counter(), 0
    for i in idxs:
        rec = ctx_prior.get(tuple(insts[i]))
        if rec is None or not rec.get("block_inc"):
            continue
        ranked = family_prior(rec)
        tops[ranked[0]] += 1; n += 1
        for rank, fam in enumerate(ranked):
            votes[fam] += len(FAMILIES) - rank                # Borda count
    if not votes or n == 0:
        return None, None
    if tops.most_common(1)[0][1] >= min_discrim * n:          # no discrimination → blind fallback
        return None, None
    order = sorted(FAMILIES, key=lambda f: -votes[f])
    tot = sum(votes.values()) or 1
    weight = {f: votes[f] / tot for f in FAMILIES}
    return order, weight


if __name__ == "__main__":                                   # smoke: summarize a prior file
    import sys
    recs = load_prior(sys.argv[1]) if len(sys.argv) > 1 else []
    print(f"{len(recs)} prior records")
    tally = Counter(family_prior(r)[0] for r in recs if r.get("block_inc"))
    for fam, n in tally.most_common():
        print(f"  top-prior {fam:10} {n:5}  ({n/max(len(recs),1):.0%})")
