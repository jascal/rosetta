#!/usr/bin/env python3
"""rosetta · serve_package.py — the REFERENCE THIN RUNTIME for a rosetta expert package, and the spec for sgiandubh's C++.

The convergence (rosetta = builder, sgiandubh = thin runtime): rosetta emits the package (circuits.abstain.dl + manifest.json,
see py/abstain_emit.py); a runtime consumes it. This reference consumer proves the package is servable **host-side with NO
souffle and NO model** — the manifest IS the compiled cover (a decision table). It does exactly what sgiandubh's runtime will:

    tokenize(query)  →  TRUSTED idioms (frame-match) → GATED n-grams (longest suffix) → {answer, tier, basis, citation}  or  ABSTAIN

The RUNTIME loads only manifest.json (load_package + serve) — no corpus, no souffle, no model. The corpus is used only by the
BUILDER (make_package, = py/abstain_emit) and by the held-out scorecard (eval), kept separate from serving.

KEY FINDING surfaced here (matters for sgiandubh): the cover is in the model's BPE-TOKEN space, so the runtime must tokenize
the query with the model's tokenizer (bundle.tokenizer.json). sgiandubh is currently word/text-based, so consuming the rosetta
cover requires a BPE tokenizer in its C++ runtime — the real Stage-2 dependency (this script is the port spec).

Deps: tokenizers (for the NL-query path). Usage: .venv/bin/python py/serve_package.py [model_dir] [W] [minsupp] [mindet]
"""
import json, os, sys, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abstain_emit import build_tab, confident_rules, emit, emit_manifest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_package(md, W, minsupp, mindet):
    """BUILDER side (= py/abstain_emit): build the cover from corpus.json, emit circuits.abstain.dl + manifest.json.
    Returns the held-out windows (for the eval scorecard only — NOT used by the runtime)."""
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    N = min(len(ids) - 1, 40000)
    wins = [(tuple(ids[i - W:i]), ids[i], i) for i in range(W, N)]
    random.Random(0).shuffle(wins)
    cut = int(len(wins) * 0.7); train, hold = wins[:cut], wins[cut:]
    tab, cites = build_tab(train, W)
    rules = confident_rules(tab, cites, W, minsupp, mindet)
    ridmap = emit(os.path.join(md, "circuits.abstain.dl"), rules, W, minsupp, mindet)
    meta_p = os.path.join(md, "corpus_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else None
    emit_manifest(os.path.join(md, "manifest.json"), ridmap, os.path.basename(md), W, minsupp, mindet, meta)
    return hold


def _cite(r):
    return r["citation"] if r.get("citation") else (f"corpus@{r['cite'][:3]}" if r.get("cite") else "")


def load_package(manifest_path):
    """THE THIN RUNTIME state from manifest.json ONLY (no corpus, no souffle, no model). Handles the TIERED package
    (idiom_learn --package: causal idioms + gated n-grams) and is backward-compatible with the flat n-gram manifest
    (abstain_emit: a rule with no 'kind' is an n-gram). Returns (idioms, ngrams, manifest):
      idioms = ordered TRUSTED rules (gate/compose), kept in priority order — all host-side matchable;
      ngrams[k][suffix] = (out, basis, cite, confidence, stratum, origin, counts) — the
      GATED tier (= 'fire only if confident', NOT gating inside the n-gram). counts is None for
      old packages.
    JSON keys are strings; normalized to ints here so a C++ porter sees the same shape."""
    m = json.load(open(manifest_path))
    idioms, ngrams = [], defaultdict(dict)
    m["_tau"] = float(m.get("strata_tau") or 0.35)
    m["_derived"] = [{"id": d["id"], "kind": d["kind"],
                      "openers": set(d.get("openers", [])),
                      "closers": set(d.get("closers", [])),
                      "members": set(d.get("members", [])),
                      "cap": int(d.get("cap", 8)), "succ": int(d.get("succ", 0)),
                      "of": d.get("of"), "of_shift": int(d.get("of_shift", 0)),
                      "avoid": set(d.get("avoid", [])), "look": int(d.get("look", 3)),
                      "entity_members": set(d.get("entity_members", [])),
                      "value_members": set(d.get("value_members", [])),
                      "within": int(d.get("within", 7)),
                      "slot": d.get("slot"), "mode": d.get("mode", "is"),
                      "entities": set(d.get("entities", [])),
                      "locations": set(d.get("locations", [])),
                      "objects": set(d.get("objects", [])),
                      "move_verbs": set(d.get("move_verbs", [])),
                      "take_verbs": set(d.get("take_verbs", [])),
                      "drop_verbs": set(d.get("drop_verbs", [])),
                      "quote_members": set(d.get("quote_members", []))}
                     for d in m.get("derived", [])]
    m["_cmap"] = {int(mm): int(rep) for rep, mem in m.get("concepts", {}).items()
                  for mm in mem}                             # member -> representative
    for r in m["rules"]:
        kind = r.get("kind", "ngram")
        if kind == "ngram":
            ctx = tuple(r["ctx"])
            ngrams[len(ctx)][ctx] = (r["out"], r.get("basis", "observational"), _cite(r),
                                     r.get("confidence"), int(r.get("stratum", 1)),
                                     r.get("origin", m.get("origin", "teacher")),
                                     tuple(int(x) for x in r["counts"])
                                     if r.get("counts") is not None else None)
        elif kind == "gate":
            idioms.append({"kind": "gate", "stratum": int(r.get("stratum", 1)), "origin": r.get("origin", m.get("origin", "teacher")), "id": r["id"], "cite": _cite(r),
                           "frame": {int(o): int(t) for o, t in r["frame"].items()}, "slot": r["slot"],
                           "table": {int(k): int(v) for k, v in r["table"].items()},
                           "confs": {int(k): float(c) for k, c in (r.get("confs") or {}).items()},
                           "counts": {int(k): tuple(int(x) for x in v)
                                      for k, v in (r.get("counts") or {}).items()}})
        elif kind == "compose":
            idioms.append({"kind": "compose", "stratum": int(r.get("stratum", 1)), "origin": r.get("origin", m.get("origin", "teacher")), "id": r["id"], "cite": _cite(r),
                           "frame": {int(o): int(t) for o, t in r["frame"].items()}, "operands": r["operands"],
                           "valmap": {int(t): int(v) for t, v in r["valmap"].items()},
                           "sum": {int(s): int(o) for s, o in r["sum"].items()}})
        elif kind == "induction":                                # causal COPY circuit, routed OOD (after n-grams)
            idioms.append({"kind": "induction", "stratum": int(r.get("stratum", 1)), "origin": r.get("origin", m.get("origin", "teacher")), "id": r["id"], "cite": _cite(r), "L": int(r["L"]),
                           "conf": r.get("confidence")})
        elif kind == "succession":                               # causal ORDINAL circuit, routed OOD (above induction)
            idioms.append({"kind": "succession", "stratum": int(r.get("stratum", 1)), "origin": r.get("origin", m.get("origin", "teacher")), "id": r["id"], "cite": _cite(r),
                           "lord": {int(t): int(o) for t, o in r["lord"].items()},
                           "lat": {int(o): int(t) for o, t in r["lat"].items()}})
        elif kind == "pointer":                                  # generalized copy: (l, lc)-cell scorer
            idioms.append({"kind": "pointer", "stratum": int(r.get("stratum", 1)), "origin": r.get("origin", m.get("origin", "teacher")), "id": r["id"], "cite": _cite(r),
                           "lmax": int(r.get("lmax", 6)),
                           "cells": {tuple(int(x) for x in k.split(":")): float(c)
                                     for k, c in r["cells"].items()}})
        elif kind == "dgate2":                                   # PAIR gate: two features jointly
            idioms.append({"kind": "dgate2", "stratum": int(r.get("stratum", 1)), "origin": r.get("origin", m.get("origin", "teacher")), "id": r["id"], "cite": _cite(r),
                           "fa": r["featureA"], "fb": r["featureB"],
                           "table": {tuple(int(x) for x in k.split(":")): int(v)
                                     for k, v in r["table"].items()},
                           "confs": {tuple(int(x) for x in k.split(":")): float(c)
                                     for k, c in (r.get("confs") or {}).items()},
                           "counts": {tuple(int(x) for x in k.split(":")):
                                      tuple(int(x) for x in v)
                                      for k, v in (r.get("counts") or {}).items()}})
        elif kind == "dgate":                                    # TWO-LAYER: gate over a DERIVED predicate
            idioms.append({"kind": "dgate", "stratum": int(r.get("stratum", 1)), "origin": r.get("origin", m.get("origin", "teacher")), "id": r["id"], "cite": _cite(r), "feature": r["feature"],
                           "table": {tuple(int(x) for x in k.split(":")): int(v)
                                     for k, v in r["table"].items()},
                           "confs": {tuple(int(x) for x in k.split(":")): float(c)
                                     for k, c in (r.get("confs") or {}).items()},
                           "counts": {tuple(int(x) for x in k.split(":")):
                                      tuple(int(x) for x in v)
                                      for k, v in (r.get("counts") or {}).items()}})
        elif kind == "relation":                                 # causal EQ-GUARD + COPY (offset-local), routed OOD
            idioms.append({"kind": "relation", "stratum": int(r.get("stratum", 1)), "origin": r.get("origin", m.get("origin", "teacher")), "id": r["id"], "cite": _cite(r),
                           "eq": [(int(i), int(j)) for i, j in r["eq"]], "copy": int(r["copy"]),
                           "conf": r.get("confidence")})
        elif kind == "khop":                                     # 2-hop circuit: q -> bridge -> pred, bridge-site excluded (mirrors pil mir_khop2/_DL_KHOP)
            idioms.append({"kind": "khop", "stratum": int(r.get("stratum", 1)), "origin": r.get("origin", m.get("origin", "teacher")), "id": r["id"], "cite": _cite(r),
                           "lo": int(r["lo"]), "hi": int(r["hi"]),
                           "conf": r.get("confidence")})
    return idioms, ngrams, m


def _khop_predict(ctx, lo, hi):
    """2-hop circuit: mirrors pil `mir_khop2`/`_DL_KHOP` EXACTLY (wyly_selfcompile.py L171-203).
    q = ctx[-1]; p1 = rightmost earlier match of q (comparison range excludes the query position
    itself: i in [0, W-2]); bridge = ctx[p1+1] (clamped to W-1); p2 = rightmost earlier match of
    bridge EXCLUDING position p1+1 itself (the BRIDGE-SITE EXCLUSION -- load-bearing: without it
    the bridge's own site could re-match itself and the second hop is wrong); returns
    ctx[p2+1] (clamped), or None if any stage fails to match."""
    W = len(ctx)
    if W < 2:
        return None
    q = ctx[-1]
    if not (lo <= q <= hi):
        return None
    cand1 = [i for i in range(W - 1) if ctx[i] == q]
    if not cand1:
        return None
    p1 = max(cand1)
    bridge = ctx[min(p1 + 1, W - 1)]
    cand2 = [i for i in range(W - 1) if ctx[i] == bridge and i != p1 + 1]
    if not cand2:
        return None
    p2 = max(cand2)
    return ctx[min(p2 + 1, W - 1)]


def serve(ctx, idioms, ngrams, W):
    """The runtime decision: TRUSTED idioms (frame-matched, in priority order) → GATED n-grams (longest matching suffix)
    → ABSTAIN. Host-side, no souffle — even the idioms (gate = frame + slot→table; compose = frame + operands→value-sum→
    output) are a structured lookup, so the whole package is consumable without an engine. (No min_det: the manifest holds
    only confident rules — gating happened at build.)"""
    for r in idioms:                                            # trusted tier (causal), first — gate/compose only
        if r["kind"] in ("induction", "succession", "relation", "dgate", "dgate2", "pointer", "khop"):  # OOD/derived route below — handled after
            continue
        fr = r["frame"]
        if not all(len(ctx) >= o and ctx[-o] == t for o, t in fr.items()):
            continue
        if r["kind"] == "gate":
            k = r["slot"]
            if len(ctx) >= k and ctx[-k] in r["table"]:
                return {"answer": r["table"][ctx[-k]], "tier": "trusted", "basis": "causal", "citation": r["cite"], "rule": r["id"]}
        else:  # compose
            k1, k2 = r["operands"]
            if len(ctx) >= max(k1, k2) and ctx[-k1] in r["valmap"] and ctx[-k2] in r["valmap"]:
                ssum = r["valmap"][ctx[-k1]] + r["valmap"][ctx[-k2]]
                if ssum in r["sum"]:
                    return {"answer": r["sum"][ssum], "tier": "trusted", "basis": "causal", "citation": r["cite"], "rule": r["id"]}
    for k in range(min(len(ctx), W), 0, -1):                    # gated n-gram tier (longest suffix wins)
        s = tuple(ctx[-k:])
        if s in ngrams[k]:
            out, basis, cite, _conf, _st, _org, _counts = ngrams[k][s]
            return {"answer": out, "tier": "gated", "basis": basis, "citation": cite, "k": k}
    # relation (causal EQ-GUARD + COPY), OOD fallback ABOVE succession/induction — the most specific of the routed
    # circuits: fires iff ctx[-i] == ctx[-j] for every pair in `eq` (offsets 1-based from the end), then copies
    # ctx[-copy]. The learned repetition rule is eq=[[1,2]], copy=1. `confidence` (optional, all trusted kinds) is
    # the rule's held-out fired-accuracy — shipped so a support-weighted runtime can arbitrate tiers per answer.
    for r in idioms:
        if r["kind"] != "relation":
            continue
        offs = [o for ij in r["eq"] for o in ij] + [r["copy"]]
        if max(offs) <= len(ctx) and all(ctx[-i] == ctx[-j] for i, j in r["eq"]):
            return {"answer": ctx[-r["copy"]], "tier": "trusted", "basis": "causal",
                    "citation": r["cite"], "rule": r["id"], "circuit": "relation"}
    # succession (causal ORDINAL), OOD fallback ABOVE induction — reached only after an n-gram miss. Predicts the
    # successor of a >=3-long consecutive ascending run of ordinal tokens ([… X X+1 X+2] → X+3); matches py_succ.
    for r in idioms:
        if r["kind"] != "succession":
            continue
        lord, lat = r["lord"], r["lat"]
        pres = {lord[t] for t in ctx if t in lord}
        if pres:
            o = max(pres)
            if (o - 1) in pres and (o - 2) in pres and (o + 1) in lat:
                return {"answer": lat[o + 1], "tier": "trusted", "basis": "causal",
                        "citation": r["cite"], "rule": r["id"], "circuit": "succession"}
    # induction (causal COPY), OOD fallback — reached ONLY after an n-gram miss, so it costs nothing on the hot path
    # (a package with no induction rules skips this loop entirely). Fires where no n-gram matched: find the previous
    # occurrence of the current L-token suffix and copy its successor ([… A B … A] → B). LONGEST L first — a longer
    # repeated context is a more specific, higher-confidence match; among several earlier occurrences the MOST RECENT
    # (max j) is the one copied from.
    for r in sorted((x for x in idioms if x["kind"] == "induction"), key=lambda x: -x["L"]):
        L = r["L"]
        if len(ctx) > L:
            suf = tuple(ctx[-L:])
            js = [j for j in range(len(ctx) - L) if tuple(ctx[j:j + L]) == suf]
            if js and max(js) + L < len(ctx):
                return {"answer": ctx[max(js) + L], "tier": "trusted", "basis": "causal",
                        "citation": r["cite"], "rule": r["id"], "circuit": "induction", "origin": r.get("origin", "teacher")}
    # khop (causal 2-HOP), OOD fallback after induction (a 2-hop circuit is checked once the
    # cheaper single-hop induction copy has already had its chance). Mirrors pil mir_khop2/
    # _DL_KHOP exactly via _khop_predict — see that function for the bridge-site exclusion.
    for r in idioms:
        if r["kind"] != "khop":
            continue
        pred = _khop_predict(ctx, r["lo"], r["hi"])
        if pred is not None:
            return {"answer": pred, "tier": "trusted", "basis": "causal",
                    "citation": r["cite"], "rule": r["id"], "circuit": "khop"}
    return None  # ABSTAIN


def serve_sw(ctx, idioms, ngrams, W, m_derived=None, cmap=None, m_tau=None):
    """SUPPORT-WEIGHTED cover (manifest cover: "support-weighted"): every applicable rule fires and
    the answer with the highest confidence wins -- the argmax policy whose dominance over every
    fixed priority is the kernel-checked C10 (i-orca Arbitration.thy: argmax_policy_optimal), with
    calibration as the stated premise. Confidences are what the package ships: per-key
    Laplace-shrunk determinism for table rules (ngram "confidence", gate "confs"), held-out
    fired-accuracy for scalar kinds (relation/induction "confidence"). Ties keep the FIRST
    candidate in manifest order (the learner's admitted order), n-grams after idioms, longest
    first."""
    best, bestc = None, float("-inf")                       # stratum-1 pool
    best2, bestc2 = None, float("-inf")                     # stratum-2 pool (fall-through)
    feats, fpos = {}, {}
    for d in (m_derived or []):
        if d["kind"] == "bracket-mate":                        # PROVED extractor (pil wyly_mate_certify):
            stack = []                                         # innermost UNCLOSED opener; with "succ"
            for i, t in enumerate(ctx):                        # the feature is the token at pos+succ
                if t in d["openers"]:                          # (ROLE COMPOSITION, e.g. after-the-mate)
                    stack.append(i)
                elif t in d["closers"] and stack:
                    stack.pop()
            fpos[d["id"]] = stack[-1] if stack else -1
            p_ = stack[-1] + d["succ"] if stack else -1
            feats[d["id"]] = ctx[p_] if 0 <= p_ < len(ctx) and stack else -1
        elif d["kind"] in ("recent-member", "recent-unique"):  # most recent member [occurring once]
            p_ = -1
            for i, t in enumerate(ctx):
                if t in d["members"] and (d["kind"] == "recent-member" or ctx.count(t) == 1):
                    p_ = i
            fpos[d["id"]] = p_
            p2 = p_ + d["succ"] if p_ >= 0 else -1
            feats[d["id"]] = ctx[p2] if 0 <= p2 < len(ctx) and p_ >= 0 else -1
        elif d["kind"] == "bracket-depth":                     # the balance counter (capped)
            depth = sum((t in d["openers"]) - (t in d["closers"]) for t in ctx)
            feats[d["id"]] = min(max(depth, 0), d["cap"])
        elif d["kind"] == "since-member":                      # DISCOURSE: tokens since last member
            p_ = -1
            for i, t in enumerate(ctx):
                if t in d["members"]:
                    p_ = i
            feats[d["id"]] = min(len(ctx) - 1 - p_, d["cap"]) if p_ >= 0 else d["cap"] + 1
        elif d["kind"] == "dstate":                            # RHETORICAL STATE: bucketed
            p_ = -1                                            # position-in-sentence x quote parity
            for i, t in enumerate(ctx):
                if t in d["members"]:
                    p_ = i
            since = min(len(ctx) - 1 - p_, 33) if p_ >= 0 else 34
            bucket = sum(since > e for e in (2, 6, 14, 32))
            par = sum(1 for t in ctx if t in d["quote_members"]) % 2
            feats[d["id"]] = bucket * 2 + par
        elif d["kind"] == "member-parity":                     # DISCOURSE: quotation scope
            feats[d["id"]] = sum(1 for t in ctx if t in d["members"]) % 2
        elif d["kind"] == "estate2":                           # WORLD-STATE FOLD: loc[entity],
            sl = d.get("slot")                                 # holder/loc/history[object]
            if sl and (len(ctx) < 2 or ctx[-2] != sl[0] or ctx[-1] != sl[1]):
                feats[d["id"]] = -1
                fpos[d["id"]] = -1
                continue
            E2, L2, O2 = d["entities"], d["locations"], d["objects"]
            MV, TK, DP = d["move_verbs"], d["take_verbs"], d["drop_verbs"]
            loc2, holder2, oloc2, ohist2 = {}, {}, {}, {}
            for i in range(len(ctx) - 1):
                w, v = ctx[i], ctx[i + 1]
                if w not in E2:
                    continue
                if v in MV:
                    dest = next((ctx[j] for j in range(i + 2, min(i + 7, len(ctx)))
                                 if ctx[j] in L2), None)
                    if dest is not None:
                        loc2[w] = dest
                        for o, h in holder2.items():
                            if h == w and oloc2.get(o) != dest:
                                ohist2.setdefault(o, []).append(oloc2.get(o, -1))
                                oloc2[o] = dest
                elif v in TK or v in DP:
                    ob = next((ctx[j] for j in range(i + 2, min(i + 6, len(ctx)))
                               if ctx[j] in O2), None)
                    if ob is None:
                        continue
                    if v in TK:
                        holder2[ob] = w
                        if w in loc2 and oloc2.get(ob) != loc2[w]:
                            ohist2.setdefault(ob, []).append(oloc2.get(ob, -1))
                            oloc2[ob] = loc2[w]
                    else:
                        holder2.pop(ob, None)
            qo = next((ctx[i] for i in range(len(ctx) - 1, -1, -1) if ctx[i] in O2), None)
            feat = -1
            if qo is not None:
                if d["mode"] == "is":
                    feat = oloc2.get(qo, -1)
                else:                                          # "before": predecessor of ref
                    qi = max(i for i in range(len(ctx)) if ctx[i] == qo)
                    ref = next((ctx[i] for i in range(len(ctx) - 1, qi, -1)
                                if ctx[i] in L2), None)
                    hist = ohist2.get(qo, []) + [oloc2.get(qo, -1)]
                    for k in range(len(hist) - 1, 0, -1):
                        if hist[k] == ref:
                            feat = hist[k - 1]
                            break
            feats[d["id"]] = feat
            fpos[d["id"]] = len(ctx) - 1
        elif d["kind"] == "estate":                            # ENTITY-STATE REGISTER (EAV,
            E, V = d["entity_members"], d["value_members"]     # one attribute per instance):
            AV, wt = d["avoid"], d["within"]                   # last-writer-wins fold
            sl = d.get("slot")
            if sl and (len(ctx) < 2 or ctx[-2] != sl[0] or ctx[-1] != sl[1]):
                feats[d["id"]] = -1
                fpos[d["id"]] = -1
                continue
            qi = -1
            for i in range(len(ctx) - 1, -1, -1):
                if ctx[i] in E:
                    qi = i
                    break
            feat = -1
            if qi >= 0:
                qt = ctx[qi]
                for i in range(qi):
                    if ctx[i] != qt or any(t in AV for t in ctx[i + 1:i + 4]):
                        continue
                    for j in range(i + 1, min(i + 1 + wt, len(ctx))):
                        if ctx[j] in V:
                            feat = ctx[j]
                            break
            feats[d["id"]] = feat
            fpos[d["id"]] = qi
        elif d["kind"] == "prev-occ":                          # CHAINED role: the previous occurrence
            bp = fpos.get(d["of"], -1)                         # of the referenced feature's token --
            if bp >= 0 and d["of_shift"]:                      # of_shift: e.g. -1 = the CLAIMANT
                bp = bp + d["of_shift"]                        # before the attribution verb
                bp = bp if 0 <= bp < len(ctx) else -1
            q = -1                                             # composed with succ: the entity ECHO
            if bp >= 0:
                for i in range(bp):
                    if ctx[i] == ctx[bp] and not any(
                            t in d["avoid"]
                            for t in ctx[i + 1:i + 1 + d["look"]]):
                        q = i
            fpos[d["id"]] = q
            p2 = q + d["succ"] if q >= 0 else -1
            feats[d["id"]] = ctx[p2] if 0 <= p2 < len(ctx) and q >= 0 else -1

    def consider(ans, c, meta, stratum=1):
        nonlocal best, bestc, best2, bestc2
        if c is None:
            return
        if stratum <= 1:
            if c > bestc:
                best, bestc = dict(meta, answer=ans, stratum=1), float(c)
        elif c > bestc2:
            best2, bestc2 = dict(meta, answer=ans, stratum=stratum), float(c)

    cmap = cmap if cmap is not None else {}
    for r in idioms:
        k = r["kind"]
        if k == "pointer":
            lmax, n = r["lmax"], len(ctx)
            bl = blc = bp = -1
            for pp in range(1, n):                             # source position: predict ctx[pp]
                l = lc = 0
                for j in range(1, lmax + 1):
                    if pp - j < 0:
                        break
                    a, b = ctx[pp - j], ctx[n - j]
                    if l == j - 1 and a == b:
                        l = j
                    if lc == j - 1 and cmap.get(a, a) == cmap.get(b, b):
                        lc = j
                    if lc < j:
                        break
                if (l >= 1 or lc >= 2) and (l, lc, pp) >= (bl, blc, bp):
                    bl, blc, bp = l, lc, pp
            if bp >= 0 and (bl, blc) in r["cells"]:
                consider(ctx[bp], r["cells"][(bl, blc)],
                         {"tier": "trusted", "basis": "causal", "citation": r["cite"],
                          "rule": r["id"], "circuit": "pointer", "origin": r.get("origin", "teacher")},
                         stratum=r.get("stratum", 1))
        elif k == "dgate2":
            fa, fb = feats.get(r["fa"], -1), feats.get(r["fb"], -1)
            if fa >= 0 and fb >= 0 and (fa, fb) in r["table"]:
                consider(r["table"][(fa, fb)], r["confs"].get((fa, fb), 0.0),
                         {"tier": "gated", "basis": "observational", "citation": r["cite"],
                          "rule": r["id"], "circuit": "dgate2", "origin": r.get("origin", "teacher")}, stratum=r.get("stratum", 1))
        elif k == "dgate":
            f = feats.get(r["feature"], -1)
            if f >= 0 and (f, ctx[-1]) in r["table"]:
                consider(r["table"][(f, ctx[-1])], r["confs"].get((f, ctx[-1]), 0.0),
                         {"tier": "gated", "basis": "observational", "citation": r["cite"],
                          "rule": r["id"], "circuit": "dgate", "origin": r.get("origin", "teacher")}, stratum=r.get("stratum", 1))
        elif k == "gate":
            fr = r["frame"]
            if not all(len(ctx) >= o and ctx[-o] == t for o, t in fr.items()):
                continue
            so = r["slot"]
            if len(ctx) >= so and ctx[-so] in r["table"]:
                consider(r["table"][ctx[-so]], r.get("confs", {}).get(ctx[-so], 0.0),
                         {"tier": "gated", "basis": "observational", "citation": r["cite"],
                          "rule": r["id"], "circuit": "gate", "origin": r.get("origin", "teacher")}, stratum=r.get("stratum", 1))
        elif k == "relation":
            offs = [o for ij in r["eq"] for o in ij] + [r["copy"]]
            if max(offs) <= len(ctx) and all(ctx[-i] == ctx[-j] for i, j in r["eq"]):
                consider(ctx[-r["copy"]], r.get("conf") or 0.0,
                         {"tier": "trusted", "basis": "causal", "citation": r["cite"],
                          "rule": r["id"], "circuit": "relation"}, stratum=r.get("stratum", 1))
        elif k == "induction":
            L = r["L"]
            if len(ctx) > L:
                suf = tuple(ctx[-L:])
                js = [j for j in range(len(ctx) - L) if tuple(ctx[j:j + L]) == suf]
                if js and max(js) + L < len(ctx):
                    consider(ctx[max(js) + L], r.get("conf") or 0.0,
                             {"tier": "trusted", "basis": "causal", "citation": r["cite"],
                              "rule": r["id"], "circuit": "induction", "origin": r.get("origin", "teacher")}, stratum=r.get("stratum", 1))
        elif k == "khop":
            pred = _khop_predict(ctx, r["lo"], r["hi"])
            if pred is not None:
                consider(pred, r.get("conf") or 0.0,
                         {"tier": "trusted", "basis": "causal", "citation": r["cite"],
                          "rule": r["id"], "circuit": "khop", "origin": r.get("origin", "teacher")}, stratum=r.get("stratum", 1))
    for k in range(min(len(ctx), W), 0, -1):
        s = tuple(ctx[-k:])
        if s in ngrams[k]:
            out, basis, cite, conf, st, org, _counts = ngrams[k][s]
            consider(out, conf if conf is not None else 0.0,
                     {"tier": "gated", "basis": basis, "citation": cite, "k": k,
                      "origin": org}, stratum=st)
    tau = (m_tau if m_tau is not None else 0.35)
    if best is not None and bestc >= tau:
        best["confidence"] = round(bestc, 4)
        return best
    pool = max(((best, bestc), (best2, bestc2)), key=lambda x: x[1])
    if pool[0] is None:
        return None  # ABSTAIN
    pool[0]["confidence"] = round(pool[1], 4)
    return pool[0]


def serve_energy(ctx, idioms, ngrams, W, M=1, beam_width=1, m_derived=None, cmap=None, m_tau=None):
    """ENERGY-BEAM cover (manifest cover: "energy-beam"), M=1 / beam_width=1 CORNER: a genuine
    REDUCTION of serve_sw, not a delegate in spirit -- at this corner the energy tuple's margin
    component ties (no hard legality term in this slice, so every candidate's PIC turnstile margin
    is identical) and det_rank collapses to serve_sw's own confidence order, so serve_sw's argmax
    commit IS the argmin-energy commit. The integer det_rank cross-multiply and a discriminating
    margin only become load-bearing in the Slice-2 M-step beam (once the manifest carries raw
    per-key counts and a hard term). Cert nuance: at M=1 the committed token carries a per-token
    cert (no lookahead), identical to classic DECIDE."""
    if M != 1 or beam_width != 1:
        raise NotImplementedError(
            "energy-beam M>1 / beam_width>1 arrives in Slice 2 (the M-step beam)")
    result = serve_sw(ctx, idioms, ngrams, W, m_derived=m_derived, cmap=cmap, m_tau=m_tau)
    if result is not None:
        result["cert_kind"] = "per-token"
    return result


def decide(ctx, idioms, ngrams, m):
    """Dispatch on the manifest's declared cover semantics."""
    if m.get("cover") == "support-weighted":
        return serve_sw(ctx, idioms, ngrams, m.get("W", 1), m_derived=m.get("_derived"),
                        cmap=m.get("_cmap"), m_tau=m.get("_tau"))
    if m.get("cover") == "energy-beam":
        return serve_energy(ctx, idioms, ngrams, m.get("W", 1), M=m.get("M", 1),
                            beam_width=m.get("beam_width", 1), m_derived=m.get("_derived"),
                            cmap=m.get("_cmap"), m_tau=m.get("_tau"))
    return serve(ctx, idioms, ngrams, m.get("W", 1))


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "models", "llama32_1b")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    minsupp = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mindet = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    name = os.path.basename(md)
    hold = make_package(md, W, minsupp, mindet)                  # builder: corpus → package (circuits.abstain.dl + manifest.json)
    idioms, ngrams, manifest = load_package(os.path.join(md, "manifest.json"))  # RUNTIME: load manifest.json only
    meta_on = any("citation" in r for r in manifest["rules"][:50])
    print(f"=== serve_package · {name} · loaded {manifest['n_rules']} rules from manifest.json "
          f"(thin runtime: no corpus/souffle/model){' · citations ✓' if meta_on else ''} ===")
    ans = cor = 0
    for ctx, o, _ in hold:                                       # held-out scorecard (eval only; uses the corpus)
        r = serve(ctx, idioms, ngrams, W)
        if r is not None:
            ans += 1; cor += (r["answer"] == o)
    H = len(hold)
    print(f"scorecard: coverage {ans/H:.0%}  precision {cor/ans if ans else 0:.0%}  abstain {1-ans/H:.0%}  (n={H})")
    dec = lambda t: f"id{t}"
    tok = None
    tp = os.path.join(md, "bundle.tokenizer.json")
    if os.path.exists(tp):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(tp); dec = lambda t: repr(tok.decode([t]))
        except Exception:
            pass
    shown_ans = shown_ab = 0
    print("sample served decisions (answer → citation, or abstain):")
    for ctx, o, _ in hold:
        r = serve(ctx, idioms, ngrams, W)
        if r is not None and shown_ans < 3:
            mark = "✓" if r["answer"] == o else "✗"
            print(f"  ANSWER {dec(r['answer'])} {mark}  [{r['tier']}/{r['basis']}]  cite: {r['citation']}")
            shown_ans += 1
        elif r is None and shown_ab < 2:
            print("  ABSTAIN (no confident rule) — would defer to backstop / refuse")
            shown_ab += 1
        if shown_ans >= 3 and shown_ab >= 2:
            break
    if tok is not None:
        print("NL-query path (tokenize → serve) — the token-space bridge sgiandubh's runtime needs:")
        for q in ["The United States of America", "xylophone quantum tariff zzzqq"]:
            ctx = tok.encode(q, add_special_tokens=False).ids[-W:]
            r = serve(ctx, idioms, ngrams, W)
            print(f"  query {q!r} → " + (f"answer {dec(r['answer'])} (cite {r['citation']})" if r else "ABSTAIN"))
    else:
        print("(no bundle.tokenizer.json — NL path skipped; this model serves token contexts only)")
    # manifest shape, for C++ porters
    s0 = manifest["rules"][0]
    print(f"manifest.json rule shape (for the C++ runtime): {{id, ctx:[token ids], out, support, determinism, cite[, citation]}}")
    print(f"  e.g. {json.dumps({k: s0[k] for k in ('id', 'ctx', 'out', 'support', 'determinism') if k in s0})}")


if __name__ == "__main__":
    main()


STOPWORDS = {"the", "of", "is", "a", "an", "in", "for", "as", "with", "what", "which",
             "does", "do", "where", "how", "many", "s", "its", "on", "table", "periodic"}


def canonicalize(query, canon):
    """map a free-phrased query onto the package's mined template inventory.
    Returns (canonical_string, {"entity": E, "template": prefix}) or None (no parse -> the
    caller should ABSTAIN, never fall through to raw fragment matching)."""
    import re as _re
    words = [w.split("'")[0] for w in _re.findall(r"[A-Za-z']+", query)]
    lower2ent = {e.lower(): e for e in canon.get("entities", [])}
    ent = next((lower2ent[w.lower()] for w in words if w.lower() in lower2ent), None)
    if ent is None:
        return None
    for t in canon.get("templates", []):                     # ALREADY canonical -> pass through
        if query.strip().rstrip(".").lower() == \
                t["prefix"].replace("{E}", ent).lower():
            return query.strip(), {"entity": ent, "template": t["prefix"], "score": 1.0}
    qkw = {w.lower().rstrip("s") for w in words} - STOPWORDS - {ent.lower()}
    best, bestscore = None, 0.0
    for t in canon.get("templates", []):
        kw = {k.rstrip("s") for k in t["keywords"]} - STOPWORDS
        hit = len(qkw & kw)
        score = hit / max(1, len(qkw))                       # q-side coverage: how much of the
        if hit and score > bestscore:                        # query the template explains
            best, bestscore = t, score
    if best is None or bestscore < 0.4:
        return None
    return (best["prefix"].replace("{E}", ent),
            {"entity": ent, "template": best["prefix"], "score": round(bestscore, 2)})
