"""rosetta · khop (2-hop) rule kind — serve-branch unit tests + round-trip.

Mirrors pil `mir_khop2`/`_DL_KHOP` (wyly_selfcompile.py L171-203) EXACTLY. The load-bearing
detail is the BRIDGE-SITE EXCLUSION (`i != p1+1`) in the second hop's rightmost-match search --
test_khop_bridge_site_exclusion is the case that fails without it. No pil import: all vectors
are hand-built here. Pure Python -- no souffle, no model, no tokenizer.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
from serve_package import load_package, decide, serve, serve_sw  # noqa: E402


def _rule(lo, hi, conf=0.9, id_="khop_t", cite="test-cite", stratum=1, origin="teacher"):
    """The idiom dict load_package would build for a `khop` manifest rule."""
    return {"kind": "khop", "stratum": stratum, "origin": origin, "id": id_, "cite": cite,
            "lo": lo, "hi": hi, "conf": conf}


def test_khop_basic_two_hop():
    """(a) basic 2-hop: q matches once, bridge's earlier match is NOT only at the bridge's own
    site (p1+1) -- another earlier occurrence exists past it, so the exclusion guard is inert
    here -- confirms the ungated case is correct."""
    ctx = [11, 12, 7, 20, 13, 20, 99, 7]           # q=7 -> p1=2 -> bridge=ctx[3]=20 -> earlier
    idioms = [_rule(0, 100)]                        # match of 20 at i=5 (not i=3) -> pred=ctx[6]=99
    ngrams = defaultdict(dict)
    r = serve(ctx, idioms, ngrams, len(ctx))
    assert r is not None and r["answer"] == 99 and r["circuit"] == "khop"
    r2 = serve_sw(ctx, idioms, ngrams, len(ctx))
    assert r2 is not None and r2["answer"] == 99 and r2["circuit"] == "khop"


def test_khop_bridge_site_exclusion():
    """(b) THE LOAD-BEARING CASE: without the `i != p1+1` guard, the bridge's own site (i=3)
    re-matches and wins the rightmost-match argmax (predicting ctx[4]=4, WRONG). With the guard,
    i=3 is excluded and the earlier real match at i=0 wins (predicting ctx[1]=2, CORRECT). This
    test fails if the guard is missing or misapplied."""
    ctx = [1, 2, 3, 1, 4, 3]                        # q=3 -> p1=2 -> bridge=ctx[3]=1
    idioms = [_rule(0, 100)]                         # earlier matches of 1: i=0 and i=3(=p1+1)
    ngrams = defaultdict(dict)
    r = serve(ctx, idioms, ngrams, len(ctx))
    assert r is not None and r["answer"] == 2, \
        f"bridge-site exclusion violated: got {r}, expected answer=2 (unguarded would give 4)"
    r2 = serve_sw(ctx, idioms, ngrams, len(ctx))
    assert r2 is not None and r2["answer"] == 2, \
        f"bridge-site exclusion violated in serve_sw: got {r2}, expected answer=2"


def test_khop_abstain_out_of_range():
    """(c1) q out of [lo,hi] -> does not fire."""
    ctx = [1, 2, 3, 1, 4, 3]
    idioms = [_rule(100, 200)]                       # q=3 not in [100,200]
    ngrams = defaultdict(dict)
    assert serve(ctx, idioms, ngrams, len(ctx)) is None
    assert serve_sw(ctx, idioms, ngrams, len(ctx)) is None


def test_khop_abstain_no_first_hop_match():
    """(c2) q has no earlier match in the context -> does not fire."""
    ctx = [1, 2, 4, 5, 6, 9]                          # q=9 appears nowhere else
    idioms = [_rule(0, 100)]
    ngrams = defaultdict(dict)
    assert serve(ctx, idioms, ngrams, len(ctx)) is None
    assert serve_sw(ctx, idioms, ngrams, len(ctx)) is None


def test_khop_abstain_no_second_hop_match():
    """(c3) bridge has no earlier match once its own site (p1+1) is excluded -> does not fire."""
    ctx = [9, 2, 3, 7, 4, 3]                          # q=3 -> p1=2 -> bridge=ctx[3]=7, which
    idioms = [_rule(0, 100)]                          # appears NOWHERE else in [0, W-2]
    ngrams = defaultdict(dict)
    assert serve(ctx, idioms, ngrams, len(ctx)) is None
    assert serve_sw(ctx, idioms, ngrams, len(ctx)) is None


def _ref_khop(ctx, lo, hi):
    """Independent local reference (transcribed directly from the SPEC section 4 / pil
    mir_khop2 -- NOT imported from serve_package, so this is a genuine cross-check of the
    implementation)."""
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


def test_khop_roundtrip_manifest_load_and_serve(tmp_path):
    """(d) full round trip: a minimal manifest.json with one khop rule -> load_package ->
    decide() over several contexts (incl. the bridge-site-exclusion ctx) -> assert the served
    prediction equals the local reference, for BOTH cover regimes (plain priority `serve` and
    `cover: support-weighted` -> `serve_sw`, dispatched via `decide`)."""
    rule = {"kind": "khop", "lo": 0, "hi": 100, "id": "khop_rt", "tier": "trusted",
            "basis": "causal", "confidence": 0.9, "citation": "rt-cite"}
    contexts = [
        [11, 12, 7, 20, 13, 20, 99, 7],   # (a)-style: guard inert, pred=99
        [1, 2, 3, 1, 4, 3],               # (b): bridge-site exclusion, pred=2
        [1, 2, 4, 5, 6, 9],               # (c2): abstain, pred=None
        [9, 2, 3, 7, 4, 3],               # (c3): abstain, pred=None
    ]

    for cover in (None, "support-weighted"):
        manifest = {"model": "khop-rt-test", "W": 8, "trusted_idioms": 1, "gated_ngrams": 0,
                    "minsupp": 1, "mindet": 1.0, "rules": [rule]}
        if cover:
            manifest["cover"] = cover
        mp = tmp_path / f"manifest_{cover or 'plain'}.json"
        mp.write_text(json.dumps(manifest))
        idioms, ngrams, m = load_package(str(mp))
        assert len(idioms) == 1 and idioms[0]["kind"] == "khop"
        assert idioms[0]["lo"] == 0 and idioms[0]["hi"] == 100
        for ctx in contexts:
            expected = _ref_khop(ctx, 0, 100)
            got = decide(ctx, idioms, ngrams, m)
            got_ans = got["answer"] if got is not None else None
            assert got_ans == expected, \
                f"cover={cover!r} ctx={ctx}: decide() answer={got_ans}, reference={expected}"
            if expected is not None:
                assert got["circuit"] == "khop" and got["citation"] == "rt-cite" \
                    and got["rule"] == "khop_rt"
