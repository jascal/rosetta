"""rosetta · the bounded-expert package must keep the causal-vs-observational distinction explicit (the convergence
strengthening: trusted causal idioms + gated observational n-grams). Synthetic idioms + corpus → fast, no oracle/learning."""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "py"))
from idiom_learn import emit_expert_package  # noqa: E402


def test_package_tags_causal_idioms_vs_observational_ngrams(tmp_path):
    # 3 instances a select-gate covers (frame: last tok=30; slot @2 → table) + 1 it doesn't (→ residual n-gram)
    insts = [[10, 20, 30], [11, 21, 30], [12, 22, 30], [13, 99, 98]]
    refs = [40, 41, 42, 50]
    idxs = [0, 1, 2, 3]
    gate = {"frame": {1: 30}, "k": 2, "table": {20: 40, 21: 41, 22: 42}, "causal": 1.0, "fmatch": [0, 1, 2]}
    out, man = emit_expert_package(str(tmp_path), insts, refs, idxs, [gate], [], [], 3, "test", minsupp=1, mindet=1.0)
    assert os.path.exists(out)                                   # circuits.expert.dl emitted
    m = json.load(open(man))
    assert m["trusted_idioms"] == 1
    g = next(r for r in m["rules"] if r["kind"] == "gate")       # the causal idiom = TRUSTED tier
    assert g["tier"] == "trusted" and g["basis"] == "causal" and g["causal"] == 1.0
    ng = [r for r in m["rules"] if r["kind"] == "ngram"]         # the residual → GATED observational tier
    assert ng, "expected a gated n-gram for the uncovered instance"
    assert all(r["tier"] == "gated" and r["basis"] == "observational" for r in ng)


def test_serve_package_consumes_tiered_package(tmp_path):
    # round-trip: emit the tiered package, then load + serve it via the runtime — idioms (trusted) before n-grams (gated).
    from serve_package import load_package, serve
    insts = [[10, 20, 30], [11, 21, 30], [12, 22, 30], [13, 99, 98]]
    refs = [40, 41, 42, 50]
    gate = {"frame": {1: 30}, "k": 2, "table": {20: 40, 21: 41, 22: 42}, "causal": 1.0, "fmatch": [0, 1, 2]}
    _, man = emit_expert_package(str(tmp_path), insts, refs, [0, 1, 2, 3], [gate], [], [], 3, "test", minsupp=1, mindet=1.0)
    idioms, ngrams, m = load_package(man)
    assert len(idioms) == 1 and idioms[0]["kind"] == "gate"
    r = serve([10, 20, 30], idioms, ngrams, 8)                   # gate-covered → TRUSTED/causal
    assert r and r["answer"] == 40 and r["tier"] == "trusted" and r["basis"] == "causal"
    r2 = serve([13, 99, 98], idioms, ngrams, 8)                  # residual → GATED/observational n-gram
    assert r2 and r2["answer"] == 50 and r2["tier"] == "gated" and r2["basis"] == "observational"
    assert serve([7, 7, 7], idioms, ngrams, 8) is None           # nothing fires → ABSTAIN


def test_induction_circuit_wired_into_package(tmp_path):
    # a causally-confirmed induction rel → a first-class `induction` manifest rule, served OOD (after n-grams).
    from serve_package import load_package, serve
    insts = [[5, 9, 5]]                                          # [… A B … A] → B: last tok 5, prev-occ successor = 9
    refs = [9]
    rels = [{"L": 1, "causal": 1.0, "obs": 1.0}]
    _, man = emit_expert_package(str(tmp_path), insts, refs, [0], [], [], rels, 8, "test", minsupp=3, mindet=1.0)
    m = json.load(open(man))
    assert m["induction_ood"] == 1 and m["induction_cover"] == 1   # induction counted, not a silent OOD limb
    ir = next(r for r in m["rules"] if r["kind"] == "induction")
    assert ir["tier"] == "trusted" and ir["basis"] == "causal" and ir["routing"] == "ood" and ir["L"] == 1
    idioms, ngrams, _ = load_package(man)
    r = serve([5, 9, 5], idioms, ngrams, 8)                       # induction fires (no n-gram at support 1) → copies 9
    assert r and r["answer"] == 9 and r["tier"] == "trusted" and r["circuit"] == "induction"
    assert serve([1, 2, 3], idioms, ngrams, 8) is None           # no recurring suffix → induction can't fire → ABSTAIN


def test_succession_circuit_wired_into_package(tmp_path):
    # a causally-confirmed ordinal alphabet (token 100→ord0, 101→ord1, …) → a first-class `succession` manifest rule.
    from serve_package import load_package, serve
    lord = {100 + i: i for i in range(6)}                         # tokens 100..105 = ordinals 0..5
    lat = {i: 100 + i for i in range(6)}
    insts = [[100, 101, 102]]                                     # a 3-run 0,1,2 → predict ord 3 = token 103
    refs = [103]
    succ = {"lord": lord, "lat": lat, "causal": 1.0}
    _, man = emit_expert_package(str(tmp_path), insts, refs, [0], [], [], [], 8, "test", minsupp=3, mindet=1.0, succ=succ)
    m = json.load(open(man))
    assert m["succession_ood"] == 1 and m["succession_cover"] == 1
    sr = next(r for r in m["rules"] if r["kind"] == "succession")
    assert sr["tier"] == "trusted" and sr["basis"] == "causal" and sr["routing"] == "ood"
    idioms, ngrams, _ = load_package(man)
    r = serve([100, 101, 102], idioms, ngrams, 8)                 # ascending run → predict the successor
    assert r and r["answer"] == 103 and r["circuit"] == "succession"
    assert serve([100, 105, 101], idioms, ngrams, 8) is None      # not a consecutive run → ABSTAIN
