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
