"""TextOracle and shared candidate-sweep unit tests."""
from __future__ import annotations

import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "py"))

from serve_package import enumerate_candidates, serve_energy  # noqa: E402
from text_oracle import TextOracle  # noqa: E402


def _ngram(out, conf, cite, counts):
    return (out, "observational", cite, conf, 1, "teacher", counts)


def test_enumerate_candidates_threads_counts_in_sweep_order():
    idioms = [
        {
            "kind": "gate", "id": "g1", "frame": {}, "slot": 1,
            "table": {1: 10}, "confs": {1: 0.6}, "counts": {1: (3, 5)},
            "cite": "g1", "stratum": 1, "origin": "teacher",
        },
        {
            "kind": "gate", "id": "g2", "frame": {}, "slot": 1,
            "table": {1: 10}, "confs": {1: 0.8}, "counts": {1: (4, 5)},
            "cite": "g2", "stratum": 1, "origin": "teacher",
        },
    ]
    ngrams = defaultdict(dict)
    ngrams[1][(1,)] = _ngram(20, 0.7, "ng1", (7, 12))

    candidates = enumerate_candidates((1,), idioms, ngrams, 1)
    assert [candidate.answer for candidate in candidates] == [10, 10, 20]
    assert [candidate.counts for candidate in candidates] == [
        (3, 5), (4, 5), (7, 12),
    ]


def test_text_oracle_deduplicates_by_best_confidence_and_keeps_first_tie():
    idioms = [
        {
            "kind": "gate", "id": rule_id, "frame": {}, "slot": 1,
            "table": {1: 10}, "confs": {1: conf}, "counts": {1: counts},
            "cite": rule_id, "stratum": 1, "origin": "teacher",
        }
        for rule_id, conf, counts in (
            ("lower", 0.6, (3, 5)),
            ("winner", 0.8, (4, 5)),
            ("tied-later", 0.8, (8, 10)),
        )
    ]
    oracle = TextOracle((1,), idioms, defaultdict(dict), 1)

    children = oracle.expand(oracle.initial_state())
    assert len(children) == 1
    child_state, steps = children[0]
    assert child_state == (1, 10)
    assert steps[0].counts == (4, 5)
    assert steps[0].meta["citation"] == "winner"
    assert steps[0].meta["conf"] == 0.8


def test_serve_energy_zero_margin_gate_and_later_wipeout_fallback():
    ngrams = defaultdict(dict)
    ngrams[1][(1,)] = _ngram(2, 0.8, "first", (8, 10))
    ngrams[1][(2,)] = _ngram(3, 0.7, "second", (7, 10))

    gated = serve_energy((1,), [], ngrams, 1, M=2, beam_width=2)
    assert gated == {
        "tier": "gated",
        "basis": "observational",
        "citation": "first",
        "k": 1,
        "origin": "teacher",
        "stratum": 1,
        "answer": 2,
        "confidence": 0.8,
        "cert_kind": "per-token",
    }

    del ngrams[1][(2,)]
    fallback = serve_energy((1,), [], ngrams, 1, M=2, beam_width=2)
    assert fallback["answer"] == 2
    assert fallback["cert_kind"] == "per-token"
