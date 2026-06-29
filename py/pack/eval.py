"""pack.eval — the expert scorecard + hard-fail gate (EXPERTS.md).

An expert is a CLAIM ("over domain D: coverage C, precision P, leak L"); this makes it true and backs it with an
artifact. We grade a built package by SERVING it (the thin runtime: trusted idioms → gated n-grams → abstain) over a
held-out in-domain set + an off-domain probe set, then the build HARD-FAILS if the scorecard misses the gate.

Reuses the core's reference runtime (serve_package) — the scorecard measures exactly what ships.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # py/ (core) on path
from serve_package import load_package, serve  # noqa: E402


def _W(ngrams):
    return max(ngrams.keys(), default=0)


def score(manifest_path, holdout, off_domain=()):
    """Grade the package at manifest_path.
      holdout    = [(ctx, gold), …]  in-domain held-out next-token pairs (the cover/n-grams should answer these).
      off_domain = [ctx, …]          out-of-domain contexts (the expert should ABSTAIN on these).
    Returns the scorecard dict (per-tier coverage/precision + overall + confident-wrong + off-domain-leak)."""
    idioms, ngrams, manifest = load_package(manifest_path)
    W = _W(ngrams)
    ans = cor = 0
    tiers = {}
    for ctx, gold in holdout:
        r = serve(list(ctx), idioms, ngrams, W)
        if r is None:
            continue
        ans += 1
        right = int(r["answer"] == gold)
        cor += right
        d = tiers.setdefault(r["tier"], {"answered": 0, "correct": 0})
        d["answered"] += 1
        d["correct"] += right
    n = len(holdout)
    leaked = sum(1 for ctx in off_domain if serve(list(ctx), idioms, ngrams, W) is not None)
    nod = len(off_domain)
    return {
        "n_rules": manifest.get("n_rules", len(manifest.get("rules", []))),
        "holdout_n": n,
        "coverage": (ans / n) if n else 0.0,
        "precision": (cor / ans) if ans else 0.0,
        "abstain": (1 - ans / n) if n else 1.0,
        "confident_wrong": ((ans - cor) / n) if n else 0.0,        # answered-but-wrong as a fraction of all (the hallucination rate)
        "off_domain_n": nod,
        "off_domain_leak": (leaked / nod) if nod else 0.0,         # answered (not abstained) on an off-domain probe
        "tiers": {t: {"coverage": d["answered"] / n if n else 0.0,
                      "precision": d["correct"] / d["answered"] if d["answered"] else 0.0}
                  for t, d in tiers.items()},
        "benchmarks": {},                                          # filled by a benchmark runner (slot; EXPERTS.md)
    }


def gate(sc, *, min_precision=None, max_leak=None, benchmarks=None):
    """Hard-fail check → (ok, reasons). Thresholds from the spec's [gate] + [[benchmark]] targets."""
    reasons = []
    if min_precision is not None and sc["precision"] < min_precision:
        reasons.append(f"precision {sc['precision']:.3f} < min_precision {min_precision}")
    if max_leak is not None and sc["off_domain_leak"] > max_leak:
        reasons.append(f"off_domain_leak {sc['off_domain_leak']:.3f} > max_leak {max_leak}")
    for b in (benchmarks or []):
        got = sc.get("benchmarks", {}).get(b.get("name"))
        if b.get("target") is not None and (got is None or got < b["target"]):
            reasons.append(f"benchmark {b.get('name')} {got} < target {b['target']}")
    return (not reasons, reasons)


def write_scorecard(sc, path):
    json.dump(sc, open(path, "w"), indent=1)
    return path
