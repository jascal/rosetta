#!/usr/bin/env python3
"""rosetta · verify_threx.py — certify the threx COMPOSED circuit against the model, the verdict computed in Datalog.

Builds the 25 composed instances (`⟨ ∿ Bi Bj · · gɪ` over all bearing pairs), reads each instance's answer off the
faithful whole.dl (the oracle), and lets dl/equiv.dl PROVE circuit == model over the whole 25-cell domain. This is the
keystone the rest of rosetta scales: a transform/circuit is trusted only when this Datalog certificate is clean.
Usage: python3 py/verify_threx.py
"""
import os
from oracle import certify

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(HERE, "reference", "threx")
WHOLE = os.path.join(REF, "whole.dl")
CIRCUIT = os.path.join(REF, "circuit.dl")
BEARINGS = [21, 22, 23, 24, 25]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", help="retain replayable verifier inputs and outputs here")
    args = parser.parse_args()
    instances = [[0, 20, bi, bj, 19, 19, 7] for bi in BEARINGS for bj in BEARINGS]  # all 25 bearing pairs
    print(f"=== rosetta · certify threx COMPOSED circuit vs whole.dl · {len(instances)} instances ===")
    r = certify(CIRCUIT, WHOLE, instances, evidence_dir=args.evidence_dir)
    if "error" in r:
        print("ERROR:", r["error"]); return 1
    print(f"  instances checked (ncover): {r['ncover']}")
    print(f"  disagreements (nmiss)     : {r['nmiss']}")
    print(f"  gaps (nuncov)             : {r['nuncov']}")
    if "evidence" in r:
        print(f"  replayable evidence      : {r['evidence']}")
    if r["mismatches"]:
        print("  mismatches (inst, model, circuit):", r["mismatches"][:8])
    ok = r["certified"]
    print(f"\n  CERTIFIED (Datalog): {ok}  — circuit is provably equivalent to the model over its whole domain"
          if ok else f"\n  NOT certified: {r}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
