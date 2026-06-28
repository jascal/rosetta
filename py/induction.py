#!/usr/bin/env python3
"""rosetta · induction.py — run the Datalog induction detector and report how much the model copies in-context.

Reports, per match length m: the follow-rate (of contexts where the m-suffix recurs in-window, how often the model
copies what followed) and coverage. Then the DISTINCTIVE part: induction hits on contexts the n-gram cover could only
resolve at long order (≥4) — i.e. decisions recall can't compress but copying explains. Usage: python3 py/induction.py [n] [w] [model_dir]
"""
import os, sys, json
from minimize import instances, get_refs
from oracle import run_induction, detect

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    md = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "reference", "threx")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    name = os.path.basename(md.rstrip("/"))
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    insts = instances(ids, n, w)
    refs = get_refs(os.path.join(md, "whole.dl"), insts, os.path.join(md, "ref_cache.json"))
    valid = [insts[i] for i in range(len(insts)) if refs[i] is not None]
    vrefs = [refs[i] for i in range(len(insts)) if refs[i] is not None]

    print(f"=== {name}: induction/copy detection (dl/induction.dl) · {len(valid)} decisions · window {w} ===")
    print(f"{'m':>2}  {'applies':>8}  {'follow (hit/apply)':>20}  {'coverage (hit/total)':>22}")
    best = None
    for m in (1, 2, 3):
        r = run_induction(valid, vrefs, m)
        ap, hit, tot = r["n_apply"], r["n_hit"], r["n_total"]
        fr = f"{hit}/{ap} ({100*hit/ap:.0f}%)" if ap else "—"
        print(f"{m:>2}  {ap:>8}  {fr:>20}  {f'{hit}/{tot} ({100*hit/tot:.0f}%)':>22}")
        if m == 2:
            best = r

    # distinctive induction: copies the n-gram cover can't compress (model needed order ≥4 there)
    _, minord = detect(valid, vrefs, w)
    tail = {i for i, k in minord.items() if k >= 4}
    if best and tail:
        dist = best["hits"] & tail
        print(f"\ndistinctive induction (m=2 hits on the n-gram tail, order ≥4): {len(dist)}/{len(tail)} "
              f"of long-order contexts ({100*len(dist)/len(tail):.0f}%) are explained by copying, not recall.")
    print("\n(In-window copy only — contexts are W-token; long-range induction needs a larger W. The detector is the"
          " deliverable; on a real model the follow-rate and the tail-coverage are where computation shows.)")


if __name__ == "__main__":
    main()
