#!/usr/bin/env python3
"""rosetta · ngram.py — run the Datalog n-gram detector (dl/ngram.dl) and decode the result.

The detection is Datalog (dl/ngram.dl): shortest model-deterministic suffix per context = a certified (k+1)-gram. This
script only STAGES the facts (the contexts, and the model's argmax per context from whole.dl) and DECODES the output for
humans — no mining logic in Python. Usage: python3 py/ngram.py [n] [w] [model_dir]
"""
import os, sys, json
from minimize import instances, model_refs
from oracle import detect   # the detection itself lives in dl/ngram.dl, driven from oracle.py

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    md = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "reference", "threx")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    name = os.path.basename(md.rstrip("/"))
    sym = {i: t[0] for i, t in enumerate(json.load(open(os.path.join(md, "lexicon.json")))["tokens"])}
    dec = lambda t: (sym.get(t, str(t)).strip() or f"[{t}]")
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    insts = instances(ids, n, w)
    refs = model_refs(md, insts)

    hist, minord = detect(insts, refs, w)        # ← the detection happens in Datalog
    tot = max(1, sum(hist.values()))
    print(f"=== {name}: n-gram detection via dl/ngram.dl · {tot} decisions · window {w} ===")
    print(f"order histogram (contexts): {dict(sorted(hist.items()))}")
    cum = 0
    for o in sorted(hist):
        cum += hist[o]
        print(f"   ≤{o}-gram: {cum}/{tot} ({100*cum/tot:.0f}%)")
    eff = next((o for o in sorted(hist) if sum(hist[k] for k in hist if k <= o) >= 0.9 * tot), max(hist, default=0))
    print(f"effective order (≥90%): {eff}-gram"
          f"{'  → bigram/trigram-dominated (recall, not computation)' if eff <= 3 else ''}\n")

    # decode the detected n-grams (presentation only): each resolved context's k-suffix → its model argmax
    for k, label in ((1, "BIGRAM  (after X → Y)"), (2, "TRIGRAM (after X Y → Z)")):
        seen = {}
        for i, ki in minord.items():
            if ki == k:
                seen.setdefault(tuple(insts[i][-k:]), refs[i])
        ex = list(seen.items())[:14]
        print(f"{label} — {len(seen)} detected, certified (prefix-invariant over {w-k} prior tokens):")
        for suf, out in ex:
            print(f"   {' '.join(dec(t) for t in suf):<22} → {dec(out)}")
        print()


if __name__ == "__main__":
    main()
