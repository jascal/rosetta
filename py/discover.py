#!/usr/bin/env python3
"""rosetta · discover.py — rediscover a COMPOSED (arithmetic) circuit from behavior alone, no grammar, no answer key.

The hard, general part of minimization: on a real LLM we don't know the rule. This recovers one using only the faithful
model (whole.dl) and a corpus:
  1. LOCALIZE  — perturb each position of a decision context; load-bearing = flips the answer. Corpus-entropy then splits
                 load-bearing positions into OPERANDS (variable) vs FRAME (fixed) vs TRIGGER (flips for ~everything).
  2. READ      — sweep the corpus-discovered operand alphabet through whole.dl → the input→output behavior table.
  3. SEARCH    — brute-force integer labelings of the alphabet; if one makes the output depend only on the SUM, the
                 circuit is addition (recovered, not assumed).
Aborts LOUDLY rather than passing vacuously. The recovered rule is then certified by dl/equiv.dl (see verify_threx.py).
Usage: python3 py/discover.py
"""
import os, json, itertools, sys
from collections import Counter
from oracle import decide

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(HERE, "reference", "threx")
WHOLE = os.path.join(REF, "whole.dl")
LEXJ = json.load(open(os.path.join(REF, "lexicon.json")))
SYM = {i: t[0] for i, t in enumerate(LEXJ["tokens"])}
sy = lambda x: SYM.get(x, "∅" if x is None else str(x))
IDS = json.load(open(os.path.join(REF, "corpus.json")))["ids"]
VOCAB = LEXJ["vocab"]
BASE = [0, 20, 25, 25, 19, 19, 7]   # a composed decision context — a frequent frame, NOT hand-parsed
TAIL = [19, 19, 7]                   # the deterministic skeleton ending it (from suffix-mining)
TSTART = len(BASE) - len(TAIL)


def corpus_vals(p):
    """tokens at BASE-position p across corpus windows whose tail matches TAIL. BASE[p] ↔ ids[i-(TSTART-p)]."""
    vals, n = Counter(), len(TAIL)
    for i in range(len(IDS) - n):
        if IDS[i:i + n] == TAIL:
            j = i - (TSTART - p)
            if 0 <= j < len(IDS):
                vals[IDS[j]] += 1
    return vals


def die(msg):
    print(f"\n✗ ABORT: {msg}  (no vacuous success)"); sys.exit(1)


def main():
    base_out = decide(WHOLE, BASE)
    print(f"=== rediscover COMPOSED from behavior · {[sy(t) for t in BASE]} → {sy(base_out)} ===\n")

    print("1. causal localization (perturb each position):")
    flips = {p: sum(decide(WHOLE, BASE[:p] + [t] + BASE[p + 1:]) != base_out
                    for t in range(VOCAB) if t != BASE[p]) for p in range(len(BASE))}
    ent = {p: len(corpus_vals(p)) for p in range(len(BASE))}
    operands = []
    for p in range(len(BASE)):
        lb, var, trig = flips[p] >= 5, ent[p] >= 3, flips[p] >= VOCAB - 2
        kind = "TRIGGER" if trig and not var else ("OPERAND" if lb and var else ("frame" if lb else "inert"))
        if kind == "OPERAND":
            operands.append(p)
        print(f"   pos {p} ({sy(BASE[p])}): flips {flips[p]:2}/{VOCAB-1}  corpus-distinct {ent[p]}  → {kind}")
    print(f"   ⇒ operands: {operands} = {[sy(BASE[p]) for p in operands]}")
    if len(operands) != 2:
        die(f"expected two operands, got {len(operands)}")

    alpha = sorted(set(corpus_vals(operands[0])) & set(corpus_vals(operands[1])))
    print(f"\n2. operand alphabet (corpus): {[sy(a) for a in alpha]}")
    if len(alpha) < 3:
        die(f"operand alphabet too small ({len(alpha)})")
    print(f"   sweep {len(alpha)}×{len(alpha)} via whole.dl …")
    T = {(a, b): decide(WHOLE, BASE[:operands[0]] + [a] + BASE[operands[0] + 1:operands[1]] + [b] + BASE[operands[1] + 1:])
         for a in alpha for b in alpha}
    if len(set(T.values())) < 3:
        die(f"output varies over only {len(set(T.values()))} value(s)")

    print(f"\n3. search for additive structure ({len(set(T.values()))} distinct outputs over {len(T)} cells):")
    found = None
    for perm in itertools.permutations(range(len(alpha))):
        lab, bysum, ok = dict(zip(alpha, perm)), {}, True
        for (a, b), o in T.items():
            if bysum.setdefault(lab[a] + lab[b], o) != o:
                ok = False; break
        if ok and len(set(bysum.values())) >= 3:
            found = (lab, bysum); break
    if not found:
        die("no additive labeling fits — not a sum (try product/max/copy next)")
    lab, bysum = found
    strengths = {sy(a): lab[a] for a in alpha}
    print(f"   ✓ ADDITIVE. recovered strengths: {strengths}")
    print(f"   recovered sum→thing: {{{', '.join(f'{s}:{sy(o)}' for s, o in sorted(bysum.items()))}}}")

    true = {SYM[int(k)]: v for k, v in LEXJ["strength"].items()}
    fwd = all(strengths.get(s) == true.get(s) for s in strengths)
    rev = all(strengths.get(s) == (len(alpha) - 1 - true.get(s)) for s in strengths)
    ok = len(strengths) == len(alpha) and (fwd or rev)
    print(f"\n4. validation vs hidden true strengths {true}: {'✓ matches (up to reversal)' if ok else '✗ MISMATCH'}")
    if not ok:
        die("recovered labeling does not match the true strengths")
    print("\n→ composed arithmetic recovered from behavior + corpus alone; grammar not used. Now certify with equiv.dl.")


if __name__ == "__main__":
    main()
