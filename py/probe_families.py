#!/usr/bin/env python3
"""rosetta · probe_families.py — probe-driven symbolic-family detection (the semantic tier of the toolkit).

Some families (copy/name-mover, coreference, succession, the ergo reasoning families) are entangled with content in a
natural corpus, so we ISOLATE them with templated/nonce stimuli + a foil, the way fieldrun's IOI work and ergo's probes
do. Toolkit-inclusion rule (per user): a family is IN the toolkit if we can DETECT it (the model follows the rule) AND
show it's CAUSAL (perturb the operand → the output follows) — whether or not any given model/dataset strictly needs it.
(Per-model COVER admission is the separate holdout+MDL question.)

Uses the model's tokenizer (rosetta .venv) + a resident `fieldrun --serve` server (FIELDRUN_SERVE).
Usage: FIELDRUN_SERVE=<port> python3 py/probe_families.py <model_dir> [n]
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import serve_decide
from tokenizers import Tokenizer

NAMES = [" John", " Mary", " Tom", " Sara", " Paul", " Anna", " Mark", " Lucy", " Mike", " Emma", " David", " Kate",
         " James", " Laura", " Peter", " Alice", " Henry", " Julia", " Robert", " Nancy"]
DAYS = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"]
MONTHS = [" January", " February", " March", " April", " May", " June", " July", " August", " September", " October"]


def single(tok, words):
    """keep only words that are a single CONTENT token (no special/BOS), so the next-token prediction is the whole word."""
    out = {}
    for wd in words:
        ids = tok.encode(wd, add_special_tokens=False).ids
        if len(ids) == 1:
            out[wd] = ids[0]
    return out


def family_ioi(tok, port, n=40):
    """copy / name-mover (IOI): 'When A and B went…, B gave a drink to' → A (the once-mentioned name). Causal: swap the
    repeated name → the answer follows to the other name (the structure, not a fixed position, drives the copy)."""
    nm = single(tok, NAMES)
    names = list(nm)
    if len(names) < 4:
        return None
    rng = random.Random(0)
    det = cfollow = trials = 0
    for _ in range(n):
        a, b = rng.sample(names, 2)
        o1 = serve_decide(port, tok.encode(f"When{a} and{b} went to the store,{b} gave a drink to").ids)
        o2 = serve_decide(port, tok.encode(f"When{b} and{a} went to the store,{a} gave a drink to").ids)
        det += (o1 == nm[a]); cfollow += (o2 == nm[b]); trials += 1
    return det / trials, cfollow / trials


def family_succession(tok, port, seq, label, n=30):
    """succession / greater-than: an ordered run 'X Y Z' → the next item. Causal: shift the window's start → the
    predicted successor shifts with it (the model tracks ordinal position, not a memorized token)."""
    s = single(tok, seq)
    items = [w for w in seq if w in s]
    if len(items) < 5:
        return None
    rng = random.Random(1)
    det = cfollow = trials = 0
    for _ in range(n):
        i = rng.randint(0, len(items) - 4)
        o1 = serve_decide(port, tok.encode(f"{items[i]}{items[i+1]}{items[i+2]}").ids)
        det += (o1 == s[items[i + 3]])
        j = (i + 1) % (len(items) - 3)                              # shift the window → successor must shift
        o2 = serve_decide(port, tok.encode(f"{items[j]}{items[j+1]}{items[j+2]}").ids)
        cfollow += (o2 == s[items[j + 3]]); trials += 1
    return det / trials, cfollow / trials


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else "models/llama32_1b"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    port = int(os.environ["FIELDRUN_SERVE"])
    tok = Tokenizer.from_file(os.path.join(md, "bundle.tokenizer.json"))
    name = os.path.basename(md.rstrip("/"))
    print(f"=== probe_families · {name} === (detect ≥80% AND causal ≥80% ⇒ in the toolkit)")
    probes = [("copy/name-mover (IOI)", lambda: family_ioi(tok, port, n)),
              ("succession (days)", lambda: family_succession(tok, port, DAYS, "days", n)),
              ("succession (months)", lambda: family_succession(tok, port, MONTHS, "months", n))]
    for label, fn in probes:
        r = fn()
        if r is None:
            print(f"  {label:28} — skipped (stimuli not single-token in this tokenizer)")
            continue
        det, caus = r
        verdict = "  ← IN TOOLKIT" if det >= 0.8 and caus >= 0.8 else ("  (causal but weak detect)" if caus >= 0.8 else "")
        print(f"  {label:28} detect {det:.0%}  causal {caus:.0%}{verdict}")


if __name__ == "__main__":
    main()
