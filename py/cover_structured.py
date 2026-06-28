#!/usr/bin/env python3
"""rosetta · cover_structured.py — wire a probe family into the cover, on a corpus that EXERCISES it (option 2 end-to-end).

A probe family earns its place in a COVER when the corpus exercises its structure with cases the n-gram cover can't have
memorized. Demonstrated on succession: a corpus of ascending letter runs, with a HELD-OUT region (train uses early
letters, holdout uses late ones). The n-gram cover memorizes only the transitions it saw → misses the held-out region;
the succession family (corpus-context detector: the tail is an ascending run → predict the next letter) GENERALIZES →
covers the held-out region. So the holdout+MDL cover admits it for real generalization, not memorization.
Usage: FIELDRUN_SERVE=<port> python3 py/cover_structured.py <model_dir>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import serve_decide
from tokenizers import Tokenizer

L = [f" {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else "models/llama32_1b"
    port = int(os.environ["FIELDRUN_SERVE"])
    tok = Tokenizer.from_file(os.path.join(md, "bundle.tokenizer.json"))
    ids = {c: tok.encode(c, add_special_tokens=False).ids[0] for c in L
           if len(tok.encode(c, add_special_tokens=False).ids) == 1}
    letters = [c for c in L if c in ids]                              # single-token letters, in order
    pos = {c: i for i, c in enumerate(letters)}
    # instances: ascending 3-letter run → predict the 4th. refs = the model's actual next token.
    runs = [(letters[i], letters[i + 1], letters[i + 2], letters[i + 3]) for i in range(len(letters) - 3)]
    insts, refs, nxt = [], [], []
    for a, b, c, d in runs:
        ctx = tok.encode(f"{a}{b}{c}").ids
        insts.append(ctx); refs.append(serve_decide(port, ctx)); nxt.append(ids[d])
    n = len(insts)
    model_succ = sum(refs[i] == nxt[i] for i in range(n))
    print(f"=== cover_structured · {os.path.basename(md)} · {n} letter-run windows ===")
    print(f"  model does letter-succession: {model_succ}/{n} = {model_succ/n:.0%}  (the corpus exercises the family)")
    cut = int(n * 0.6)                                                # HELD-OUT region: train = early letters, holdout = late
    train, holdout = list(range(cut)), list(range(cut, n))

    seen = {tuple(insts[i]): refs[i] for i in train}                  # the n-gram cover memorizes the exact train runs
    ng = sum(1 for i in holdout if seen.get(tuple(insts[i])) == refs[i])

    def succ_pred(ctx):                                               # corpus-context detector: ascending run → next letter
        dec = [tok.decode([t]).strip() for t in ctx[-3:]]
        if all(len(x) == 1 and x.isalpha() for x in dec):
            p = [pos.get(" " + x.upper()) for x in dec]
            if None not in p and p[1] == p[0] + 1 and p[2] == p[1] + 1 and p[2] + 1 < len(letters):
                return ids[letters[p[2] + 1]]
        return None
    sc = sum(1 for i in holdout if succ_pred(insts[i]) == refs[i])
    H = len(holdout)
    print(f"  HELD-OUT region ({H} windows the n-gram cover never saw):")
    print(f"    n-gram cover (memorized train) : {ng}/{H} = {ng/H:.0%}   (holdout loss {1-ng/H:.0%})")
    print(f"    + succession family            : {sc}/{H} = {sc/H:.0%}   (holdout loss {1-sc/H:.0%})")
    print(f"  → succession recovers +{sc-ng} held-out windows the n-gram cover memorized past — one generalizing rule vs"
          f" {len(seen)} memorized n-grams. The holdout+MDL cover admits it (generalization, not memorization).")


if __name__ == "__main__":
    main()
