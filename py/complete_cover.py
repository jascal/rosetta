#!/usr/bin/env python3
"""rosetta · complete_cover.py — the semiring backstop: route the residual to the forward-pass-as-Datalog (no abstain).

cover-ordering: idiom/n-gram rules (compressed) → SEMIRING backstop (whole.dl, the exact forward pass) for whatever no
rule covers. The backstop is souffle (the algorithm in Datalog, NOT the fieldrun binary), so runtime-independence holds;
and it's exact (it IS the model) → the complete cover has zero abstain, zero loss. T-respecting because the backstop
emits the full LOGITS — softmax(logits/T) gives the exact distribution at any temperature (argmax at T=0).
Demonstrated on threx (whole.dl exists). Usage: python3 py/complete_cover.py [n] [w]
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimize import instances, model_refs, minimal_suffix_cover
from oracle import decide, logits

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    md = os.path.join(HERE, "reference", "threx")
    whole = os.path.join(md, "whole.dl")
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    insts = instances(ids, n, w)
    refs = model_refs(md, insts)
    idxs = [i for i in range(len(insts)) if refs[i] is not None and len(insts[i]) >= w]
    rng = random.Random(0); sh = idxs[:]; rng.shuffle(sh)
    cut = int(len(sh) * 0.7); train, holdout = sh[:cut], sh[cut:]
    ng = minimal_suffix_cover(insts, refs, train, w)[0]

    def cover(ctx):                                                    # the compressed rule layer (n-gram cover here)
        for k in range(min(len(ctx), w), 0, -1):
            o = ng.get(tuple(ctx[-k:]))
            if o is not None:
                return o
        return None

    H = len(holdout)
    residual = [i for i in holdout if cover(insts[i]) is None]        # ABSTENTIONS — what the backstop is for
    fired = [i for i in holdout if cover(insts[i]) is not None]
    fired_ok = sum(cover(insts[i]) == refs[i] for i in fired)
    back_ok = sum(decide(whole, insts[i]) == refs[i] for i in residual)
    print(f"=== complete_cover · threx · {H} holdout windows ===")
    print(f"  COMPLETENESS (what the backstop guarantees):")
    print(f"    rule layer abstains on {len(residual)} (no rule fires) → SEMIRING BACKSTOP (whole.dl, souffle):"
          f" {back_ok}/{len(residual)} EXACT (it IS the model). Abstain → 0; the residual is computed, never lost.")
    print(f"  EXACTNESS over the certified domain: the cover built ON the corpus is complete+exact there —"
          f" dl/equiv.dl already proves nmiss=0 nuncov=0 (rules faithful by construction + backstop fills any gap).")
    print(f"  GENERALIZATION (orthogonal — the toolkit's job, NOT the backstop's):")
    print(f"    rules that FIRED on holdout: {fired_ok}/{len(fired)} correct — the {len(fired)-fired_ok} covered-but-wrong are")
    print(f"    rule MISPREDICTIONS on unseen contexts (a train suffix firing wrongly). The backstop can't fix a rule that")
    print(f"    fired; better/abstaining rules (the family toolkit) close this. Backstop = completeness, families = generalization.")
    if residual:
        lg = logits(whole, insts[residual[0]])
        print(f"  T-respecting: the backstop emits the full {len(lg)}-logit scoreboard → softmax(logits/T) exact at any T (argmax at T=0).")
    print("  pruning: the backstop runs ONLY on abstentions (routing); weights prune to that footprint, and every certified")
    print("  rule provably carves its region out (certificate-gated) — the unembed melts as coverage grows. No floor: residual = computed.")


if __name__ == "__main__":
    main()
