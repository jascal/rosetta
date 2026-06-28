#!/usr/bin/env python3
"""rosetta · holdout_score.py — does the SKELETON idiom actually generalize, or are we just memoizing?

The honest test of "is it us, not the model". Split the corpus windows train/holdout; build the cover on TRAIN ONLY;
predict HELDOUT windows (which the cover never saw) and compare to the model's refs. n-gram rules generalize only when a
held-out window's lexical suffix recurred in train; SKELETON rules also fire when the held-out window shares a *syntactic
frame* with train but has new content. The delta — held-out windows the skeleton gets right that n-grams miss — is
exactly the cross-content generalization our lexical cover was leaving on the table (= "us memoizing", now recovered).
Precedence mirrors fieldrun: lexical n-gram (specific) wins; skeleton (abstracted) is the fallback; else abstain.
Usage: python3 py/holdout_score.py [n] [w] [model_dir] [hold_frac] [closed_n]
"""
import os, sys, json, random
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimize import instances, model_refs, minimal_suffix_cover

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = -1


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    md = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "reference", "threx")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    hold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    closed_n = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    name = os.path.basename(md.rstrip("/"))
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    insts = instances(ids, n, w)
    refs = model_refs(md, insts)
    idxs = [i for i in range(len(insts)) if refs[i] is not None and len(insts[i]) >= w]
    rng = random.Random(0)
    shuf = idxs[:]; rng.shuffle(shuf)
    cut = int(len(shuf) * (1 - hold))
    train, holdout = shuf[:cut], shuf[cut:]

    ng = minimal_suffix_cover(insts, refs, train, w)[0]                # {suffix: out} from TRAIN
    def ng_pred(ctx):
        for k in range(min(len(ctx), w), 0, -1):
            o = ng.get(tuple(ctx[-k:]))
            if o is not None:
                return o
        return None

    closed = {t for t, _ in Counter(t for i in train for t in insts[i]).most_common(closed_n)}   # closed class from TRAIN
    skel = lambda ctx, k: tuple(t if t in closed else O for t in ctx[-k:])
    skr = {}
    for k in range(1, w + 1):
        g = defaultdict(list)
        for i in train:
            if len(insts[i]) >= k:
                g[skel(insts[i], k)].append(i)
        for sk, mem in g.items():
            if O in sk and len({refs[i] for i in mem}) == 1 and len({tuple(insts[i][-k:]) for i in mem}) >= 2:
                skr[sk] = refs[mem[0]]
    def sk_pred(ctx):
        for k in range(min(len(ctx), w), 0, -1):
            o = skr.get(skel(ctx, k))
            if o is not None:
                return o
        return None

    def score(predfns):
        cov = corr = 0
        for i in holdout:
            p = None
            for f in predfns:
                p = f(insts[i])
                if p is not None:
                    break
            if p is not None:
                cov += 1; corr += (p == refs[i])
        return cov, corr

    H = len(holdout)
    c1, k1 = score([ng_pred])
    c2, k2 = score([ng_pred, sk_pred])
    print(f"=== holdout_score · {name} · {len(train)} train / {H} holdout windows (W={w}, {len(ng)} n-gram + {len(skr)} skeleton rules) ===")
    print(f"  n-gram only : covers {c1/H:.0%}  correct {k1}/{H} = {k1/H:.0%}   (holdout loss {1-k1/H:.0%})")
    print(f"  + skeleton  : covers {c2/H:.0%}  correct {k2}/{H} = {k2/H:.0%}   (holdout loss {1-k2/H:.0%})")
    print(f"  → skeleton recovers +{k2-k1} held-out windows ({(k2-k1)/H:+.1%}) that lexical n-grams missed"
          + ("  ← cross-content generalization (was 'us memoizing')" if k2 > k1 else "  (no gain — redundant with n-grams here)"))


if __name__ == "__main__":
    main()
