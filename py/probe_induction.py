#!/usr/bin/env python3
"""rosetta · probe_induction.py — measure a model's COPY/INDUCTION circuit, isolated from the n-gram confound.

Induction (the canonical transformer circuit) copies "the token following the previous occurrence of the current token".
Its OBSERVATIONAL rate on natural text is confounded by n-gram determinism (a recurring suffix yields the same next
token for n-gram reasons, not copying). To isolate TRUE induction we use repeated NOVEL (random) token sequences S+S:
there is no n-gram to lean on, so only a copy circuit can predict the repeat. And we confirm CAUSALLY — perturb the
copied-from token; a real induction head makes the output follow it.

This is the build-time refs/perturbation oracle (fieldrun bundle), same as the idiom learner. On pythia-160m (known to
have induction heads): observational ~80%, CAUSAL ~85-90%. On a model without the circuit it stays near 0.
Usage: FIELDRUN_BIN=… python3 py/probe_induction.py <bundle-stem> [n_seqs] [seqlen]
"""
import sys, os, random
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import fieldrun_decide, serve_decide

_SERVE = os.environ.get("FIELDRUN_SERVE")   # a resident `fieldrun --serve <port>` — loads the bundle once (big models)

VLO, VHI = 200, 40000   # sample novel tokens from a mid-vocab range (avoids specials / very rare ids)


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else "models/pythia160m/bundle"
    n_seqs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    seqlen = int(sys.argv[3]) if len(sys.argv) > 3 else 22
    cache = {}

    def dec(ctx):
        k = tuple(ctx)
        if k not in cache:
            cache[k] = serve_decide(int(_SERVE), ctx) if _SERVE else fieldrun_decide(bundle, ctx)
        return cache[k]

    def fill(ctxs, workers=8):
        miss = list({tuple(c) for c in ctxs} - set(cache))
        if not miss:
            return
        d = (lambda t: serve_decide(int(_SERVE), list(t))) if _SERVE else (lambda t: fieldrun_decide(bundle, list(t)))
        with ThreadPoolExecutor(max_workers=(2 if _SERVE else workers)) as ex:   # server is sequential; few clients suffice
            for c, o in zip(miss, ex.map(d, miss)):
                cache[c] = o

    # repeated NOVEL sequences S + S[:k]: predict S[k], reachable only by copying the first occurrence
    insts = []
    for seed in range(n_seqs):
        r = random.Random(seed)
        S = [r.choice(range(VLO, VHI)) for _ in range(seqlen)]
        for k in range(1, seqlen - 1):
            insts.append(S + S[:k])
    fill(insts)
    refs = [dec(c) for c in insts]
    idxs = [i for i in range(len(insts)) if refs[i] is not None]
    print(f"=== probe_induction · {os.path.basename(bundle)} · {len(idxs)} repeated-novel instances ===\n")

    for L in (1, 2, 3):
        app, hit = [], 0
        for i in idxs:
            ctx = insts[i]
            suf = tuple(ctx[-L:])
            js = [j for j in range(len(ctx) - L) if tuple(ctx[j:j + L]) == suf]
            if js:
                ptr = max(js) + L
                if ptr < len(ctx):
                    app.append((i, ptr)); hit += (ctx[ptr] == refs[i])
        if not app:
            continue
        rng = random.Random(1)
        tests = []
        for i, ptr in app[:100]:
            ctx = insts[i]; xp = rng.choice(range(VLO, VHI))
            while xp == ctx[ptr]:
                xp = rng.choice(range(VLO, VHI))
            p = ctx[:]; p[ptr] = xp; tests.append((p, xp))
        fill([p for p, _ in tests])
        causal = sum(1 for p, xp in tests if dec(p) == xp) / len(tests)
        real = causal >= 0.8 and hit / len(app) >= 0.5
        print(f"  induction L={L}: applicable {len(app)}  observational {hit/len(app):.0%}  "
              f"CAUSAL {causal:.0%}{'  ← REAL copy/induction circuit' if real else ''}")
    print("\n(observational alone is n-gram-confoundable; the causal perturbation is the proof of a copy circuit.)")


if __name__ == "__main__":
    main()
