#!/usr/bin/env python3
"""rosetta · induction_package.py — wire a model's CAUSALLY-CONFIRMED induction circuit into an expert package, and
measure its coverage on HELD-OUT induction stimuli.

The counterpart to the model-distilled ANSWER tier (EXPERTS.md logic ablation): here the SMART tier is a real model
CIRCUIT. On a natural corpus `idiom_learn` finds 0 gate/compose idioms and induction stays masked (IDIOM_LEARNER.md),
so this exercises the circuit with repeated NOVEL sequences S+S — where the only way to predict the repeat is to COPY —
and confirms it CAUSALLY (perturb the copied-from token; a real induction head follows it). The confirmed circuit is
emitted by `emit_expert_package` as a first-class `induction` rule in manifest.json (was souffle-only OOD before) and
served by `serve_package`.

The decisive measurement is on a DISJOINT held-out set of novel-repeat sequences: the n-gram tier has no support there
(unseen tokens), so a non-zero coverage is the induction rule GENERALIZING, not memorizing. We report the served tier
split WITH induction vs n-gram-only — the induction rule's marginal, generalizing contribution.

Needs a fieldrun bundle (the build-time oracle). Usage:
    .venv/bin/python py/induction_package.py [bundle-stem] [n_seqs] [seqlen] [port]
    # e.g. .venv/bin/python py/induction_package.py models/pythia160m/bundle 30 20 8199
"""
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oracle import serve_decide  # noqa: E402
from idiom_learn import emit_expert_package  # noqa: E402
from serve_package import load_package, serve  # noqa: E402

FIELDRUN = os.environ.get("FIELDRUN", os.path.join(HERE, "..", "..", "fieldrun", "target", "release", "fieldrun"))
W, VLO, VHI = 8, 200, 40000


def gen(seeds, seqlen):
    """Repeated-novel stimuli S + S[:k]: predicting S[k] is reachable only by copying the first occurrence."""
    insts = []
    for seed in seeds:
        r = random.Random(seed)
        s = [r.choice(range(VLO, VHI)) for _ in range(seqlen)]
        for k in range(1, seqlen - 1):
            insts.append(s + s[:k])
    return insts


def confirm_induction(insts, refs, idxs, dec, fill):
    """Detect + CAUSALLY confirm induction on TRAIN. Returns the admitted rels ([{L, causal, obs}], causal>=.8 obs>=.5)."""
    rels = []
    for L in (1, 2, 3):
        app, hit = [], 0
        for i in idxs:
            ctx = insts[i]
            suf = tuple(ctx[-L:])
            js = [j for j in range(len(ctx) - L) if tuple(ctx[j:j + L]) == suf]
            if js and max(js) + L < len(ctx):
                app.append((i, max(js) + L))
                hit += (ctx[max(js) + L] == refs[i])
        if not app:
            continue
        obs = hit / len(app)
        rng = random.Random(1)
        tests = []
        for i, ptr in app[:100]:                                 # causal: perturb the copied-from token
            ctx = insts[i][:]
            xp = rng.choice(range(VLO, VHI))
            while xp == insts[i][ptr]:
                xp = rng.choice(range(VLO, VHI))
            ctx[ptr] = xp
            tests.append((ctx, xp))
        fill([c for c, _ in tests])
        causal = sum(1 for c, xp in tests if dec(c) == xp) / len(tests)
        real = causal >= 0.8 and obs >= 0.5
        print(f"  induction L={L}: applicable {len(app)}  obs {obs:.0%}  CAUSAL {causal:.0%}{'  ← REAL' if real else ''}")
        if real:
            rels.append({"L": L, "causal": causal, "obs": obs})
    return rels


def serve_holdout(insts, refs, idxs, idioms, ngrams):
    """Serve held-out contexts; return (coverage, correct, abstain, induction_fired, induction_correct)."""
    ans = cor = ab = ind_n = ind_ok = 0
    for i in idxs:
        r = serve(insts[i], idioms, ngrams, W)
        if r is None:
            ab += 1
            continue
        ok = (r["answer"] == refs[i])
        ans += 1
        cor += ok
        if r.get("circuit") == "induction":
            ind_n += 1
            ind_ok += ok
    return ans, cor, ab, ind_n, ind_ok


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else "models/pythia160m/bundle"
    n_seqs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seqlen = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 8199
    if not os.path.exists(FIELDRUN):
        sys.exit(f"fieldrun not found: {FIELDRUN} (set $FIELDRUN)")

    proc = subprocess.Popen([FIELDRUN, "--bundle", bundle, "--serve", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        up = False
        for _ in range(60):
            if proc.poll() is not None:
                sys.exit(f"fieldrun --serve exited early (code {proc.returncode}) — bad bundle stem '{bundle}'?")
            try:
                serve_decide(port, [1, 2, 3])
                up = True
                break
            except Exception:
                time.sleep(1)
        if not up:
            sys.exit("oracle did not come up")

        cache = {}

        def fill(ctxs):
            miss = list({tuple(c) for c in ctxs} - set(cache))
            with ThreadPoolExecutor(max_workers=2) as ex:        # a resident server is sequential
                for c, o in zip(miss, ex.map(lambda t: serve_decide(port, list(t)), miss)):
                    cache[c] = o

        def dec(ctx):
            k = tuple(ctx)
            if k not in cache:
                cache[k] = serve_decide(port, ctx)
            return cache[k]

        train, hold = gen(range(n_seqs), seqlen), gen(range(100, 100 + n_seqs), seqlen)
        fill(train + hold)
        tr_refs, ho_refs = [dec(c) for c in train], [dec(c) for c in hold]
        tr_idx = [i for i in range(len(train)) if tr_refs[i] is not None]
        ho_idx = [i for i in range(len(hold)) if ho_refs[i] is not None]
        print(f"=== induction_package · {os.path.basename(bundle)} · train {len(tr_idx)} · holdout {len(ho_idx)} ===\n"
              "confirm induction (causal) on TRAIN:")
        rels = confirm_induction(train, tr_refs, tr_idx, dec, fill)
        if not rels:
            sys.exit("no induction confirmed — nothing to wire in")

        md = os.path.join(HERE, "..", "models", os.path.basename(os.path.dirname(bundle)) or "induction_pkg")
        _, man = emit_expert_package(md, train, tr_refs, tr_idx, [], [], rels, W, os.path.basename(bundle), 3, 1.0)
        idioms, ngrams, m = load_package(man)
        print(f"[manifest] {len(m['rules'])} rules — trusted_idioms={m['trusted_idioms']} "
              f"gated_ngrams={m['gated_ngrams']} induction_ood={m.get('induction_ood')} "
              f"(induction rules in the served manifest: {sum(1 for r in idioms if r['kind'] == 'induction')})")

        print("\nHELD-OUT (novel-repeat, disjoint seeds — n-grams have no support here):")
        for label, idi in [("WITH induction", idioms), ("n-gram only", [r for r in idioms if r['kind'] != 'induction'])]:
            ans, cor, ab, ind_n, ind_ok = serve_holdout(hold, ho_refs, ho_idx, idi, ngrams)
            H = len(ho_idx)
            extra = f"  · induction fired {ind_n} correct {ind_ok} ({ind_ok / ind_n if ind_n else 0:.0%})" if "WITH" in label else ""
            print(f"  [{label:<14}] coverage {ans}/{H}={ans / H:.0%}  precision {cor}/{ans}="
                  f"{cor / ans if ans else 0:.0%}  abstain {ab}/{H}={ab / H:.0%}{extra}")
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
