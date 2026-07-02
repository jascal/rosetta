#!/usr/bin/env python3
"""rosetta · circuit_prevalence.py — does a confirmed circuit help on NATURAL in-domain text, or only on synthetic
probes? The experiment that gates the OOD-circuit line (see IDIOM_LEARNER.md).

For a fixed model, over each natural corpus: build the n-gram memorization tier on a TRAIN split (corpus continuations,
confident rules), then on a HELD-OUT split measure — of the positions where the n-gram tier ABSTAINS — how often the
causally-confirmed induction circuit (a) fires and (b) matches the MODEL's argmax (faithful marginal coverage). The
contrast is the signal: a code corpus (identifier/bracket copying = induction-dense) vs a prose corpus (little exact
repetition), SAME model, so the variable is domain STRUCTURE, not the model.

Usage: .venv/bin/python py/circuit_prevalence.py <bundle-stem> <code.py> <prose.txt> [port] [n_holdout]
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
from abstain_emit import build_tab, confident_rules  # noqa: E402

FIELDRUN = os.environ.get("FIELDRUN", os.path.join(HERE, "..", "..", "fieldrun", "target", "release", "fieldrun"))
LCTX, W, MINSUPP, MINDET = 32, 8, 3, 1.0                     # induction scan window / n-gram suffix / cover gates
VLO, VHI = 200, 40000


def induction_pred(ctx, Ls):
    """Longest-L-first induction: copy the successor of the current L-suffix's most recent earlier occurrence."""
    for L in sorted(Ls, reverse=True):
        if len(ctx) > L:
            suf = tuple(ctx[-L:])
            js = [j for j in range(len(ctx) - L) if tuple(ctx[j:j + L]) == suf]
            if js and max(js) + L < len(ctx):
                return ctx[max(js) + L]
    return None


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else "models/qwen25coder15b/bundle"
    code_path = sys.argv[2] if len(sys.argv) > 2 else "/usr/lib/python3.12/json/__init__.py"
    prose_path = sys.argv[3] if len(sys.argv) > 3 else "examples/logic/logic_kb.txt"
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 8195
    n_hold = int(sys.argv[5]) if len(sys.argv) > 5 else 400
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(bundle + ".tokenizer.json")

    proc = subprocess.Popen([FIELDRUN, "--bundle", bundle, "--serve", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        up = False
        for _ in range(90):
            if proc.poll() is not None:
                sys.exit(f"fieldrun --serve exited early (code {proc.returncode})")
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
            with ThreadPoolExecutor(max_workers=2) as ex:
                for c, o in zip(miss, ex.map(lambda t: serve_decide(port, list(t)), miss)):
                    cache[c] = o

        def dec(ctx):
            k = tuple(ctx)
            if k not in cache:
                cache[k] = serve_decide(port, ctx)
            return cache[k]

        # 1. CONFIRM induction on this model (novel-repeat S+S) — which L's are causally real (>=0.8).
        rng = random.Random(0)
        seqs = [[rng.choice(range(VLO, VHI)) for _ in range(18)] for _ in range(20)]
        stim = [s + s[:k] for s in seqs for k in range(1, 17)]
        fill(stim)
        Ls = []
        print("confirm induction on the model (novel-repeat):")
        for L in (1, 2, 3):
            app, hit = [], 0
            for ctx in stim:
                suf = tuple(ctx[-L:])
                js = [j for j in range(len(ctx) - L) if tuple(ctx[j:j + L]) == suf]
                if js and max(js) + L < len(ctx):
                    app.append((ctx, max(js) + L))
                    hit += (ctx[max(js) + L] == dec(ctx))
            if not app:
                continue
            obs = hit / len(app)
            tests = []
            for ctx, ptr in app[:80]:
                c = ctx[:]
                xp = rng.choice(range(VLO, VHI))
                while xp == ctx[ptr]:
                    xp = rng.choice(range(VLO, VHI))
                c[ptr] = xp
                tests.append((c, xp))
            fill([c for c, _ in tests])
            causal = sum(1 for c, xp in tests if dec(c) == xp) / len(tests)
            real = causal >= 0.8 and obs >= 0.5
            print(f"  L={L}: obs {obs:.0%}  causal {causal:.0%}{'  ← confirmed' if real else ''}")
            if real:
                Ls.append(L)
        if not Ls:
            print("  no induction confirmed on this model — measurement moot")
        print(f"  confirmed induction orders: {Ls}\n")

        # 2. per-corpus marginal coverage of induction in the n-gram ABSTAIN region (on natural in-domain text).
        def measure(label, path):
            ids = tok.encode(open(path, encoding="utf-8").read()).ids
            if len(ids) < 4 * LCTX:
                print(f"[{label}] corpus too short ({len(ids)} toks)")
                return
            cut = int(len(ids) * 0.6)                            # positional train/holdout split (disjoint regions)
            tr = [(tuple(ids[i - W:i]), ids[i], i) for i in range(W, cut)]
            tab, cites = build_tab(tr, W)
            conf = confident_rules(tab, cites, W, MINSUPP, MINDET)     # the n-gram memorization tier (corpus-built)
            hold_pos = list(range(max(cut, LCTX), len(ids)))
            if len(hold_pos) > n_hold:                            # bound oracle cost: strided sample across the region
                step = len(hold_pos) / n_hold
                hold_pos = [hold_pos[int(k * step)] for k in range(n_hold)]
            ctxs = {i: ids[i - LCTX:i] for i in hold_pos}
            fill(list(ctxs.values()))
            ng = abst = ind_fire = ind_match = 0
            for i in hold_pos:
                ctx = ctxs[i]
                ref = dec(ctx)
                if any(tuple(ctx[-k:]) in conf.get(k, {}) for k in range(1, W + 1)):
                    ng += 1                                       # n-gram tier fires (memorized)
                    continue
                abst += 1                                         # ABSTAIN region — the only place a circuit can add value
                p = induction_pred(ctx, Ls)
                if p is not None:
                    ind_fire += 1
                    ind_match += (p == ref)
            H = len(hold_pos)
            print(f"[{label:<6}] holdout n={H}  | n-gram covers {ng} ({ng/H:.0%})  abstains {abst} ({abst/H:.0%})")
            print(f"         in the abstain region: induction FIRES {ind_fire} ({ind_fire/abst if abst else 0:.0%} of abstains), "
                  f"MATCHES model {ind_match} ({ind_match/ind_fire if ind_fire else 0:.0%} of fires)")
            print(f"         ==> MARGINAL circuit coverage = {ind_match}/{H} = {ind_match/H:.1%} of all in-domain holdout positions\n")

        measure("code", code_path)
        measure("prose", prose_path)
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
