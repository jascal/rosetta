#!/usr/bin/env python3
"""rosetta · succession_package.py — wire a model's SUCCESSION/ordinal circuit into an expert package and measure it on
HELD-OUT letter runs. The SECOND OOD circuit across the manifest boundary (after `induction_package.py`).

Confirms succession CAUSALLY (shift the run one letter up; the predicted successor must shift — it tracks ordinal
position, not a memorized token), emits it via `emit_expert_package(succ=…)` as a first-class `succession` manifest
rule (tier=trusted, basis=causal, routing=ood), and serves it on HELD-OUT LATE-letter runs the n-gram tier never saw —
where a memorized suffix can't generalize but the ordinal rule can. Reports coverage WITH succession vs n-gram-only.

Needs a fieldrun bundle + its tokenizer. Usage:
    .venv/bin/python py/succession_package.py [bundle-stem] [port]   # default models/llama32_1b/bundle 8197
"""
import os
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
LETTERS = [f" {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
W = 8


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else "models/llama32_1b/bundle"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8197
    if not os.path.exists(FIELDRUN):
        sys.exit(f"fieldrun not found: {FIELDRUN} (set $FIELDRUN)")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(bundle + ".tokenizer.json")

    # The SAME letter tokenizes differently by spacing ("A"=32 after a comma vs " A"=362 after a space), so `lord` must
    # enumerate every single-token VARIANT of each letter (build-time; the served rule stays tokenizer-free). `sp` is the
    # leading-space OUTPUT form the model emits; `lat` maps ordinal → that output id.
    CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sp, variants = {}, {}
    for ch in CHARS:
        for form in (f" {ch}", ch):
            ids = tok.encode(form, add_special_tokens=False).ids
            if len(ids) == 1:
                variants.setdefault(ch, set()).add(ids[0])
                if form == f" {ch}":
                    sp[ch] = ids[0]
    letters = [ch for ch in CHARS if ch in sp]                  # need the leading-space output form
    if len(letters) < 8:
        sys.exit(f"only {len(letters)} single-token letters — need >=8")
    pos = {ch: i for i, ch in enumerate(letters)}
    lord = {tid: pos[ch] for ch in letters for tid in variants[ch]}   # every spacing variant → ordinal
    lat = {pos[ch]: sp[ch] for ch in letters}                        # ordinal → leading-space output token
    runs = [(letters[i], letters[i + 1], letters[i + 2], letters[i + 3]) for i in range(len(letters) - 3)]

    proc = subprocess.Popen([FIELDRUN, "--bundle", bundle, "--serve", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        up = False
        for _ in range(90):
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
            with ThreadPoolExecutor(max_workers=2) as ex:
                for c, o in zip(miss, ex.map(lambda t: serve_decide(port, list(t)), miss)):
                    cache[c] = o

        def dec(ctx):
            k = tuple(ctx)
            if k not in cache:
                cache[k] = serve_decide(port, ctx)
            return cache[k]

        # FORMAT-ROBUST (CROSS_ARCH): a code model reads ' A B C' as a token list but 'A, B, C,' as a sequence — keep
        # whichever join the model actually continues (highest detect).
        fmts = [lambda a, b, c: f" {a} {b} {c}", lambda a, b, c: f"{a}, {b}, {c},"]
        best = None
        for fmt in fmts:
            ins = [tok.encode(fmt(a, b, c)).ids for a, b, c, _ in runs]
            fill(ins)
            rf = [dec(x) for x in ins]
            tr = [sp[d] for *_, d in runs]
            det = sum(rf[i] == tr[i] for i in range(len(runs))) / len(runs)
            if best is None or det > best[0]:
                best = (det, fmt, ins, rf, tr)
        detect, fmt, insts, refs, truth = best
        n = len(insts)
        print(f"=== succession_package · {os.path.basename(bundle)} · {n} letter runs · detect {detect:.0%} ===")

        # CAUSAL — shift the whole run one letter up; the predicted successor must shift with it.
        shifted, sh_truth, base = [], [], []
        for i in range(len(letters) - 4):
            shifted.append(tok.encode(fmt(letters[i + 1], letters[i + 2], letters[i + 3])).ids)
            sh_truth.append(sp[letters[i + 4]])
            base.append(i)
        fill(shifted)
        did = [i for i in base if refs[i] == truth[i]]           # runs the model actually continues
        causal = (sum(1 for k, i in enumerate(base) if i in did and dec(shifted[k]) == sh_truth[k]) / len(did)) if did else 0.0
        print(f"  causal {causal:.0%} over {len(did)} continued runs" + ("  ← REAL" if causal >= 0.8 else ""))

        # TRAIN = early-letter runs, HELD-OUT = late-letter runs (the n-gram tier never saw the late transitions).
        cut = int(n * 0.55)
        tr_idx, ho_idx = list(range(cut)), list(range(cut, n))
        md = os.path.join(HERE, "..", "models", os.path.basename(os.path.dirname(bundle)))
        succ = {"lord": lord, "lat": lat, "causal": causal}
        _, man = emit_expert_package(md, insts, refs, tr_idx, [], [], [], W, os.path.basename(bundle), 3, 1.0, succ=succ)
        idioms, ngrams, m = load_package(man)
        print(f"[manifest] rules={len(m['rules'])} succession_ood={m.get('succession_ood')} "
              f"succession_cover(train)={m.get('succession_cover')}  gated_ngrams={m['gated_ngrams']}")

        print("\nHELD-OUT (late-letter runs — the n-gram tier has no support here):")
        for label, idi in [("WITH succession", idioms), ("n-gram only", [r for r in idioms if r['kind'] != 'succession'])]:
            ans = cor = ab = sc = 0
            for i in ho_idx:
                r = serve(insts[i], idi, ngrams, W)
                if r is None:
                    ab += 1
                else:
                    ans += 1
                    cor += (r["answer"] == refs[i])
                    sc += (r.get("circuit") == "succession")
            H = len(ho_idx)
            extra = f"  · succession fired {sc}" if "WITH" in label else ""
            print(f"  [{label:<15}] coverage {ans}/{H}={ans / H:.0%}  precision {cor}/{ans}="
                  f"{cor / ans if ans else 0:.0%}  abstain {ab}/{H}={ab / H:.0%}{extra}")
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
