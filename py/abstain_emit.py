#!/usr/bin/env python3
"""rosetta · abstain_emit.py — emit an ABSTAINING (bounded-expert) circuits.dl, the abstaining-rules build informed by
sgiandubh (the partial-backstop runtime). Each rule CARRIES its confidence (support + determinism, computed at build);
the cover FIRES only confident rules and ABSTAINS otherwise (no cdecide → defer to the backstop, or refuse if there is
none — the bounded expert). This is the souffle artifact behind py/abstain_cover.py's measurement.

Design (from the sgiandubh look, see SAE_BRIDGE.md / memory):
 - carry per-rule confidence set at BUILD, gate at RUNTIME (sgiandubh's hit->margin) — here: support + determinism;
 - the reliable unit is the confident rule (≈ sgiandubh's curated item / the causal idiom); the rest abstains;
 - bounded-expert scorecard = coverage (answer rate) / precision (correct when answered) / abstain — a reject-option
   classifier, not 'loss'. With an exact backstop (whole.dl) abstain→exact; with a partial one (sgiandubh) abstain=refuse.

Usage: python3 py/abstain_emit.py [model_dir] [W] [minsupp] [mindet]
"""
import json, os, sys, random, subprocess, tempfile
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_tab(train, W):
    tab = [defaultdict(Counter) for _ in range(W + 1)]
    for ctx, o in train:
        for k in range(1, W + 1):
            tab[k][ctx[-k:]][o] += 1
    return tab


def confident_rules(tab, W, minsupp, mindet):
    """rules whose suffix is reliable on train: seen >= minsupp times AND determinism (majority fraction) >= mindet.
    Returns {k: {suffix: (output, support, det)}}. The carried confidence is (support, det) — sgiandubh's margin analogue."""
    out = defaultdict(dict)
    for k in range(1, W + 1):
        for s, c in tab[k].items():
            tot = sum(c.values()); top, cnt = c.most_common(1)[0]
            if tot >= minsupp and cnt / tot >= mindet:
                out[k][s] = (top, tot, cnt / tot)
    return out


def emit(path, rules, W, minsupp, mindet):
    L = [f"// rosetta · circuits.dl — ABSTAINING bounded-expert cover (fire only confident rules: support>={minsupp},",
         f"// determinism>={mindet}); ABSTAIN otherwise (no cdecide → defer to backstop / refuse). Longest confident suffix wins.",
         "// Each gramN fact is a rule whose suffix met the confidence bar at build time (carried as the fact's existence).",
         ".decl tok(inst:number,pos:number,id:number)", ".input tok",
         ".decl mp(inst:number,m:number)", "mp(I,M) :- M = max P : { tok(I,P,_) }.",
         ".decl cdecide(inst:number,out:number)", ".output cdecide"]
    lens = sorted(rules)
    for k in lens:
        N = k + 1
        cols = ",".join(f"c{i}:number" for i in range(k))
        L += [f".decl gram{N}({cols},t:number)", f".decl gram{N}_hit(inst:number,t:number)", f".decl gram{N}_any(inst:number)"]
        L += [f"gram{N}({','.join(map(str, s))},{o})." for s, (o, sup, d) in rules[k].items()]
        cvars = [f"C{i}" for i in range(k)]
        body = ["mp(I,P)"] + [f"tok(I,{'P' if i == k - 1 else f'Pm{k-1-i}'},{cvars[i]})" for i in range(k)]
        body += [f"Pm{j}=P-{j}" for j in range(1, k)] + [f"gram{N}({','.join(cvars)},T)"]
        L += [f"gram{N}_hit(I,T) :- {', '.join(body)}.", f"gram{N}_any(I) :- gram{N}_hit(I,_)."]
    for k in lens:
        guard = "".join(f", !gram{m+1}_any(I)" for m in lens if m > k)
        L.append(f"cdecide(I,T) :- gram{k+1}_hit(I,T){guard}.")
    open(path, "w").write("\n".join(L) + "\n")
    return sum(len(rules[k]) for k in lens)


def predict(ctx, rules, W):
    for k in range(min(len(ctx), W), 0, -1):
        s = ctx[-k:]
        if s in rules[k]:
            return rules[k][s][0]
    return None  # ABSTAIN


def scorecard(name, rules, hold, W):
    ans = cor = 0
    for ctx, o in hold:
        p = predict(ctx, rules, W)
        if p is not None:
            ans += 1; cor += (p == o)
    H = len(hold)
    prec = cor / ans if ans else 0.0
    print(f"  {name:30} coverage {ans/H:5.0%}  precision {prec:5.0%}  abstain {1-ans/H:5.0%}  "
          f"| complete-loss(if exact backstop) {(ans-cor)/H:.1%}")
    return ans, cor


def souffle_check(dl_path, hold_sample, rules, W):
    """run the emitted circuits.dl in souffle on a sample and confirm cdecide matches the Python cover (and abstains
    exactly where Python abstains)."""
    with tempfile.TemporaryDirectory() as d:
        ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
        os.makedirs(ind); os.makedirs(outd)
        with open(os.path.join(ind, "tok.facts"), "w") as f:
            for i, (ctx, o) in enumerate(hold_sample):
                for p, t in enumerate(ctx):
                    f.write(f"{i}\t{p}\t{t}\n")
        r = subprocess.run(["souffle", dl_path, "-F", ind, "-D", outd], capture_output=True, text=True)
        if r.returncode != 0:
            return f"souffle error: {(r.stderr.strip().splitlines() or [''])[-1]}"
        got = {}
        cp = os.path.join(outd, "cdecide.csv")
        if os.path.exists(cp):
            for ln in open(cp).read().splitlines():
                i, t = ln.split("\t"); got[int(i)] = int(t)
    mism = 0
    for i, (ctx, o) in enumerate(hold_sample):
        py = predict(ctx, rules, W)
        sf = got.get(i)  # None = souffle abstained (no cdecide row)
        if py != sf:
            mism += 1
    ab = sum(1 for i in range(len(hold_sample)) if i not in got)
    return f"souffle vs python: {mism} mismatch / {len(hold_sample)} (abstained {ab}) — {'EXACT' if mism == 0 else 'MISMATCH'}"


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "models", "stories110M")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    minsupp = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mindet = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    N = min(len(ids) - 1, 40000)
    wins = [(tuple(ids[i - W:i]), ids[i]) for i in range(W, N)]
    random.Random(0).shuffle(wins)
    cut = int(len(wins) * 0.7); train, hold = wins[:cut], wins[cut:]
    print(f"=== abstain_emit · {os.path.basename(md)} · {len(train)} train / {len(hold)} hold (W={W}) ===")
    tab = build_tab(train, W)
    naive = confident_rules(tab, W, 1, 0.0)         # all rules (baseline)
    conf = confident_rules(tab, W, minsupp, mindet)  # confident rules only (abstaining)
    print("bounded-expert scorecard (reject-option classifier):")
    scorecard("naive (all rules fire)", naive, hold, W)
    nc = scorecard(f"abstaining (supp>={minsupp}, det>={mindet})", conf, hold, W)
    out = os.path.join(md, "circuits.abstain.dl")
    nr = emit(out, conf, W, minsupp, mindet)
    print(f"emitted {nr} confident rules → {out}")
    print(" ", souffle_check(out, hold[:400], conf, W))
    print("note: the IDIOM tier (causal, generalizes) would be the TRUSTED ungated unit; this demo model is n-gram only.")


if __name__ == "__main__":
    main()
