#!/usr/bin/env python3
"""rosetta · abstain_emit.py — emit the sgiandubh-consumable EXPERT PACKAGE: an ABSTAINING (bounded-expert) circuits.dl
+ a manifest.json with per-rule CONFIDENCE and CITATION (provenance). The abstaining-rules build informed by the
sgiandubh look (the partial-backstop runtime); Stage 1 of the rosetta→sgiandubh convergence (rosetta = builder/emitter,
sgiandubh = thin runtime that loads this package).

Each rule CARRIES its confidence (support + determinism, computed at build) and FIRES only where confident; otherwise it
ABSTAINS (no cdecide → defer to backstop / refuse). It also CARRIES provenance: every rule has an id, the cover emits
`cprov(inst, ruleid)` at runtime, and manifest.json maps ruleid → {ctx, out, support, determinism, cite} so the runtime
can cite the answer (answer → rule → corpus positions → passage, via an optional corpus_meta.json offset→citation map).

Package = circuits.abstain.dl (decode + rules + cprov) + manifest.json. Bounded-expert scorecard = coverage/precision/abstain.
Usage: python3 py/abstain_emit.py [model_dir] [W] [minsupp] [mindet]
"""
import json, os, sys, random, subprocess, tempfile
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_tab(train, W):
    """train = [(ctx, out, pos)]. Returns tab[k][suffix]=Counter(out) and cites[k][suffix][out]=[corpus positions]."""
    tab = [defaultdict(Counter) for _ in range(W + 1)]
    cites = [defaultdict(lambda: defaultdict(list)) for _ in range(W + 1)]
    for ctx, o, pos in train:
        for k in range(1, W + 1):
            s = ctx[-k:]
            tab[k][s][o] += 1
            cites[k][s][o].append(pos)
    return tab, cites


def confident_rules(tab, cites, W, minsupp, mindet, ncite=5):
    """rules whose suffix is reliable on train (support>=minsupp AND determinism>=mindet). Returns
    {k: {suffix: (output, support, det, cite_positions)}} — cite = sample positions supporting (suffix→output)."""
    out = defaultdict(dict)
    for k in range(1, W + 1):
        for s, c in tab[k].items():
            tot = sum(c.values()); top, cnt = c.most_common(1)[0]
            if tot >= minsupp and cnt / tot >= mindet:
                out[k][s] = (top, tot, cnt / tot, cites[k][s][top][:ncite])
    return out


def emit(path, rules, W, minsupp, mindet):
    """Emit circuits.abstain.dl. Each gramN fact carries a ruleid; cdecide = longest confident match; cprov = its ruleid
    (so the runtime can cite). No confident rule → no cdecide/cprov = ABSTAIN. Returns {ruleid: (k, suffix, out, sup, det, cite)}."""
    rid = 0
    ridmap = {}
    L = [f"// rosetta · circuits.abstain.dl — ABSTAINING bounded-expert cover (fire only confident rules: support>={minsupp},",
         f"// determinism>={mindet}); ABSTAIN otherwise (no cdecide → defer to backstop / refuse). Longest confident suffix wins.",
         "// gramN(ctx…, token, ruleid): a confident rule (its existence = it passed the bar); ruleid → manifest.json (cite+confidence).",
         ".decl tok(inst:number,pos:number,id:number)", ".input tok",
         ".decl mp(inst:number,m:number)", "mp(I,M) :- M = max P : { tok(I,P,_) }.",
         ".decl cdecide(inst:number,out:number)", ".output cdecide",
         ".decl cprov(inst:number,ruleid:number)", ".output cprov"]
    lens = sorted(rules)
    for k in lens:
        N = k + 1
        cols = ",".join(f"c{i}:number" for i in range(k))
        L += [f".decl gram{N}({cols},t:number,rid:number)", f".decl gram{N}_hit(inst:number,t:number,rid:number)",
              f".decl gram{N}_any(inst:number)"]
        for s, (o, sup, d, cite) in rules[k].items():
            ridmap[rid] = (k, list(s), o, sup, d, cite)
            L.append(f"gram{N}({','.join(map(str, s))},{o},{rid}).")
            rid += 1
        cvars = [f"C{i}" for i in range(k)]
        body = ["mp(I,P)"] + [f"tok(I,{'P' if i == k - 1 else f'Pm{k-1-i}'},{cvars[i]})" for i in range(k)]
        body += [f"Pm{j}=P-{j}" for j in range(1, k)] + [f"gram{N}({','.join(cvars)},T,RID)"]
        L += [f"gram{N}_hit(I,T,RID) :- {', '.join(body)}.", f"gram{N}_any(I) :- gram{N}_hit(I,_,_)."]
    for k in lens:
        guard = "".join(f", !gram{m+1}_any(I)" for m in lens if m > k)
        L.append(f"cdecide(I,T) :- gram{k+1}_hit(I,T,_){guard}.")
        L.append(f"cprov(I,RID) :- gram{k+1}_hit(I,_,RID){guard}.")
    open(path, "w").write("\n".join(L) + "\n")
    return ridmap


def emit_manifest(path, ridmap, model, W, minsupp, mindet, meta=None):
    """manifest.json: ruleid → ctx/out/confidence/cite. If meta (corpus_meta.json: [{start,end,citation}]) is present,
    resolve cite positions → citation strings (the human passage); else cite stays raw corpus offsets (provenance)."""
    def resolve(pos):
        if not meta:
            return None
        for m in meta:
            if m["start"] <= pos < m["end"]:
                return m["citation"]
        return None
    rules = []
    for rid, (k, ctx, o, sup, d, cite) in sorted(ridmap.items()):
        e = {"id": rid, "ctx": ctx, "out": o, "support": sup, "determinism": round(d, 3), "cite": cite}
        if meta:
            e["citation"] = sorted({c for c in (resolve(p) for p in cite) if c})
        rules.append(e)
    json.dump({"model": model, "params": {"W": W, "minsupp": minsupp, "mindet": mindet},
               "decode": "circuits.abstain.dl", "n_rules": len(rules), "rules": rules},
              open(path, "w"))
    return len(rules)


def predict(ctx, rules, W):
    for k in range(min(len(ctx), W), 0, -1):
        s = ctx[-k:]
        if s in rules[k]:
            return rules[k][s][0]
    return None


def scorecard(name, rules, hold, W):
    ans = cor = 0
    for ctx, o, _ in hold:
        p = predict(ctx, rules, W)
        if p is not None:
            ans += 1; cor += (p == o)
    H = len(hold); prec = cor / ans if ans else 0.0
    print(f"  {name:30} coverage {ans/H:5.0%}  precision {prec:5.0%}  abstain {1-ans/H:5.0%}  "
          f"| complete-loss(if exact backstop) {(ans-cor)/H:.1%}")


def souffle_check(dl_path, hold_sample, rules, ridmap, W):
    """run circuits.abstain.dl in souffle; confirm cdecide matches the Python cover AND cprov names the right rule (cite link)."""
    rev = {}  # (k, suffix) -> ruleid, to check cprov
    for rid, (k, ctx, o, *_ ) in ridmap.items():
        rev[(k, tuple(ctx))] = rid
    with tempfile.TemporaryDirectory() as d:
        ind, outd = os.path.join(d, "in"), os.path.join(d, "out")
        os.makedirs(ind); os.makedirs(outd)
        with open(os.path.join(ind, "tok.facts"), "w") as f:
            for i, (ctx, o, _) in enumerate(hold_sample):
                for p, t in enumerate(ctx):
                    f.write(f"{i}\t{p}\t{t}\n")
        r = subprocess.run(["souffle", dl_path, "-F", ind, "-D", outd], capture_output=True, text=True)
        if r.returncode != 0:
            return f"souffle error: {(r.stderr.strip().splitlines() or [''])[-1]}"
        dec, prov = {}, {}
        if os.path.exists(os.path.join(outd, "cdecide.csv")):
            for ln in open(os.path.join(outd, "cdecide.csv")).read().splitlines():
                i, t = ln.split("\t"); dec[int(i)] = int(t)
        if os.path.exists(os.path.join(outd, "cprov.csv")):
            for ln in open(os.path.join(outd, "cprov.csv")).read().splitlines():
                i, rid = ln.split("\t"); prov[int(i)] = int(rid)
    mism = provbad = 0
    for i, (ctx, o, _) in enumerate(hold_sample):
        py = predict(ctx, rules, W)
        if py != dec.get(i):
            mism += 1
        if py is not None:  # the fired rule = longest matching suffix; check cprov names it
            k = next(kk for kk in range(min(len(ctx), W), 0, -1) if ctx[-kk:] in rules[kk])
            if prov.get(i) != rev.get((k, ctx[-k:])):
                provbad += 1
    ab = sum(1 for i in range(len(hold_sample)) if i not in dec)
    return (f"souffle: {mism} decide-mismatch, {provbad} cite-mismatch / {len(hold_sample)} (abstained {ab}) — "
            f"{'EXACT (decode + provenance)' if mism == 0 and provbad == 0 else 'MISMATCH'}")


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "models", "stories110M")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    minsupp = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mindet = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    meta_p = os.path.join(md, "corpus_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else None
    N = min(len(ids) - 1, 40000)
    wins = [(tuple(ids[i - W:i]), ids[i], i) for i in range(W, N)]  # carry the corpus position i (provenance/cite)
    random.Random(0).shuffle(wins)
    cut = int(len(wins) * 0.7); train, hold = wins[:cut], wins[cut:]
    print(f"=== abstain_emit · {os.path.basename(md)} · {len(train)} train / {len(hold)} hold (W={W}) "
          f"{'· corpus_meta ✓' if meta else '· (no corpus_meta → raw offsets)'} ===")
    tab, cites = build_tab(train, W)
    naive = confident_rules(tab, cites, W, 1, 0.0)
    conf = confident_rules(tab, cites, W, minsupp, mindet)
    print("bounded-expert scorecard:")
    scorecard("naive (all rules)", naive, hold, W)
    scorecard(f"abstaining (supp>={minsupp}, det>={mindet})", conf, hold, W)
    out = os.path.join(md, "circuits.abstain.dl")
    ridmap = emit(out, conf, W, minsupp, mindet)
    man = os.path.join(md, "manifest.json")
    nr = emit_manifest(man, ridmap, os.path.basename(md), W, minsupp, mindet, meta)
    print(f"emitted package: {out} ({nr} rules) + {man} (per-rule confidence + cite)")
    print(" ", souffle_check(out, hold[:400], conf, ridmap, W))
    ex = json.load(open(man))["rules"][0]
    print(f"  manifest sample rule: id={ex['id']} ctx={ex['ctx']}→out={ex['out']} support={ex['support']} "
          f"det={ex['determinism']} cite={ex['cite']}" + (f" citation={ex.get('citation')}" if meta else ""))


if __name__ == "__main__":
    main()
