#!/usr/bin/env python3
"""rosetta · minimize.py — build a COMPLETE certified circuits-only program for a model, then prove it in Datalog.

Model-general. Strategy:
  1. instances  = deduped corpus decision windows (length W).
  2. ref        = the model's argmax per instance, read off whole.dl ONCE and cached (slow oracle, paid once; compiled).
  3. plugin     = optional hand/auto-discovered certified circuit (threx ships the composed arithmetic rule). Off by
                  default — a real LM has no such gift, and that's the point of testing breadth.
  4. cover rest = minimal-suffix cover: assign each remaining instance the SHORTEST suffix under which the model is
                  deterministic over the instance set. With max length = W this covers everything by construction; the
                  quality is HOW FEW rules result (short = priors, longer = structural/idiomatic, full-W = memorized).
  5. emit + certify = write a multi-instance circuits.dl and let dl/equiv.dl PROVE it == model (nmiss=0 ∧ nuncov=0).
Usage: python3 py/minimize.py [n_instances] [window] [model_dir]   (model_dir default: reference/threx)
"""
import os, sys, json
from collections import defaultdict, Counter
from oracle import decide, run_equiv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEARINGS, THINGS = [21, 22, 23, 24, 25], [10, 11, 12, 13, 26, 27, 28, 29, 30]  # threx composed plugin


def instances(ids, n, w):
    seen, out = set(), []
    for i in range(w, len(ids)):
        win = tuple(ids[i - w:i])
        if win not in seen:
            seen.add(win); out.append(list(win))
            if len(out) >= n:
                break
    return out


def get_refs(whole, insts, cache_path):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    refs, miss = [], 0
    for ctx in insts:
        k = ",".join(map(str, ctx))
        if k not in cache:
            cache[k] = decide(whole, ctx); miss += 1
            if miss % 25 == 0:
                json.dump(cache, open(cache_path, "w")); print(f"   …{miss} oracle calls")
        refs.append(cache[k])
    json.dump(cache, open(cache_path, "w"))
    return refs


def composed_fires(ctx):
    """threx-only plugin: the composed arithmetic rule's output, or None if the frame doesn't match."""
    if len(ctx) < 6 or ctx[-1] != 7 or ctx[-6] != 20:
        return None
    bi, bj = ctx[-5], ctx[-4]
    if bi not in BEARINGS or bj not in BEARINGS or ctx[-3] != 19 or ctx[-2] != 19:
        return None
    s = (bi - 21) + (bj - 21)
    return THINGS[s] if 0 <= s < len(THINGS) else None


def minimal_suffix_cover(insts, refs, idxs, wmax):
    rules, remaining = {}, set(idxs)
    for k in range(1, wmax + 1):
        groups = defaultdict(list)
        for i in remaining:
            groups[tuple(insts[i][-k:])].append(i)
        for suf, members in groups.items():
            outs = {refs[i] for i in members}
            if len(outs) == 1:
                rules[suf] = outs.pop()
                remaining -= set(members)
        if not remaining:
            break
    return rules, remaining


def emit(out_path, rules, composed):
    L = [".decl mp(inst:number,m:number)", "mp(I,M) :- M = max P : { tok(I,P,_) }.",
         ".decl cdecide(inst:number,out:number)"]
    if composed:
        L += [".decl strength(b:number,s:number)", ".decl sumthing(s:number,t:number)",
              ".decl comp(inst:number,t:number)", ".decl has_comp(inst:number)"]
        L += [f"strength({b},{i})." for i, b in enumerate(BEARINGS)]
        L += [f"sumthing({s},{t})." for s, t in enumerate(THINGS)]
        L += ["comp(I,T) :- mp(I,P), tok(I,P,7), tok(I,P5,20),P5=P-5, tok(I,P4,Bi),P4=P-4, tok(I,P3,Bj),P3=P-3, "
              "strength(Bi,Si), strength(Bj,Sj), sumthing(Si+Sj,T).", "has_comp(I) :- comp(I,_)."]
    bylen = defaultdict(dict)
    for suf, o in rules.items():
        bylen[len(suf)][suf] = o
    lens = sorted(bylen)
    for n in lens:
        cols = ",".join(f"c{i}:number" for i in range(n))
        L += [f".decl s{n}({cols},t:number)", f".decl s{n}p(inst:number,t:number)", f".decl any{n}(inst:number)"]
        L += [f"s{n}({','.join(map(str,suf))},{o})." for suf, o in bylen[n].items()]
        atoms = ["mp(I,P)"] + [f"tok(I,{'P' if i==n-1 else f'P{n-1-i}'},C{i})" for i in range(n)]
        atoms += [f"P{k}=P-{k}" for k in range(1, n)] + [f"s{n}({','.join(f'C{i}' for i in range(n))},T)"]
        L += [f"s{n}p(I,T) :- {', '.join(atoms)}.", f"any{n}(I) :- s{n}p(I,_)."]
    if composed:
        L.append("cdecide(I,T) :- comp(I,T).")
    for n in lens:
        guard = "".join(f", !any{m}(I)" for m in lens if m > n) + (", !has_comp(I)" if composed else "")
        L.append(f"cdecide(I,T) :- s{n}p(I,T){guard}.")
    open(out_path, "w").write("\n".join(L) + "\n")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    md = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "reference", "threx")
    md = md if os.path.isabs(md) else os.path.join(HERE, md)
    name = os.path.basename(md.rstrip("/"))
    whole = os.path.join(md, "whole.dl")
    out = os.path.join(md, "circuits.dl")
    ids = json.load(open(os.path.join(md, "corpus.json")))["ids"]
    sym = {i: t[0] for i, t in enumerate(json.load(open(os.path.join(md, "lexicon.json")))["tokens"])}
    use_composed = name == "threx"

    insts = instances(ids, n, w)
    print(f"=== rosetta · minimize+certify {name} · {len(insts)} unique decision windows (W={w}) ===")
    print("1. oracle (whole.dl, compiled) — cached:")
    refs = get_refs(whole, insts, os.path.join(md, "ref_cache.json"))
    valid = [i for i in range(len(insts)) if refs[i] is not None]

    comp_idx = []
    if use_composed:
        comp_idx = [i for i in valid if composed_fires(insts[i]) is not None and composed_fires(insts[i]) == refs[i]]
        print(f"2. composed plugin covers {len(comp_idx)} instances (1 rule)")
    rest = [i for i in valid if i not in set(comp_idx)]
    rules, remaining = minimal_suffix_cover(insts, refs, rest, w)
    bylen = Counter(len(s) for s in rules)
    print(f"3. minimal-suffix cover on {len(rest)}: {len(rules)} rules, {len(remaining)} unresolved")
    print(f"   rule lengths {dict(sorted(bylen.items()))}  (len1=priors · mid=structural/idiom · len{w}=memorized)")
    emit(out, rules, use_composed)
    print("4. certify whole program vs the model, in Datalog (equiv.dl):")
    r = run_equiv(out, [insts[i] for i in valid], [refs[i] for i in valid])
    if "error" in r:
        print("   ERROR:", r["error"]); return
    print(f"   ncover={r['ncover']}  nmiss={r['nmiss']}  nuncov={r['nuncov']}")
    if r["mismatches"]:
        print("   mismatches:", [(sym.get(a), sym.get(b)) for _, a, b in r["mismatches"][:6]])
    nrules = (1 if use_composed else 0) + len(rules)
    ok = r["nmiss"] == 0 and r["nuncov"] == 0 and r["ncover"] == len(valid)
    print(f"\n   {name} FULLY CERTIFIED (Datalog): {ok}")
    longest = max(bylen) if bylen else 0
    memo = bylen.get(w, 0)
    print(f"   program: {nrules} rules for {len(valid)} decisions "
          f"({100*(1-nrules/max(1,len(valid))):.0f}% fewer than memorizing); {memo} are full-W (memorized tail)")


if __name__ == "__main__":
    main()
