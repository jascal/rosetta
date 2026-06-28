#!/usr/bin/env python3
"""rosetta · minimize.py — build a COMPLETE certified circuits-only program for a model, then prove it in Datalog.

Fully validates a model (threx) before moving to the next. Strategy:
  1. instances  = deduped corpus decision windows (length W).
  2. ref        = the model's argmax per instance, read off whole.dl ONCE and cached (the slow oracle, paid once).
  3. composed   = the rediscovered arithmetic rule covers its instances with ONE rule (vs ~25 memorized suffixes).
  4. cover rest = minimal-suffix cover: assign each remaining instance the SHORTEST suffix under which the model is
                  deterministic over the instance set. With max length = W this covers everything by construction; the
                  quality is HOW FEW rules result (short suffixes = priors, longer = selected/structural).
  5. emit + certify = write a multi-instance circuits.dl and let dl/equiv.dl PROVE it == model (nmiss=0 ∧ nuncov=0).
The honest payoff is the breakdown: composed (1 rule) vs short-suffix priors vs longer structural/selected rules, and
the rule-count vs instance-count compression. Usage: python3 py/minimize.py [n_instances] [window]
"""
import os, sys, json
from collections import defaultdict, Counter
from oracle import decide, run_equiv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(HERE, "reference", "threx")
WHOLE = os.path.join(REF, "whole.dl")
OUT = os.path.join(REF, "circuits.dl")
CACHE = os.path.join(REF, "ref_cache.json")
LEXJ = json.load(open(os.path.join(REF, "lexicon.json")))
SYM = {i: t[0] for i, t in enumerate(LEXJ["tokens"])}
IDS = json.load(open(os.path.join(REF, "corpus.json")))["ids"]
BEARINGS, THINGS = [21, 22, 23, 24, 25], [10, 11, 12, 13, 26, 27, 28, 29, 30]


def instances(n, w):
    """deduped decision windows (length w) from the corpus — each is a context whose continuation the model decides."""
    seen, out = set(), []
    for i in range(w, len(IDS)):
        win = tuple(IDS[i - w:i])
        if win not in seen:
            seen.add(win); out.append(list(win))
            if len(out) >= n:
                break
    return out


def get_refs(insts):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    refs, miss = [], 0
    for ctx in insts:
        k = ",".join(map(str, ctx))
        if k not in cache:
            cache[k] = decide(WHOLE, ctx); miss += 1
            if miss % 10 == 0:
                json.dump(cache, open(CACHE, "w")); print(f"   …{miss} oracle calls")
        refs.append(cache[k])
    json.dump(cache, open(CACHE, "w"))
    return refs


def composed_fires(ctx):
    """python mirror of the composed rule: returns its output, or None if the frame doesn't match."""
    if len(ctx) < 6 or ctx[-1] != 7 or ctx[-6] != 20:
        return None
    bi, bj = ctx[-5], ctx[-4]
    if bi not in BEARINGS or bj not in BEARINGS or ctx[-3] != 19 or ctx[-2] != 19:
        return None
    s = (bi - 21) + (bj - 21)
    return THINGS[s] if 0 <= s < len(THINGS) else None


def minimal_suffix_cover(insts, refs, idxs, wmax):
    """assign each instance in idxs the shortest suffix that is model-deterministic over the instance set."""
    rules = {}                       # suffix(tuple) -> out
    remaining = set(idxs)
    for k in range(1, wmax + 1):
        groups = defaultdict(list)
        for i in remaining:
            groups[tuple(insts[i][-k:])].append(i)
        for suf, members in groups.items():
            outs = {refs[i] for i in members}
            if len(outs) == 1:                      # model-deterministic at length k → assign
                rules[suf] = outs.pop()
                remaining -= set(members)
        if not remaining:
            break
    return rules, remaining


def emit(rules):
    # NB: tok/ref are declared (and .input) by equiv.dl, which #includes this file — do not redeclare them here.
    L = [".decl mp(inst:number,m:number)", "mp(I,M) :- M = max P : { tok(I,P,_) }.",
         ".decl cdecide(inst:number,out:number)"]
    # composed circuit (the rediscovered arithmetic) — priority
    L += [".decl strength(b:number,s:number)", ".decl sumthing(s:number,t:number)",
          ".decl comp(inst:number,t:number)", ".decl has_comp(inst:number)"]
    L += [f"strength({b},{i})." for i, b in enumerate(BEARINGS)]
    L += [f"sumthing({s},{t})." for s, t in enumerate(THINGS)]
    L += ["comp(I,T) :- mp(I,P), tok(I,P,7), tok(I,P5,20),P5=P-5, tok(I,P4,Bi),P4=P-4, tok(I,P3,Bj),P3=P-3, "
          "strength(Bi,Si), strength(Bj,Sj), sumthing(Si+Sj,T).", "has_comp(I) :- comp(I,_)."]
    # suffix rules grouped by length; longest match wins
    bylen = defaultdict(dict)
    for suf, out in rules.items():
        bylen[len(suf)][suf] = out
    lens = sorted(bylen)
    for n in lens:
        cols = ",".join(f"c{i}:number" for i in range(n))
        L += [f".decl s{n}({cols},t:number)", f".decl s{n}p(inst:number,t:number)", f".decl any{n}(inst:number)"]
        L += [f"s{n}({','.join(map(str,suf))},{out})." for suf, out in bylen[n].items()]
        atoms = ["mp(I,P)"] + [f"tok(I,{'P' if i==n-1 else f'P{n-1-i}'},C{i})" for i in range(n)]
        atoms += [f"P{k}=P-{k}" for k in range(1, n)] + [f"s{n}({','.join(f'C{i}' for i in range(n))},T)"]
        L += [f"s{n}p(I,T) :- {', '.join(atoms)}.", f"any{n}(I) :- s{n}p(I,_)."]
    # router: composed > longest suffix
    L.append("cdecide(I,T) :- comp(I,T).")
    for n in lens:
        guard = "".join(f", !any{m}(I)" for m in lens if m > n) + ", !has_comp(I)"
        L.append(f"cdecide(I,T) :- s{n}p(I,T){guard}.")
    open(OUT, "w").write("\n".join(L) + "\n")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    insts = instances(n, w)
    print(f"=== rosetta · fully minimize+certify threx · {len(insts)} unique decision windows (W={w}) ===")
    print("1. oracle (whole.dl) — cached:")
    refs = get_refs(insts)
    comp_idx = [i for i, c in enumerate(insts) if composed_fires(c) is not None and composed_fires(c) == refs[i]]
    comp_wrong = [i for i, c in enumerate(insts) if composed_fires(c) is not None and composed_fires(c) != refs[i]]
    rest = [i for i in range(len(insts)) if i not in set(comp_idx)]
    print(f"2. composed rule covers {len(comp_idx)} instances (1 rule); composed-misfires: {len(comp_wrong)}")
    rules, remaining = minimal_suffix_cover(insts, refs, rest, w)
    print(f"3. minimal-suffix cover on the other {len(rest)}: {len(rules)} rules, {len(remaining)} unresolved")
    bylen = Counter(len(s) for s in rules)
    print(f"   rule lengths: {dict(sorted(bylen.items()))}  (len1 = marginal priors; longer = selected/structural)")
    emit(rules)
    print("4. certify the whole program vs the model, in Datalog (equiv.dl):")
    r = run_equiv(OUT, insts, refs)
    if "error" in r:
        print("   ERROR:", r["error"]); return
    print(f"   ncover={r['ncover']}  nmiss={r['nmiss']}  nuncov={r['nuncov']}")
    if r["mismatches"]:
        print("   mismatches:", [(SYM.get(a), SYM.get(b)) for _, a, b in r["mismatches"][:6]])
    ok = r["nmiss"] == 0 and r["nuncov"] == 0 and r["ncover"] == len([x for x in refs if x is not None])
    nrules = 1 + len(rules)
    print(f"\n   threx FULLY CERTIFIED (Datalog): {ok}")
    print(f"   program: 1 composed rule + {len(rules)} suffix rules = {nrules} rules for {len(insts)} decisions "
          f"({100*(1-nrules/len(insts)):.0f}% fewer rules than memorized)")


if __name__ == "__main__":
    main()
