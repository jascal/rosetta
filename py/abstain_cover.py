#!/usr/bin/env python3
"""rosetta · abstain_cover.py — CONFIDENCE-GATED / ABSTAINING rules for an OOD-exact cover (measure-before-build).

The complete_cover nuance: the semiring backstop catches ABSTENTIONS (no rule fires) exactly, but NOT MISPREDICTIONS (a
rule fired but is wrong on an unseen context) — n-grams can't self-limit, they fire on any matching suffix. So a naive
cover mispredicts on holdout and the backstop can't save it. Fix: rules ABSTAIN where unreliable and defer to the backstop
→ the complete artifact's loss → the residual confident-but-wrong rate (OOD-near-exact).

Measures abstention policies on a real diverse cover (stories110M, corpus next-token = real suffix ambiguity), train/val/
holdout split, longest-matching backoff n-gram, complete-loss = rule-mispredictions (backstop fixes abstentions exactly).
RESULT: naive (always fire) = 50% holdout mispredict; VAL-CALIBRATED (fire only suffixes a val split confirms) = ~3% loss
at ~75% abstain; support>=3 AND train-determinism==1.0 = 1.1% at 85% abstain; oracle = 0% at 54%. => abstention turns the
naive 50%-loss cover into an OOD-near-exact complete artifact (~3%), by deferring the uncertain majority to the exact
backstop. On diverse text only ~25% of contexts have a reliably-generalizing short rule; the rest genuinely need the model
(made explicit + exact, not guessed wrong). Support alone is weak; val-calibration is the practical signal.

NEXT (build): wire val-calibration into the cover emit (temperature.dist_cover / idiom_learn) — split build/val, emit only
calibrated rules, uncalibrated contexts ABSTAIN → backstop. Trades in-domain full-coverage for OOD reliability.
Usage: .venv/bin/python py/abstain_cover.py
"""

import json, os, random
from collections import Counter, defaultdict
md="models/stories110M"
ids=json.load(open(f"{md}/corpus.json"))["ids"]
W=8; N=min(len(ids)-1, 40000)
# windows over the corpus: ctx = ids[i-W:i], ref = ids[i] (next-token; corpus is the model's own generation)
wins=[(tuple(ids[i-W:i]), ids[i]) for i in range(W, N)]
rng=random.Random(0); rng.shuffle(wins)
a=int(len(wins)*.5); b=int(len(wins)*.65)
train,val,hold=wins[:a],wins[a:b],wins[b:]
print(f"corpus windows: train {len(train)} / val {len(val)} / hold {len(hold)} (W={W})")
# backoff n-gram cover from train: per suffix length k, suffix -> Counter(outputs)
tab=[defaultdict(Counter) for _ in range(W+1)]
for ctx,o in train:
    for k in range(1,W+1): tab[k][ctx[-k:]][o]+=1
rule=[{s:c.most_common(1)[0][0] for s,c in tab[k].items()} for k in range(W+1)]
supp=[{s:sum(c.values()) for s,c in tab[k].items()} for k in range(W+1)]
det =[{s:c.most_common(1)[0][1]/sum(c.values()) for s,c in tab[k].items()} for k in range(W+1)]
# val accuracy per (k,suffix): does the train rule match val occurrences?
vacc=[defaultdict(lambda:[0,0]) for _ in range(W+1)]
for ctx,o in val:
    for k in range(1,W+1):
        s=ctx[-k:]
        if s in rule[k]: vacc[k][s][1]+=1; vacc[k][s][0]+= (rule[k][s]==o)
def longest(ctx, fire):
    for k in range(W,0,-1):
        s=ctx[-k:]
        if s in rule[k] and fire(k,s): return rule[k][s]
    return None
def evaluate(name, fire):
    cov=cor=0
    for ctx,o in hold:
        p=longest(ctx,fire)
        if p is not None: cov+=1; cor+= (p==o)
    H=len(hold); mis=cov-cor; ab=H-cov
    # complete artifact: rules where fired (cor right, mis wrong) + backstop EXACT on abstentions
    print(f"  {name:34} fire {cov/H:.0%}  abstain→backstop {ab/H:.0%}  | rule-mispred(=complete loss) {mis/H:.1%}")
print("\npolicy (longest matching rule, gated):")
evaluate("none (always fire) — baseline", lambda k,s: True)
for kmin in (2,3,5,10):
    evaluate(f"support >= {kmin}", lambda k,s,kmin=kmin: supp[k][s]>=kmin)
evaluate("train-determinism == 1.0", lambda k,s: det[k][s]>=0.999)
evaluate("val-calibrated (>=1 val, all correct)", lambda k,s: vacc[k][s][1]>0 and vacc[k][s][0]==vacc[k][s][1])
evaluate("val-calibrated (acc>=0.8, >=2 val)", lambda k,s: vacc[k][s][1]>=2 and vacc[k][s][0]/vacc[k][s][1]>=0.8)
evaluate("support>=3 AND det==1.0", lambda k,s: supp[k][s]>=3 and det[k][s]>=0.999)
# ORACLE: abstain exactly where the longest-always rule is wrong (upper bound)
covB=corB=0
for ctx,o in hold:
    p=longest(ctx,lambda k,s:True)
    if p is not None: covB+=1; corB+=(p==o)
print(f"\n  ORACLE (abstain iff would-mispredict): complete loss 0% at abstain {(len(hold)-corB)/len(hold):.0%}  (baseline mispred was {(covB-corB)/len(hold):.1%})")
