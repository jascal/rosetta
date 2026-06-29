#!/usr/bin/env python3
"""rosetta · feat_idiom.py — phase 1.5 of the SAE/feature bridge: can a feature-keyed IDIOM (a generalizing rule) be
EXTRACTED from SAE features and clear the cover's discipline (detect + causal + certify)? Tests the kill signal before
building the fieldrun feature-export plumbing. Deps: requirements-sae.txt. Substrate: pythia-160m + EleutherAI SAEs.

FINDING (decisive, honest NEGATIVE — substrate-limited, not bridge-limited):
 1. IOI (feature-relative copy pointer): the simplest feature idiom does NOT clear the gate — best ~65% vs truth /
    ~54% vs model / 7% causal (cover needs >=80%). Phase-0 showed the features CARRY the info (63-81% patchable), but it
    is NOT cleanly extractable as a generalizing rule — the forge tax / 'doesn't factor through features', concretized.
    Also IOI is TOKEN-STRUCTURAL (answer = the non-duplicated name = a duplicate-token primitive), so it doesn't even
    NEED features.
 2. Semantic gender-binding (the genuinely feature-needing case — selection by in-context gender, both names symmetric in
    position): pythia-160m gets it 21% (< the 50% two-name chance; it mostly predicts punctuation/articles, not names).
    No capability → nothing to extract.
 => On pythia-160m there is no task that is BOTH genuinely-semantic AND within the model's ability. The bottleneck is the
    MODEL (160m, NeoX-IOI-weak, fails all semantic families per CROSS_ARCH.md), not the SAE or the bridge. The earned next
    substrate is a strong model with per-layer SAEs (Gemma Scope on Gemma-2), where semantic binding is within ability.

Usage: .venv/bin/python py/feat_idiom.py   (IOI pointer extraction; see py/sae_bridge.py for the phase-0 causal patching)
"""

import glob, os, torch, numpy as np
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
M="EleutherAI/pythia-160m"; tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
SNAP=glob.glob(f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--EleutherAI--sae-pythia-160m-32k/snapshots/*")[0]
def mksae(L):
    p=glob.glob(f"{SNAP}/layers.{L}/*.safetensors")[0]; W={}
    with safe_open(p,framework="pt") as f:
        for k in f.keys(): W[k]=f.get_tensor(k).float()
    We,be,Wd,bd=W["encoder.weight"],W["encoder.bias"],W["W_dec"],W["b_dec"]; K=32
    def enc(x):
        pre=(x-bd)@We.T+be; v,i=pre.topk(K,-1); v=v.clamp(min=0)
        a=torch.zeros_like(pre); a.scatter_(-1,i,v); return a
    return enc,Wd
pool=[" John"," Mary"," Tom"," Sara"," Paul"," Anna"," Mark"," Lucy"," David"," Emma"," Peter"," Laura"," James"," Kate"," Henry"," Alice"," Sam"," Rose"," Jack"," Lily"," Frank"," Grace"," Bob"," Jane"]
names=[n for n in pool if len(tok(n,add_special_tokens=False).input_ids)==1]
tid=lambda t: tok(t,add_special_tokens=False).input_ids[0]
rng=np.random.default_rng(0)
def ex(a,b,giver):
    s=f"When{a} and{b} went to the store,{giver} gave a drink to"; io=a if giver==b else b; return s,io
data=[]
for _ in range(120):
    a,b=rng.choice(names,2,replace=False); giver=rng.choice([a,b]); s,io=ex(a,b,giver)
    ids=tok(s,return_tensors="pt").input_ids; data.append((ids,a,b,io,giver))
L0=data[0][0].shape[1]; data=[d for d in data if d[0].shape[1]==L0]
IDX_A,IDX_B=1,3; sp=int(len(data)*0.6); train,hold=data[:sp],data[sp:]
def feats(ids,L,enc): return enc(model(ids,output_hidden_states=True).hidden_states[L+1][0])
mpred=lambda ids: tok.decode([model(ids).logits[0,-1].argmax().item()]).strip()
print(f"N={len(data)} train{len(train)}/hold{len(hold)}; model IOI vs TRUTH (hold): {sum(mpred(ids)==io.strip() for ids,a,b,io,g in hold)/len(hold):.0%}")
for L in (6,7,8,9,10):
    enc,Wd=mksae(L)
    role=torch.zeros(32768)
    for ids,a,b,io,g in train:
        F=feats(ids,L,enc); iop=IDX_A if io==a else IDX_B; spp=IDX_B if io==a else IDX_A
        role+=F[iop]-F[spp]
    role/=len(train)
    for K in (1,5,20):
        idx=role.abs().topk(K).indices; sgn=torch.sign(role[idx])
        def score(F,p): return float((F[p,idx]*sgn).sum())
        dm=dt=0
        for ids,a,b,io,g in hold:
            F=feats(ids,L,enc); pick=a if score(F,IDX_A)>=score(F,IDX_B) else b
            dm+=pick.strip()==mpred(ids); dt+=pick.strip()==io.strip()
        print(f"  layer {L} top-{K:>2} role-feats: detect-vs-model {dm/len(hold):.0%}  vs-truth {dt/len(hold):.0%}")
