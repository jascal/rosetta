#!/usr/bin/env python3
"""rosetta · gemma_feat.py — SAE/feature bridge on a CAPABLE substrate (Gemma Scope), the fair test of phase 1.5's kill
signal. Deps: requirements-sae.txt. Substrate: gemma-2-2b + google/gemma-scope-2b-pt-res (JumpReLU SAEs, width_16k,
per layer; SAE layer_L ≈ hidden_states[L+1], recon cos ~0.94).

FINDING (the pivot was right about the substrate, but the simple idiom still fails):
 - gemma-2-2b DOES semantic gender-binding (90% — picks the in-context male name, overriding name-gender priors), vs
   pythia-160m's 21%. So the substrate is capable.
 - BUT the simple feature-relative-pointer idiom (copy the name a role feature marks) gets ~65% vs truth / 60% vs model
   on gemma — same ~65% as pythia. It does NOT clear the cover's detect>=80% gate even on a capable model that demonstrably
   does the task. => not a substrate excuse: entity selection (the distributed name-mover; phase-0 showed the signal is at
   END, not the name slot) does NOT factor into a sparse generalizing rule, though features CARRY it (phase-0 patching
   63-81%). The forge tax / "composition doesn't factor through features", now on the cover-extraction task itself.

Implication: the feature bridge's value as a CERTIFIED, GENERALIZING COVER idiom (simple pointer/gate shape) is a negative.
Features remain valuable as the NON-DESTRUCTIVE LABELING layer (phase 1, lossless) and the patching-level analysis; a
certified generalizing feature COVER would need the richer feature->feature circuit form (Phase 4, a large build, uncertain).
Usage: .venv/bin/python py/gemma_feat.py
"""

import glob, os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
M="google/gemma-2-2b"; tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16).eval()
def load_sae(L):
    p=glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res/snapshots/*/layer_{L}/width_16k/*/params.npz"))[0]
    z=np.load(p); return {k:torch.tensor(z[k],dtype=torch.float32) for k in z.files}
def enc(x,S): pre=x.float()@S["W_enc"]+S["b_enc"]; return torch.where(pre>S["threshold"],pre,torch.zeros_like(pre))
def dec(a,S): return a@S["W_dec"]+S["b_dec"]
def hs(ids,L): return model(ids,output_hidden_states=True).hidden_states[L][0].float()
# 1. reconstruction sweep (layer_21 SAE → which hidden_states index)
S21=load_sae(21); ids0=tok("Tom is a man and Sara is a woman. The man is called",return_tensors="pt").input_ids
print("recon sweep (layer_21 SAE):")
for L in (20,21,22):
    x=hs(ids0,L); r=dec(enc(x,S21),S21)
    print(f"  hs[{L}]: cos={torch.cosine_similarity(x,r,dim=-1).mean():.3f} L0={(enc(x,S21)>0).float().sum(-1).mean():.0f}")
# 2. gender-binding corpus (vary names + which is male); answer = in-context male name
pool=[" Tom"," Sara"," Anna"," Mark"," Paul"," Lucy"," David"," Emma"," Peter"," Laura"," James"," Kate"," Sam"," Rose"," Jack"," Lily"," John"," Mary"," Frank"," Grace"]
names=[n for n in pool if len(tok(n,add_special_tokens=False).input_ids)==1]
tid=lambda t: tok(t,add_special_tokens=False).input_ids[0]; rng=np.random.default_rng(0)
def ex(a,b,a_man):
    g1,g2=("man","woman") if a_man else ("woman","man")
    return f"{a} is a {g1} and{b} is a {g2}. The man is called", (a if a_man else b)
data=[]
for _ in range(40):
    a,b=rng.choice(names,2,replace=False); am=bool(rng.integers(2)); s,io=ex(a,b,am)
    ids=tok(s,return_tensors="pt").input_ids; data.append((ids,a,b,io,am))
L0=data[0][0].shape[1]; data=[d for d in data if d[0].shape[1]==L0]
t0=[tok.decode([x]) for x in data[0][0][0]]; IDX_A=[i for i,t in enumerate(t0) if t.strip()==data[0][1].strip()][0]; IDX_B=[i for i,t in enumerate(t0) if t.strip()==data[0][2].strip()][0]
mpred=lambda ids: tok.decode([model(ids).logits[0,-1].float().argmax().item()]).strip()
sp=int(len(data)*0.5); tr,ho=data[:sp],data[sp:]
acc=sum(mpred(ids)==io.strip() for ids,a,b,io,m in ho)/len(ho)
print(f"\ncorpus N={len(data)} (tr{len(tr)}/ho{len(ho)}); name pos A={IDX_A} B={IDX_B}; MODEL acc(hold)={acc:.0%}")
LAY=22  # hidden_states index validated above
S=load_sae(21)
role=torch.zeros(S["b_enc"].shape[0])
for ids,a,b,io,m in tr:
    F=enc(hs(ids,LAY),S); mp=IDX_A if io==a else IDX_B; fp=IDX_B if io==a else IDX_A; role+=F[mp]-F[fp]
role/=len(tr)
for K in (1,5):
    idx=role.abs().topk(K).indices; sgn=torch.sign(role[idx]); dm=dt=0
    for ids,a,b,io,m in ho:
        F=enc(hs(ids,LAY),S); pick=a if float((F[IDX_A,idx]*sgn).sum())>=float((F[IDX_B,idx]*sgn).sum()) else b
        dm+=pick.strip()==mpred(ids); dt+=pick.strip()==io.strip()
    print(f"  FEATURE-idiom top-{K} (layer21): detect-vs-model {dm/len(ho):.0%}  vs-truth {dt/len(ho):.0%}  feats={idx.tolist()}")
