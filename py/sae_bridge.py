#!/usr/bin/env python3
"""rosetta · sae_bridge.py — phase 1 of the SAE/feature bridge: does the model do higher-order ENTITY reasoning in a
SPARSE-FEATURE basis that the TOKEN vocabulary can't express? (Hypothesis-validation before building the Datalog bridge.)

Substrate: pythia-160m + EleutherAI/sae-pythia-160m-32k (TopK SAE, k=32, 32768 latents, residual stream, one per layer;
SAE at layers.L hooks gpt_neox.layers[L] output = hidden_states[L+1]; reconstruction cos>0.96 confirms model+hookpoint).

Test: IOI minimal pairs differing only in the SUBJECT name — clean ABB "...A and B... B gave to"→A vs corrupt ABA
"...A and B... A gave to"→B. The FINAL token is "to" in both, so the surface token can't distinguish the answer; the
entity binding (who is the indirect object) lives in the residual/features. Activation-patch (corrupt→clean) at each layer:
  - residual patch  = upper bound of that layer/position's causal contribution to the flip;
  - SAE-feature patch (swap only the SAE-reconstructed part, keep the residual error) = how much the SPARSE basis carries.
Metric = logit-diff recovery, LD = logit(IO)-logit(S). RESULT (layers 8-11): the token-IDENTICAL END position recovers
63-81% via features (tokens carry 0 there) → entity reasoning is feature-carried; the causal locus migrates subject→END
across layers (the name-mover). Forge tax ~20-35% (residual carries more than the sparse basis) routes to the backstop.

Needs the SAE deps (requirements-sae.txt): torch (CPU ok), transformers, safetensors. Auto-downloads the SAE on first run.
Usage: .venv/bin/python py/sae_bridge.py
"""

import glob, os, torch, numpy as np
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
M="EleutherAI/pythia-160m"; tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
SNAP=glob.glob(f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--EleutherAI--sae-pythia-160m-32k/snapshots/*")[0]
def load_sae(L):
    p=glob.glob(f"{SNAP}/layers.{L}/*.safetensors")[0]; W={}
    with safe_open(p,framework="pt") as f:
        for k in f.keys(): W[k]=f.get_tensor(k).float()
    return W["encoder.weight"],W["encoder.bias"],W["W_dec"],W["b_dec"]
def mksae(L):
    We,be,Wd,bd=load_sae(L); K=32
    def enc(x):
        pre=(x-bd)@We.T+be; v,i=pre.topk(K,-1); v=v.clamp(min=0)
        a=torch.zeros_like(pre); a.scatter_(-1,i,v); return a
    def dec(a): return a@Wd+bd
    return enc,dec
pool=[" John"," Mary"," Tom"," Sara"," Paul"," Anna"," Mark"," Lucy"," David"," Emma"," Peter"," Laura"," James"," Kate"," Henry"," Alice"]
names=[n for n in pool if len(tok(n,add_special_tokens=False).input_ids)==1]
rng=np.random.default_rng(0)
mk=lambda a,b,subj:f"When{a} and{b} went to the store,{subj} gave a drink to"
tid=lambda t: tok(t,add_special_tokens=False).input_ids[0]
pairs=[tuple(rng.choice(names,2,replace=False)) for _ in range(24)]
def hs(ids,idx): return model(ids,output_hidden_states=True).hidden_states[idx][0]
def patched_ld(ids,L,pos,vec,io,s):
    def hook(m,i,o):
        ten=o[0] if isinstance(o,tuple) else o; ten=ten.clone(); ten[:,pos,:]=vec
        return (ten,)+tuple(o[1:]) if isinstance(o,tuple) else ten
    h=model.gpt_neox.layers[L].register_forward_hook(hook)
    try: lg=model(ids).logits[0,-1]
    finally: h.remove()
    return (lg[tid(io)]-lg[tid(s)]).item()
print(f"{'layer':>5} {'recon':>6} {'ENDtok-identical: resid/feat':>30} {'pos9 subj(tok-diff): resid/feat':>32}")
for L in (6,7,8,9,10,11):
    enc,dec=mksae(L); idx=L+1
    cos=[]; cl=[]; co=[]; er=[]; ef=[]; pr9=[]; pf9=[]
    for a,b in pairs:
        cid=tok(mk(a,b,b),return_tensors="pt").input_ids; kid=tok(mk(a,b,a),return_tensors="pt").input_ids
        if cid.shape!=kid.shape: continue
        io,s=a,b; ce=hs(cid,idx); ke=hs(kid,idx); P=cid.shape[1]-1
        cos.append(torch.cosine_similarity(ce,dec(enc(ce)),dim=-1).mean().item())
        lc=patched_ld(cid,L,0,ce[0],io,s); lk=model(kid).logits[0,-1]; lk=(lk[tid(io)]-lk[tid(s)]).item()
        cl.append(lc); co.append(lk)
        # END (pos P, token "to" identical): residual vs feature-only swap
        er.append((patched_ld(cid,L,P,ke[P],io,s)-lc)/(lk-lc+1e-9))
        pf=ce[P]-dec(enc(ce[P]))+dec(enc(ke[P])); ef.append((patched_ld(cid,L,P,pf,io,s)-lc)/(lk-lc+1e-9))
        # pos9 subject (token differs) sanity
        pr9.append((patched_ld(cid,L,9,ke[9],io,s)-lc)/(lk-lc+1e-9))
        pf9c=ce[9]-dec(enc(ce[9]))+dec(enc(ke[9])); pf9.append((patched_ld(cid,L,9,pf9c,io,s)-lc)/(lk-lc+1e-9))
    print(f"{L:>5} {np.mean(cos):>6.3f}   END resid={np.mean(er):>+5.0%} feat={np.mean(ef):>+5.0%}        pos9 resid={np.mean(pr9):>+5.0%} feat={np.mean(pf9):>+5.0%}")
