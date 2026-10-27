"""Build-time full-attention observations/substitutions on the official graph."""
import math
from qwen35_lazy_reference import LazyReference

class AttentionReference(LazyReference):
    def probe(self,ids,heads=(),patches=()):
        """Observe last-query attention and post-gate head vectors; substitute scalars.

        Patches: (layer,position,head,dimension,value), immediately before o_proj.
        Gated DeltaNet layers are intentionally unsupported by this probe.
        """
        cfg=self.config;heads=[tuple(h) for h in heads];patches=[tuple(p) for p in patches]
        width=cfg.head_dim;count=cfg.num_attention_heads
        def valid_head(l,h):
            return type(l) is int and type(h) is int and 0<=l<cfg.num_hidden_layers and cfg.layer_types[l]=='full_attention' and 0<=h<count
        if len(set(heads))!=len(heads):raise ValueError('duplicate head')
        if any(not valid_head(l,h) for l,h in heads):raise ValueError('invalid full-attention head')
        seen=set()
        for l,p,h,d,v in patches:
            if not (valid_head(l,h) and type(p) is int and 0<=p<len(ids) and type(d) is int and 0<=d<width and math.isfinite(v)):raise ValueError('invalid head patch')
            if (l,p,h,d) in seen:raise ValueError('duplicate head patch')
            seen.add((l,p,h,d))
        hooks=[];attention=[];vectors=[];applied=[]
        try:
            for l in sorted({l for l,h in heads}|{l for l,p,h,d,v in patches}):
                module=self.model.model.layers[l].self_attn
                observe=tuple(h for layer,h in heads if layer==l)
                substitutions=tuple((p,h,d,v) for layer,p,h,d,v in patches if layer==l)
                def before(_module,args,l=l,observe=observe,substitutions=substitutions):
                    x=args[0]
                    for h in observe:
                        for d,v in enumerate(x[0,-1,h*width:(h+1)*width].tolist()):
                            vectors.append({'layer':l,'position':len(ids)-1,'head':h,'dimension':d,'value':v})
                    if not substitutions:return None
                    changed=x.clone()
                    for p,h,d,v in substitutions:
                        original=float(changed[0,p,h*width+d]);changed[0,p,h*width+d]=v
                        applied.append({'layer':l,'position':p,'head':h,'dimension':d,'before':original,'after':float(changed[0,p,h*width+d])})
                    return (changed,)+args[1:]
                def after(_module,args,output,l=l,observe=observe):
                    weights=output[1]
                    if observe and weights is None:raise ValueError('attention weights unavailable')
                    for h in observe:
                        for p,v in enumerate(weights[0,h,-1].tolist()):
                            attention.append({'layer':l,'head':h,'position':len(ids)-1,'source':p,'value':v})
                hooks.append(module.o_proj.register_forward_pre_hook(before))
                hooks.append(module.register_forward_hook(after))
            scores=self.logits(ids)
            return {'logits':scores,'attention':attention,'vectors':vectors,'applied':applied}
        finally:
            for h in hooks:h.remove()
