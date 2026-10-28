"""Build-time raw head-gate and per-position value capture; no patched computation."""
from qwen35_attention_reference import AttentionReference

class AttentionFactorReference(AttentionReference):
    def capture(self,ids,heads):
        cfg=self.config;heads=[tuple(x) for x in heads];width=cfg.head_dim
        if len(set(heads))!=len(heads):raise ValueError('duplicate factor head')
        for l,h in heads:
            if not (type(l) is int and type(h) is int and 0<=l<cfg.num_hidden_layers and cfg.layer_types[l]=='full_attention' and 0<=h<cfg.num_attention_heads):raise ValueError('invalid factor head')
        groups=cfg.num_attention_heads//cfg.num_key_value_heads;hooks=[];gates=[];values=[]
        try:
            for l in sorted({l for l,h in heads}):
                module=self.model.model.layers[l].self_attn;observe=tuple(h for layer,h in heads if layer==l)
                def query(_m,args,output,l=l,observe=observe):
                    raw=output[0,-1].view(cfg.num_attention_heads,2*width)
                    for h in observe:
                        for d,v in enumerate(raw[h,width:].tolist()):gates.append({'layer':l,'head':h,'dimension':d,'value':v})
                def value(_m,args,output,l=l,observe=observe):
                    raw=output[0].view(len(ids),cfg.num_key_value_heads,width)
                    for h in observe:
                        for p,row in enumerate(raw[:,h//groups,:].tolist()):
                            for d,v in enumerate(row):values.append({'layer':l,'head':h,'source':p,'dimension':d,'value':v})
                hooks.extend((module.q_proj.register_forward_hook(query),module.v_proj.register_forward_hook(value)))
            result=self.probe(ids,heads,());result['raw_gate']=gates;result['values']=values
            return result
        finally:
            for h in hooks:h.remove()
