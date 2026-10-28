"""Capture actual normalized/rotated Q,K at the official eager kernel boundary."""
from qwen35_attention_factor_reference import AttentionFactorReference

class AttentionQKReference(AttentionFactorReference):
    def capture(self,ids,heads):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
        heads=[tuple(h) for h in heads];original=implementation.eager_attention_forward
        queries=[];keys=[];masks=[];scales=[];groups=self.config.num_attention_heads//self.config.num_key_value_heads
        def observe(module,query,key,value,attention_mask,**kwargs):
            layer=module.layer_idx
            for l,h in heads:
                if l!=layer:continue
                for d,v in enumerate(query[0,h,-1].tolist()):queries.append({'layer':l,'head':h,'dimension':d,'value':v})
                for p,row in enumerate(key[0,h//groups].tolist()):
                    for d,v in enumerate(row):keys.append({'layer':l,'head':h,'source':p,'dimension':d,'value':v})
                row=[0.0]*key.shape[-2] if attention_mask is None else attention_mask[0,0,-1].tolist()
                masks.extend({'layer':l,'head':h,'source':p,'value':v} for p,v in enumerate(row))
                scales.append({'layer':l,'head':h,'value':float(kwargs['scaling'])})
            return original(module,query,key,value,attention_mask,**kwargs)
        implementation.eager_attention_forward=observe
        try:
            result=super().capture(ids,heads)
            result.update({'rotated_query':queries,'rotated_keys':keys,'mask':masks,'scale':scales})
            return result
        finally:implementation.eager_attention_forward=original
