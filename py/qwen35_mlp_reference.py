"""Build-time MLP observation/exchange on the official low-memory forward graph."""
import math
from qwen35_lazy_reference import LazyReference


class MlpReference(LazyReference):
    def probe(self,ids,units=(),patches=()):
        """Observe last-position gate/up/product; patch (layer,position,neuron,value).

        Observations precede patches at their layer. Later layers see patches.
        Full logits are returned for the caller's Datalog reference query.
        """
        units=[tuple(unit) for unit in units];patches=[tuple(patch) for patch in patches];cfg=self.config
        if len(set(units))!=len(units):raise ValueError('duplicate observed unit')
        for l,n in units:
            if not (type(l) is int and type(n) is int and 0<=l<cfg.num_hidden_layers and 0<=n<cfg.intermediate_size):
                raise ValueError('invalid observed unit')
        seen=set()
        for l,p,n,v in patches:
            if not (type(l) is int and type(p) is int and type(n) is int and 0<=l<cfg.num_hidden_layers
                    and 0<=p<len(ids) and 0<=n<cfg.intermediate_size and math.isfinite(v)):
                raise ValueError('invalid patch')
            if (l,p,n) in seen:raise ValueError('duplicate patch')
            seen.add((l,p,n))
        hooks=[];gate={};up={};observed=[];applied=[]
        try:
            for l in sorted({l for l,n in units}|{l for l,p,n,v in patches}):
                module=self.model.model.layers[l].mlp
                layer_units=tuple(n for layer,n in units if layer==l)
                layer_patches=tuple((p,n,v) for layer,p,n,v in patches if layer==l)
                def capture_gate(_module,args,output,l=l):gate[l]=output[0,-1].detach().clone()
                def capture_up(_module,args,output,l=l):up[l]=output[0,-1].detach().clone()
                def patch(_module,args,l=l,layer_units=layer_units,layer_patches=layer_patches):
                    products=args[0]
                    for n in layer_units:
                        observed.append({'layer':l,'position':len(ids)-1,'neuron':n,
                            'gate':float(gate[l][n]),'value':float(up[l][n]),'product':float(products[0,-1,n])})
                    if not layer_patches:return None
                    changed=products.clone()
                    for p,n,v in layer_patches:
                        before=float(changed[0,p,n]);changed[0,p,n]=v
                        applied.append({'layer':l,'position':p,'neuron':n,'before':before,'after':float(changed[0,p,n])})
                    return (changed,)+args[1:]
                hooks.extend((module.gate_proj.register_forward_hook(capture_gate),
                    module.up_proj.register_forward_hook(capture_up),module.down_proj.register_forward_pre_hook(patch)))
            logits=self.logits(ids)
            return {'logits':logits,'observations':observed,'applied':applied}
        finally:
            for hook in hooks:hook.remove()
