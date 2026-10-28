"""The probe must match direct resident-model product substitution."""
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))


def test_product_patch_matches_direct_reference_and_removes_hooks(tmp_path):
    torch=pytest.importorskip('torch')
    from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
    from safetensors.torch import save_file
    from qwen35_mlp_reference import MlpReference
    torch.set_num_threads(1);torch.manual_seed(9193)
    cfg=Qwen3_5TextConfig(vocab_size=37,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,layer_types=['linear_attention']*3+['full_attention'],
        linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        linear_conv_kernel_dim=4,rope_parameters={'rope_type':'default','rope_theta':10000000.,
        'partial_rotary_factor':.25,'mrope_interleaved':True,'mrope_section':[1,1,0]},
        tie_word_embeddings=False,attn_implementation='eager',dtype='float32')
    direct=Qwen3_5ForCausalLM(cfg).eval()
    (tmp_path/'config.json').write_text(json.dumps(cfg.to_dict()))
    save_file({k:v.detach().clone() for k,v in direct.state_dict().items()},tmp_path/'model.safetensors')
    oracle=MlpReference(tmp_path,vocab_chunk=7);ids=[2,3,5,7,11];units=[(0,0),(0,1),(3,0)]
    baseline=oracle.probe(ids,units)
    torch.testing.assert_close(oracle.probe(ids,[list(u) for u in units])['logits'],baseline['logits'],atol=2e-6,rtol=2e-6)
    identity=[(r['layer'],r['position'],r['neuron'],r['product']) for r in baseline['observations']]
    torch.testing.assert_close(oracle.probe(ids,units,identity)['logits'],baseline['logits'],atol=2e-6,rtol=2e-6)
    patches=[(0,0,0,0.),(0,4,1,1.5),(3,4,0,-2.)]
    hooks=[]
    for layer in (0,3):
        def patch(_module,args,layer=layer):
            x=args[0].clone()
            for l,p,n,v in patches:
                if l==layer:x[0,p,n]=v
            return (x,)+args[1:]
        hooks.append(direct.model.layers[layer].mlp.down_proj.register_forward_pre_hook(patch))
    with torch.no_grad():expected=direct(torch.tensor([ids]),use_cache=False).logits[0,-1]
    for hook in hooks:hook.remove()
    observed=oracle.probe(ids,units,patches)
    torch.testing.assert_close(observed['logits'],expected,atol=2e-6,rtol=2e-6)
    # A following unpatched request must not retain either patch values or hooks.
    torch.testing.assert_close(oracle.probe(ids,units)['logits'],baseline['logits'],atol=2e-6,rtol=2e-6)
    with pytest.raises(ValueError):oracle.probe(ids,units,[(0,5,0,0.)])
    with pytest.raises(ValueError):oracle.probe(ids,units,[(0,0,0,float('nan'))])
