import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
def test_attention_capture_patch_and_cleanup_match_direct_model(tmp_path):
    torch=pytest.importorskip('torch')
    from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
    from safetensors.torch import save_file
    from qwen35_attention_reference import AttentionReference
    torch.set_num_threads(1);torch.manual_seed(3001)
    cfg=Qwen3_5TextConfig(vocab_size=37,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,layer_types=['linear_attention']*3+['full_attention'],
        linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        linear_conv_kernel_dim=4,rope_parameters={'rope_type':'default','rope_theta':10000000.,'partial_rotary_factor':.25,
        'mrope_interleaved':True,'mrope_section':[1,1,0]},tie_word_embeddings=False,attn_implementation='eager',dtype='float32')
    direct=Qwen3_5ForCausalLM(cfg).eval();(tmp_path/'config.json').write_text(json.dumps(cfg.to_dict()))
    save_file({k:v.detach().clone() for k,v in direct.state_dict().items()},tmp_path/'model.safetensors')
    oracle=AttentionReference(tmp_path,vocab_chunk=7);ids=[2,3,5,7,11];heads=[(3,0),(3,1)]
    base=oracle.probe(ids,heads);captured={}
    def capture(_m,args,out):captured['attention']=out[1].detach().clone()
    hook=direct.model.layers[3].self_attn.register_forward_hook(capture)
    with torch.no_grad():expected=direct(torch.tensor([ids]),use_cache=False).logits[0,-1]
    hook.remove();torch.testing.assert_close(base['logits'],expected,atol=2e-6,rtol=2e-6)
    assert len(base['attention'])==10 and len(base['vectors'])==32
    for r in base['attention']:assert r['value']==float(captured['attention'][0,r['head'],4,r['source']])
    identity=[(r['layer'],r['position'],r['head'],r['dimension'],r['value']) for r in base['vectors']]
    torch.testing.assert_close(oracle.probe(ids,heads,identity)['logits'],base['logits'],atol=0,rtol=0)
    patches=[(3,4,1,7,-2.0),(3,0,0,1,0.0)]
    def substitute(_m,args):
        x=args[0].clone()
        for l,p,h,d,v in patches:x[0,p,h*16+d]=v
        return (x,)+args[1:]
    hook=direct.model.layers[3].self_attn.o_proj.register_forward_pre_hook(substitute)
    with torch.no_grad():expected=direct(torch.tensor([ids]),use_cache=False).logits[0,-1]
    hook.remove();actual=oracle.probe(ids,heads,patches)
    torch.testing.assert_close(actual['logits'],expected,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(oracle.probe(ids,heads)['logits'],base['logits'],atol=0,rtol=0)
    with pytest.raises(ValueError):oracle.probe(ids,[(0,0)])
    with pytest.raises(ValueError):oracle.probe(ids,heads,[(3,4,0,16,0.)])
