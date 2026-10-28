import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
def test_raw_gate_value_capture_reconstructs_distinct_grouped_kv_heads(tmp_path):
    torch=pytest.importorskip('torch')
    from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
    from safetensors.torch import save_file
    from qwen35_attention_factor_reference import AttentionFactorReference
    torch.set_num_threads(1);torch.manual_seed(3101)
    cfg=Qwen3_5TextConfig(vocab_size=37,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,layer_types=['linear_attention']*3+['full_attention'],
        linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        linear_conv_kernel_dim=4,rope_parameters={'rope_type':'default','rope_theta':10000000.,'partial_rotary_factor':.25,
        'mrope_interleaved':True,'mrope_section':[1,1,0]},tie_word_embeddings=False,attn_implementation='eager',dtype='float32')
    direct=Qwen3_5ForCausalLM(cfg).eval();(tmp_path/'config.json').write_text(json.dumps(cfg.to_dict()))
    save_file({k:v.detach().clone() for k,v in direct.state_dict().items()},tmp_path/'model.safetensors')
    model=AttentionFactorReference(tmp_path,vocab_chunk=7);ids=[2,3,5,7,11];heads=[(3,1),(3,2)]
    data=model.capture(ids,heads)
    for layer,head in heads:
        w=torch.tensor([r['value'] for r in data['attention'] if r['head']==head])
        g=torch.tensor([r['value'] for r in data['raw_gate'] if r['head']==head])
        v=torch.tensor([r['value'] for r in data['values'] if r['head']==head]).view(5,16)
        y=torch.tensor([r['value'] for r in data['vectors'] if r['head']==head])
        torch.testing.assert_close((w@v)*torch.sigmoid(g),y,atol=2e-6,rtol=2e-6)
    with torch.no_grad():expected=direct(torch.tensor([ids]),use_cache=False).logits[0,-1]
    torch.testing.assert_close(data['logits'],expected,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(model.probe(ids)['logits'],data['logits'],atol=0,rtol=0)
    with pytest.raises(ValueError):model.capture(ids,[(0,0)])
