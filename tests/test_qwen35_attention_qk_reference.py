import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
def test_rotated_query_key_capture_matches_attention_and_restores_kernel(tmp_path):
    torch=pytest.importorskip('torch')
    from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
    from safetensors.torch import save_file
    from qwen35_attention_qk_reference import AttentionQKReference
    torch.set_num_threads(1);torch.manual_seed(3201)
    cfg=Qwen3_5TextConfig(vocab_size=37,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,layer_types=['linear_attention']*3+['full_attention'],
        linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        linear_conv_kernel_dim=4,rope_parameters={'rope_type':'default','rope_theta':10000000.,'partial_rotary_factor':.25,
        'mrope_interleaved':True,'mrope_section':[1,1,0]},tie_word_embeddings=False,attn_implementation='eager',dtype='float32')
    direct=Qwen3_5ForCausalLM(cfg).eval();(tmp_path/'config.json').write_text(json.dumps(cfg.to_dict()))
    save_file({k:v.detach().clone() for k,v in direct.state_dict().items()},tmp_path/'model.safetensors')
    oracle=AttentionQKReference(tmp_path,vocab_chunk=7);ids=[2,3,5,7,11];original=implementation.eager_attention_forward
    result=oracle.capture(ids,[(3,1),(3,2)])
    assert implementation.eager_attention_forward is original
    for head in (1,2):
        q=torch.tensor([r['value'] for r in result['rotated_query'] if r['head']==head])
        k=torch.tensor([r['value'] for r in result['rotated_keys'] if r['head']==head]).view(5,16)
        mask=torch.tensor([r['value'] for r in result['mask'] if r['head']==head])
        scale=[r['value'] for r in result['scale'] if r['head']==head][0]
        actual=torch.tensor([r['value'] for r in result['attention'] if r['head']==head])
        torch.testing.assert_close(torch.softmax((k@q)*scale+mask,dim=0),actual,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(oracle.probe(ids)['logits'],result['logits'],atol=0,rtol=0)
    with pytest.raises(ValueError):oracle.capture(ids,[(0,0)])
    assert implementation.eager_attention_forward is original
