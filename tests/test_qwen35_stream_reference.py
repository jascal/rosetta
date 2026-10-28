"""The memory-bounded oracle must reproduce ordinary full-model logits."""
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))


@pytest.mark.parametrize('tied,sharded,bf16',[(False,True,True),(True,False,False)])
def test_streamed_hybrid_matches_full_reference(tmp_path,tied,sharded,bf16):
    torch=pytest.importorskip('torch')
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
    from qwen35_stream_reference import StreamReference
    from qwen35_lazy_reference import LazyReference
    torch.set_num_threads(1);torch.manual_seed(9190)
    cfg=Qwen3_5TextConfig(vocab_size=37,hidden_size=32,intermediate_size=48,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        layer_types=['linear_attention']*3+['full_attention'],linear_num_key_heads=1,
        linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,linear_conv_kernel_dim=4,
        rope_parameters={'rope_type':'default','rope_theta':10000000.,'partial_rotary_factor':.25,
            'mrope_interleaved':True,'mrope_section':[1,1,0]},
        tie_word_embeddings=tied,attn_implementation='eager',dtype='float32')
    model=Qwen3_5ForCausalLM(cfg).eval()
    if bf16:model=model.bfloat16().float()
    state={name.replace('model.','model.language_model.',1) if name.startswith('model.') else name:
        tensor.detach().to(torch.bfloat16 if bf16 else torch.float32).contiguous().clone()
        for name,tensor in model.state_dict().items() if not(tied and name=='lm_head.weight')}
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'qwen3_5','text_config':cfg.to_dict()}))
    if sharded:
        groups=[{},{}];mapping={}
        for j,(name,tensor) in enumerate(state.items()):
            shard=j%2;groups[shard][name]=tensor;mapping[name]=f'part-{shard}.safetensors'
        for shard,group in enumerate(groups):save_file(group,tmp_path/f'part-{shard}.safetensors')
        (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))
    else:save_file(state,tmp_path/'model.safetensors')
    streamed=StreamReference(tmp_path,vocab_chunk=7)
    lazy=LazyReference(tmp_path,vocab_chunk=7)
    for ids in ([2],[2,3,5,7,11],[11,7,5]):
        with torch.no_grad():expected=model(torch.tensor([ids]),use_cache=False).logits[0,-1]
        actual=streamed.logits(list(ids))
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(lazy.logits(list(ids)),expected,atol=2e-6,rtol=2e-6)
    with pytest.raises(ValueError):streamed.logits([])
    with pytest.raises(ValueError):streamed.logits([37])
