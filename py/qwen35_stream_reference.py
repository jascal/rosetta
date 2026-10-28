"""Build-time text-only FP32 oracle with one resident Qwen3.5 decoder block.

Weights are loaded exactly from their stored dtype and widened to FP32. No
quantization, model decisions, circuit mining or runtime artifact lives here.
"""
import json
from pathlib import Path


class WeightReader:
    def __init__(self,directory):
        from safetensors import safe_open
        self.directory=Path(directory)
        index=self.directory/'model.safetensors.index.json'
        if index.exists():
            self.files=json.loads(index.read_text())['weight_map']
        else:
            self.files={}
            for path in sorted(self.directory.glob('*.safetensors')):
                with safe_open(path,framework='pt',device='cpu') as stream:
                    for key in stream.keys():
                        if key in self.files:raise ValueError(f'duplicate weight: {key}')
                        self.files[key]=path.name
        if not self.files:raise ValueError('no safetensors weights')
        if 'model.language_model.embed_tokens.weight' in self.files:self.prefix='model.language_model.'
        elif 'model.embed_tokens.weight' in self.files:self.prefix='model.'
        else:raise ValueError('missing text embedding')

    def tensor(self,key):
        from safetensors import safe_open
        with safe_open(self.directory/self.files[key],framework='pt',device='cpu') as stream:
            return stream.get_tensor(key)

    def rows(self,key,start,end):
        from safetensors import safe_open
        with safe_open(self.directory/self.files[key],framework='pt',device='cpu') as stream:
            return stream.get_slice(key)[start:end]


class StreamReference:
    def __init__(self,directory,vocab_chunk=1024):
        from transformers import Qwen3_5TextConfig
        if vocab_chunk<1:raise ValueError('vocab_chunk must be positive')
        self.reader=WeightReader(directory);self.vocab_chunk=vocab_chunk
        config=json.loads((Path(directory)/'config.json').read_text())
        config=config.get('text_config',config)
        if config.get('model_type') not in ('qwen3_5_text','qwen3_5'):
            raise ValueError('unsupported text architecture')
        self.config=Qwen3_5TextConfig.from_dict(config)
        self.config._attn_implementation='eager'

    def logits(self,ids):
        import torch
        from transformers.masking_utils import create_causal_mask
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,Qwen3_5RMSNorm,Qwen3_5TextRotaryEmbedding)
        cfg=self.config;reader=self.reader;prefix=reader.prefix
        if not ids or any(type(t) is not int or t<0 or t>=cfg.vocab_size for t in ids):
            raise ValueError('nonempty in-vocabulary integer token ids required')
        with torch.inference_mode():
            embedding=prefix+'embed_tokens.weight'
            hidden=torch.cat([reader.rows(embedding,t,t+1).float() for t in ids]).unsqueeze(0)
            text_positions=torch.arange(len(ids)).unsqueeze(0)
            positions=text_positions.unsqueeze(0).expand(3,1,-1)
            rotary=Qwen3_5TextRotaryEmbedding(config=cfg,device='cpu')
            position_embeddings=rotary(hidden,positions)
            causal_mask=create_causal_mask(config=cfg,inputs_embeds=hidden,attention_mask=None,
                past_key_values=None,position_ids=text_positions)
            for layer_index in range(cfg.num_hidden_layers):
                with torch.device('meta'):block=Qwen3_5DecoderLayer(cfg,layer_index)
                layer_prefix=f'{prefix}layers.{layer_index}.'
                state={key[len(layer_prefix):]:reader.tensor(key).float()
                    for key in reader.files if key.startswith(layer_prefix)}
                block.load_state_dict(state,strict=True,assign=True)
                block.eval()
                hidden=block(hidden,position_embeddings=position_embeddings,
                    attention_mask=None if cfg.layer_types[layer_index]=='linear_attention' else causal_mask,
                    position_ids=text_positions,past_key_values=None,use_cache=False)
                del block,state
            norm=Qwen3_5RMSNorm(cfg.hidden_size,eps=cfg.rms_norm_eps)
            norm.weight.copy_(reader.tensor(prefix+'norm.weight').float())
            last=norm(hidden)[0,-1]
            output='lm_head.weight'
            if output not in reader.files:
                if not cfg.tie_word_embeddings:raise ValueError('missing untied output weights')
                output=embedding
            scores=torch.empty(cfg.vocab_size,dtype=torch.float32)
            for start in range(0,cfg.vocab_size,self.vocab_chunk):
                end=min(start+self.vocab_chunk,cfg.vocab_size)
                weight=reader.rows(output,start,end).float()
                scores[start:end]=weight @ last
                del weight
            return scores
