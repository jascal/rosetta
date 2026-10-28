"""Official Qwen3.5 forward graph with disk-backed, FP32-per-block weights.

This build-time oracle retains the upstream forward/attention/state code.
Only weight materialization and embedding/output row access are replaced.
"""
from qwen35_stream_reference import StreamReference


class LazyReference(StreamReference):
    def __init__(self,directory,vocab_chunk=1024):
        import torch
        from transformers import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
        super().__init__(directory,vocab_chunk)
        reader=self.reader;cfg=self.config
        embedding=reader.prefix+'embed_tokens.weight'
        output='lm_head.weight' if 'lm_head.weight' in reader.files else embedding
        if output==embedding and not cfg.tie_word_embeddings:raise ValueError('missing untied output weights')

        class EmbeddingRows(torch.nn.Module):
            def forward(self,ids):
                rows=[reader.rows(embedding,int(t),int(t)+1).float() for t in ids.reshape(-1).tolist()]
                return torch.cat(rows).reshape(*ids.shape,cfg.hidden_size)

        class OutputRows(torch.nn.Module):
            def forward(self,hidden):
                scores=torch.empty((*hidden.shape[:-1],cfg.vocab_size),dtype=torch.float32)
                for start in range(0,cfg.vocab_size,vocab_chunk):
                    end=min(start+vocab_chunk,cfg.vocab_size)
                    weights=reader.rows(output,start,end).float()
                    scores[...,start:end]=hidden @ weights.T
                return scores

        with torch.device('meta'):self.model=Qwen3_5ForCausalLM(cfg)
        self.model.model.embed_tokens=EmbeddingRows()
        self.model.lm_head=OutputRows()
        self.model.model.rotary_emb=Qwen3_5TextRotaryEmbedding(config=cfg,device='cpu')
        self.model.model.norm.load_state_dict({'weight':reader.tensor(reader.prefix+'norm.weight').float()},assign=True)
        for i,block in enumerate(self.model.model.layers):
            layer_prefix=f'{reader.prefix}layers.{i}.'
            def materialize(module,args,layer_prefix=layer_prefix):
                state={name[len(layer_prefix):]:reader.tensor(name).float()
                    for name in reader.files if name.startswith(layer_prefix)}
                module.load_state_dict(state,strict=True,assign=True)
            def release(module,args,output):module.to_empty(device='meta')
            block.register_forward_pre_hook(materialize)
            block.register_forward_hook(release)
        self.model.eval()

    def logits(self,ids):
        import torch
        if not ids or any(type(t) is not int or t<0 or t>=self.config.vocab_size for t in ids):
            raise ValueError('nonempty in-vocabulary integer token ids required')
        with torch.inference_mode():
            return self.model(torch.tensor([ids]),use_cache=False,logits_to_keep=1).logits[0,-1]
