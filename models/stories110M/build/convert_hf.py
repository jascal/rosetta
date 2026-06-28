#!/usr/bin/env python3
"""Convert Karpathy's llama2.c stories260K checkpoint → HF Llama format (config.json + model.safetensors).

llama2.c follows the Meta/original-Llama RoPE convention (interleaved pairs); HF Llama uses rotate-half. So wq/wk must
be permuted (the standard HF conversion permute). Everything else is a straight key rename. Untied lm_head (output.weight).
"""
import json, os, torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hf")
os.makedirs(OUT, exist_ok=True)

ck = torch.load(os.path.join(HERE, "stories110M.pt"), map_location="cpu", weights_only=False)
A = ck["model_args"]
sd = {k.replace("_orig_mod.", ""): v.float() for k, v in ck["model"].items()}
D, NL, NH, NKV, V, FF = A["dim"], A["n_layers"], A["n_heads"], A["n_kv_heads"], A["vocab_size"], None
FF = sd["layers.0.feed_forward.w1.weight"].shape[0]


def permute(w, n_heads):
    """interleaved (llama2.c/Meta) → HF rotate-half layout for q/k projections."""
    d0, d1 = w.shape
    hd = d0 // n_heads
    return w.view(n_heads, hd // 2, 2, d1).transpose(1, 2).reshape(d0, d1)


# stories260K ties output.weight to tok_embeddings.weight (shared_classifier) → tied=1, no separate lm_head.
tied = sd["output.weight"].data_ptr() == sd["tok_embeddings.weight"].data_ptr() or \
    torch.equal(sd["output.weight"], sd["tok_embeddings.weight"])
hf = {"model.embed_tokens.weight": sd["tok_embeddings.weight"].clone(),
      "model.norm.weight": sd["norm.weight"]}
if not tied:
    hf["lm_head.weight"] = sd["output.weight"]
for i in range(NL):
    p = f"layers.{i}."
    o = f"model.layers.{i}."
    hf[o + "self_attn.q_proj.weight"] = permute(sd[p + "attention.wq.weight"], NH)
    hf[o + "self_attn.k_proj.weight"] = permute(sd[p + "attention.wk.weight"], NKV)
    hf[o + "self_attn.v_proj.weight"] = sd[p + "attention.wv.weight"]
    hf[o + "self_attn.o_proj.weight"] = sd[p + "attention.wo.weight"]
    hf[o + "mlp.gate_proj.weight"] = sd[p + "feed_forward.w1.weight"]
    hf[o + "mlp.down_proj.weight"] = sd[p + "feed_forward.w2.weight"]
    hf[o + "mlp.up_proj.weight"] = sd[p + "feed_forward.w3.weight"]
    hf[o + "input_layernorm.weight"] = sd[p + "attention_norm.weight"]
    hf[o + "post_attention_layernorm.weight"] = sd[p + "ffn_norm.weight"]

save_file({k: v.contiguous() for k, v in hf.items()}, os.path.join(OUT, "model.safetensors"))
config = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": D, "intermediate_size": FF, "num_hidden_layers": NL,
    "num_attention_heads": NH, "num_key_value_heads": NKV, "vocab_size": V,
    "max_position_embeddings": A["max_seq_len"], "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
    "hidden_act": "silu", "tie_word_embeddings": bool(tied), "torch_dtype": "float32",
    "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
}
json.dump(config, open(os.path.join(OUT, "config.json"), "w"), indent=2)
print(f"wrote {OUT}: {len(hf)} tensors, tied={tied}, config {[NL,NH,NKV,D//NH,D,FF,V,int(tied)]}")
print("  embed facts:", V * D, " unembed facts:", V * D, " (dense-Gram wall ~4M → fine)")
