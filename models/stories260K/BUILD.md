# stories260K — build recipe (model #2)

A real Llama-architecture model (Karpathy's `llama2.c` tiny model): **dim 64, 5 layers, 8 heads, 4 KV-heads (GQA),
ffn 172, vocab 512, tied embeddings**, trained on TinyStories. Chosen as rosetta's second model because it is a genuine
rope/Llama model (RMSNorm + SwiGLU + RoPE — exactly what `fieldrun emit_whole` supports) yet has a small enough vocab
(512) to clear the dense-Gram wall (`512 × 64 = 32,768` embed facts ≪ 4M).

`whole.dl` (7 MB) and its split (`whole.forward.dl`, `whole.weights/`) are **gitignored** — regenerate them:

## 1. Fetch the checkpoint + tokenizer
```bash
for f in stories260K.pt tok512.bin; do
  curl -sLO "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/$f"
done
```

## 2. llama2.c → HF Llama (config.json + model.safetensors)
Key subtlety: llama2.c uses the **interleaved** RoPE convention (Meta), HF uses rotate-half — so `wq`/`wk` need the
standard HF permute. Embeddings are **tied** (`output.weight` shares `tok_embeddings.weight`). See `build/convert_hf.py`:
strip the `_orig_mod.` prefix, `permute(wq, n_heads=8)` / `permute(wk, n_heads=4)`, map
`feed_forward.w1/w2/w3 → mlp.gate/down/up`, `attention_norm/ffn_norm → input/post_attention_layernorm`, drop the tied
`lm_head`. Config: `hidden_size 64, intermediate_size 172, num_hidden_layers 5, num_attention_heads 8,
num_key_value_heads 4, vocab_size 512, rms_norm_eps 1e-5, rope_theta 10000, tie_word_embeddings true`.

**Faithfulness checks (both passed):** greedy decode reads as TinyStories ("Once upon a time, there was a little girl
named Lily…"), and `fieldrun` argmax == HF argmax 14/14 on story prefixes.

## 3. HF → fieldrun bundle → whole.dl
```bash
fieldrun convert --model hf --arch rope --dtype f32 -o stories260k
fieldrun --bundle ./stories260k export --logic-whole --out whole.dl --maxpos 16   # 261k lines, no dense-Gram wall
```

## 4. corpus + lexicon (this dir)
`corpus.json` = id streams sampled from the model (temp 0.9, 40 samples); `lexicon.json` = id→string from `tok512.bin`.
See `build/setup_model.py`.

## 5. minimize + certify
```bash
python3 py/minimize.py 400 8 models/stories260K   # certified circuits-only program (equiv.dl)
python3 py/ngram.py   400 8 models/stories260K     # read the cover as an n-gram detector
```
The oracle auto-splits `whole.dl` into `forward.dl` + `weights/*.facts` (see `py/split_facts.py`) so souffle bulk-loads
the weights as data (~0.4 s/call instead of ~30–60 s).
