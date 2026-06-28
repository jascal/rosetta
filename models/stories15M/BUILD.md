# stories15M — build recipe (model #3)

Karpathy `llama2.c` **stories15M**: dim 288, 6 layers, 6 heads (MHA, no GQA), ffn 768, **vocab 32000**, tied — a real
15.2M-param Llama on TinyStories. First model on the **fieldrun-refs path**: the dense-Gram wall (9.2M embed facts) and
the per-call throughput wall made the pure whole.dl forward impractical, so refs come from the **fieldrun binary at
build time** (fast, faithful — `fieldrun==HF` 14/14). The minimized `circuits.dl` has **no fieldrun dependency at
runtime** (souffle-only; see AGENTS runtime-independence invariant).

## Reproduce
1. **Checkpoint → HF → fieldrun bundle** (see `../stories260K/BUILD.md` for the llama2.c→HF converter; same `build/convert_hf.py`):
   ```bash
   curl -sLO https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt
   python3 build/convert_hf.py                       # → hf/ (config.json + model.safetensors)
   fieldrun convert --model hf --arch rope --dtype f32 -o bundle   # → bundle.fieldrun.{json,bin}  (gitignored)
   ```
2. **corpus.json** = id streams sampled from the model (temp 0.9). **lexicon.json** = a `{"tokens":[],"vocab":32000}`
   stub (ids shown raw; the llama tokenizer needs sentencepiece, not required for minimization).
3. **Minimize + certify** (build-time fieldrun refs; runtime is souffle-only):
   ```bash
   FIELDRUN_BIN=/path/to/fieldrun python3 py/minimize.py 3000 8 models/stories15M
   ```
   Result: **882/882 decisions CERTIFIED** (`nmiss=0 ∧ nuncov=0`), 658 rules, effective order 4-gram.
4. **Run the minimized model, souffle only** (no fieldrun):
   ```bash
   printf '0\t0\t<id0>\n0\t1\t<id1>\n…' > ctx/tok.facts
   souffle models/stories15M/run.dl -F ctx -D out      # out/cdecide.csv = (inst, argmax id)
   ```

`bundle.fieldrun.*` (58 MB) and `whole.*` are gitignored — regenerate as above. Committed: `corpus.json`,
`lexicon.json`, `circuits.dl` (the minimized program), `run.dl` (standalone harness), `CERTIFICATE.md`, `build/`.
