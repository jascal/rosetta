# stories110M — build recipe (model #4)

Karpathy `llama2.c` **stories110M**: dim 768, 12 layers, 12 heads (MHA), ffn 2048, **vocab 32000**, tied — a real
~110M-param Llama (GPT-2-small scale) on TinyStories; biggest in the stories series. Same fieldrun-refs path as s15M
(see `../stories15M/BUILD.md`): `stories110M.pt` → `build/convert_hf.py` → `fieldrun convert --arch rope --dtype f32
-o bundle` → minimize. Shares the llama tokenizer lexicon with stories15M. fieldrun==HF verified 6/6; runtime is
souffle-only (no fieldrun in `circuits.dl`/`run.dl`).

```bash
curl -sLO https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt
python3 build/convert_hf.py && fieldrun convert --model hf --arch rope --dtype f32 -o bundle
FIELDRUN_BIN=/path/to/fieldrun python3 py/minimize.py 99999 8 models/stories110M
```
`bundle.fieldrun.*` (418 MB f32) is gitignored — regenerate as above.
