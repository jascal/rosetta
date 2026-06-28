#!/usr/bin/env python3
"""rosetta · make_corpus.py — tokenize a text file into a model's corpus.json using its fieldrun bundle tokenizer.

Reproducible corpus building for real models (where the corpus is real text, not the model's own generation). Needs the
rosetta .venv (tokenizers) and a converted bundle with a <stem>.tokenizer.json.
Usage: .venv/bin/python py/make_corpus.py <model_dir> <text_file> [bundle_stem]
       (bundle_stem defaults to <model_dir>/bundle)
"""
import sys, os, json
from tokenizers import Tokenizer

md = sys.argv[1]
text_file = sys.argv[2]
stem = sys.argv[3] if len(sys.argv) > 3 else os.path.join(md, "bundle")
tok = Tokenizer.from_file(stem + ".tokenizer.json")
ids = tok.encode(open(text_file).read()).ids
out = os.path.join(md, "corpus.json")
json.dump({"ids": ids}, open(out, "w"))
print(f"{len(ids)} tokens from {text_file} → {out}")
