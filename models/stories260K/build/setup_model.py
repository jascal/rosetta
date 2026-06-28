#!/usr/bin/env python3
"""Set up rosetta/models/stories260K: whole.dl (copied), corpus.json (sampled from the model), lexicon.json (tok512)."""
import json, os, struct, torch, warnings
warnings.filterwarnings("ignore")
from transformers import LlamaForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(DST, exist_ok=True)

# lexicon from llama2.c tok512.bin (id -> display string)
vocab = []
with open(os.path.join(HERE, "tok512.bin"), "rb") as f:
    struct.unpack("i", f.read(4))
    for _ in range(512):
        struct.unpack("f", f.read(4)); l = struct.unpack("i", f.read(4))[0]
        vocab.append(f.read(l).replace(b"\xe2\x96\x81", b" ").decode("utf-8", "replace"))
json.dump({"tokens": [[v, "", ""] for v in vocab], "vocab": 512},
          open(os.path.join(DST, "lexicon.json"), "w"), ensure_ascii=False)

# corpus: sample diverse in-distribution id streams from the model
m = LlamaForCausalLM.from_pretrained(os.path.join(HERE, "hf"), torch_dtype=torch.float32); m.eval()
torch.manual_seed(0)
ids = []
for s in range(40):
    seq = [1]
    with torch.no_grad():
        for _ in range(70):
            logits = m(torch.tensor([seq])).logits[0, -1]
            probs = torch.softmax(logits / 0.9, -1)
            nxt = int(torch.multinomial(probs, 1))
            seq.append(nxt)
            if nxt == 2:
                break
    ids += seq
json.dump({"ids": ids}, open(os.path.join(DST, "corpus.json"), "w"))
import shutil
shutil.copyfile(os.path.join(HERE, "whole.dl"), os.path.join(DST, "whole.dl"))
print(f"corpus: {len(ids)} tokens ({40} samples); lexicon 512; whole.dl copied → {DST}")
