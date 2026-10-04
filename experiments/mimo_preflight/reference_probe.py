"""Run the official MiMo-7B checkpoint on the fixed eight-window parity domain."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    ids = json.loads((Path(__file__).parent / "ids.json").read_text())["holdout_ids"]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
        max_memory={"cpu": "7GiB"},
        offload_folder="/tmp/rosetta-mimo-reference-offload",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    rows = []
    with torch.no_grad():
        for position in range(16, 24):
            logits = model(input_ids=torch.tensor([ids[position - 16:position]]), use_cache=False).logits[0, -1].float()
            top = torch.topk(logits, 2)
            rows.append({"position": position, "top2_ids": top.indices.tolist(), "top2_logits": top.values.tolist()})
    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
