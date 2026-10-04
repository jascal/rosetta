"""Collect official-model top-two logits on frozen matched MiMo cases."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cpu-memory", default="7GiB")
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    suite_sha256 = hashlib.sha256(raw).hexdigest()
    suite = json.loads(raw)
    config_sha256 = hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest()
    if hashlib.sha256((args.checkpoint / "tokenizer.json").read_bytes()).hexdigest() != suite["tokenizer_sha256"]:
        raise ValueError("checkpoint tokenizer differs from frozen suite")
    previous = [json.loads(line) for line in args.out.read_text().splitlines()] if args.out.exists() else []
    if any(row["suite_sha256"] != suite_sha256 or row["config_sha256"] != config_sha256 or
           row["id"] != i for i, row in enumerate(previous)):
        raise ValueError("existing output does not match this suite/checkpoint")
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, dtype=torch.float32, device_map="auto",
        max_memory={"cpu": args.cpu_memory}, offload_folder="/tmp/rosetta-mimo-reference-offload",
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval()
    with args.out.open("a") as output, torch.no_grad():
        for case in suite["cases"][len(previous):]:
            logits = model(input_ids=torch.tensor([case["ids"]]), use_cache=False).logits[0, -1].float()
            top = torch.topk(logits, 2)
            row = {"id": case["id"], "family": case["family"], "suite_sha256": suite_sha256,
                   "config_sha256": config_sha256, "dtype": "fp32", "top2_ids": top.indices.tolist(),
                   "top2_logits": top.values.tolist()}
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(f"{case['id'] + 1}/{len(suite['cases'])}: {case['family']} {row['top2_ids'][0]}", flush=True)


if __name__ == "__main__":
    main()
