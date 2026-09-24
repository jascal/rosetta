#!/usr/bin/env python3
"""Collect graph-depth responses from an official LoopFormer HF checkpoint.

Requires an already downloaded local model directory. The published model uses
an elastic trajectory `steps`; for budget K, we use K equal steps summing to 1.
The model is scored by constrained yes/no logits, matching the fieldrun probes.
"""
import argparse
import json
from pathlib import Path

from recursion_ladder import prompt, read_jsonl, write_jsonl


def collect(cases, model_dir, budgets, architecture, threads=4, prompt_style="plain"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model.eval()
    # The published safetensors stores only wte for its tied output head. With
    # transformers 5.x, loading the custom 4.x wrapper reports lm_head missing
    # and leaves it untied. Restore the architecture's declared tie explicitly.
    model.gpt.lm_head.weight = model.gpt.transformer.wte.weight
    assert model.gpt.lm_head.weight.data_ptr() == model.gpt.transformer.wte.weight.data_ptr()

    def token_ids(words):
        return {ids[0] for word in words if len(ids := tok.encode(word, add_special_tokens=False)) == 1}

    yes = token_ids((" yes", " Yes", "yes", "Yes"))
    no = token_ids((" no", " No", "no", "No"))
    if not yes or not no or yes & no:
        raise ValueError("tokenizer needs distinct single-token yes/no answers")
    rows = []
    with torch.inference_mode():
        for case in cases:
            encoded = tok(prompt(case, prompt_style), return_tensors="pt", add_special_tokens=False)
            if encoded["input_ids"].shape[1] > model.gpt.config.block_size:
                raise ValueError(f"case {case['id']} exceeds model context limit")
            for budget in budgets:
                if budget < 1 or budget > 8:
                    raise ValueError("this checkpoint supports budgets 1..8")
                logits = model(**encoded, steps=[1.0 / budget] * budget).logits[0, -1]
                y = max(float(logits[t]) for t in yes)
                n = max(float(logits[t]) for t in no)
                rows.append({"id": case["id"], "architecture": architecture, "budget": budget,
                             "prompt_style": prompt_style,
                             "answer": int(y > n), "yes_logit": y, "no_logit": n})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--architecture", default="loopformer_3x8_fineweb300k")
    parser.add_argument("--prompt-style", choices=("plain", "fewshot"), default="plain")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="pilot only; cannot certify a full domain")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = read_jsonl(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    rows = collect(cases, str(args.model_dir), args.budgets, args.architecture, args.threads,
                   args.prompt_style)
    write_jsonl(args.out, rows)
    print(json.dumps({"cases": len(cases), "budgets": args.budgets, "responses": len(rows)}))


if __name__ == "__main__":
    main()
