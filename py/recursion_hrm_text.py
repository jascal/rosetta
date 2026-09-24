#!/usr/bin/env python3
"""Collect graph-depth responses from the public HRM-Text-1B checkpoint.

Budget is the count of recurrent stack applications: H * (L + 1). We vary
(H,L)=(1,1),(1,3),(2,3), giving budgets 2, 4, 8. Inference runs without a KV
cache so the recurrence count can change on the same loaded model.
"""
import argparse
import json
from pathlib import Path

from recursion_ladder import prompt, read_jsonl, write_jsonl


SCHEDULE = ((1, 1), (1, 3), (2, 3))


def collect(cases, model_dir, architecture="hrm_text_1b", threads=4,
            prompt_style="plain", schedule=SCHEDULE):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    if (model.config.H_cycles, model.config.L_cycles) != (2, 3):
        raise ValueError("expected public HRM-Text 1B checkpoint with H=2, L=3")

    def token_ids(words):
        return {ids[0] for word in words if len(ids := tok.encode(word, add_special_tokens=False)) == 1}

    yes = token_ids((" yes", " Yes", "yes", "Yes"))
    no = token_ids((" no", " No", "no", "No"))
    if not yes or not no or yes & no:
        raise ValueError("tokenizer needs distinct single-token yes/no answers")
    rows = []
    with torch.inference_mode():
        for case in cases:
            encoded = tok(prompt(case, prompt_style), return_tensors="pt")
            for h_cycles, l_cycles in schedule:
                model.config.H_cycles = h_cycles
                model.config.L_cycles = l_cycles
                output = model(**encoded, use_cache=False, logits_to_keep=1)
                logits = output.logits[0, -1]
                y = max(float(logits[t]) for t in yes)
                n = max(float(logits[t]) for t in no)
                rows.append({"id": case["id"], "architecture": architecture,
                             "prompt_style": prompt_style,
                             "budget": h_cycles * (l_cycles + 1),
                             "H_cycles": h_cycles, "L_cycles": l_cycles,
                             "answer": int(y > n), "yes_logit": y, "no_logit": n})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--architecture", default="hrm_text_1b")
    parser.add_argument("--prompt-style", choices=("plain", "fewshot"), default="plain")
    parser.add_argument("--default-only", action="store_true", help="only the released H=2,L=3 budget")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="pilot only; cannot certify a full domain")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = read_jsonl(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    schedule = SCHEDULE[-1:] if args.default_only else SCHEDULE
    rows = collect(cases, str(args.model_dir), args.architecture, args.threads,
                   args.prompt_style, schedule)
    write_jsonl(args.out, rows)
    print(json.dumps({"cases": len(cases), "budgets": [h * (l + 1) for h, l in schedule],
                      "responses": len(rows)}))


if __name__ == "__main__":
    main()
