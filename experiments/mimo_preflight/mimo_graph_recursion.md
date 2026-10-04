# MiMo-7B recursive graph rule comparison

**Empirical model measurements on a finite domain.** The same 200 branched
graph cases from `experiments/recursion_ladder/branched_cases.jsonl` were
scored with plain text prompts and constrained yes/no choices from top-token
logits. Souffle certifies the recursive reachability circuit against all 200
generated labels (`nmiss=0`, `nuncov=0`). Model certificates compare every
recorded model answer with that circuit.

| Model / precision | Rule mismatches | Accuracy | Paired edge-edit success at depths 1–5 (20 pairs each) |
|---|---:|---:|---|
| MiMo-7B-Base int8, 200 cases | 61/200 | 139/200 | 18, 8, 5, 5, 7 |
| MiMo-7B-RL int8, 200 cases | 94/200 | 106/200 | 3, 2, 1, 0, 0 |

Because the RL int8 bundle disagreed with the FP32 reference on three graph
contexts in the 32-context parity suite, a stratified subset was also run on
RL f16. It contains trials 0–3 at each of the five depths: 40 cases total.
On this identical subset, Base int8 has 12 rule mismatches, RL int8 has 18,
and RL f16 has 19. Their paired edge-edit successes are:

| Model / precision | Mismatches / 40 | Paired edge-edit success by depth (4 pairs each) |
|---|---:|---|
| Base int8 | 12 | 2, 1, 1, 2, 2 |
| RL int8 | 18 | 0, 2, 0, 0, 0 |
| RL f16 | 19 | 0, 1, 0, 0, 0 |

RL int8 and f16 differ on 1/40 model answers in this subset. The Base/RL gap
persists at RL f16 on these cases, although the subset has only four pairs per
depth. The full 200-case RL f16 sweep was stopped because it drove memory use
to 15 GiB of swap with only 3.5 GiB free; the 40-case stratified check is the
saved precision follow-up.

For context, previous runs on the same 200-case branched domain had:

| Prior model / prompt | Rule mismatches / 200 | Paired edge-edit success by depth (20 pairs each) |
|---|---:|---|
| Qwen2.5-Coder-1.5B, few-shot | 74 | 13, 6, 3, 2, 2 |
| HRM-Text-1B, 8 calls, few-shot | 81 | 8, 5, 2, 4, 0 |
| LoopFormer, 4 steps, plain | 97 | 2, 1, 2, 2, 1 |

Base int8 has the strongest match to recursive reachability among these rows.
RL int8/f16 is weaker than Base and the prior few-shot Qwen run on paired edge
edits. Prompt differences matter: MiMo used plain prompts, while Qwen and one
HRM row used few-shot prompts. These results describe the models' answers on
this task; they do not show which internal mechanism produced them.

The complete response logits and rule verdicts are preserved in
[`mimo_base_int8_graph.json`](mimo_base_int8_graph.json),
[`mimo_rl_int8_graph.json`](mimo_rl_int8_graph.json), and
[`mimo_rl_f16_graph_subset.json`](mimo_rl_f16_graph_subset.json), with raw
responses beside each file. The generated recursive rule remains standalone
Souffle; model certificates are false wherever a model answer differs from
that rule.
