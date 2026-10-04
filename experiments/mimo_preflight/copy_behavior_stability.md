# Copy behavior stability and top-1 margins

**Empirical diagnostic on one frozen held-out dataset.** Base and RL were
scored on the same 144 test contexts (dataset SHA256
`73d6f38ffe62f9727ca2fc03bab8edc72760d50a68d7dca1033e70c864232bff`).
Fieldrun's `/topk` first token matched the already frozen benchmark reference
on all 144 cases per checkpoint. Margins are raw top-1 minus top-2 logits,
recorded to diagnose behavior; they are not part of a candidate Datalog rule.

| Model | Task-correct top-1 margin | Task-wrong top-1 margin |
|---|---:|---:|
| Base int8 | median 5.45 (n=122; range 0.35–9.87) | median 0.83 (n=22; range 0.001–2.10) |
| RL int8 | median 6.31 (n=113; range 0.096–11.08) | median 0.57 (n=31; range 0.049–2.16) |

Wrong copy answers tend to have smaller margins, but the ranges overlap. Some
correct answers have very small margins, and some wrong answers have margins
above 2. A threshold chosen from these test margins would be test leakage; no
threshold was selected. Also, a runtime guard based directly on model logits
would violate Rosetta's Soufflé-only runtime contract unless that score were
itself represented in the standalone circuit, which this experiment does not
do.

Performance varied across randomly generated token groups with identical
structural copy rules. Base scored 8/18–18/18 by group; RL scored 5/18–18/18.
This is evidence of content sensitivity in these nonce-token prompts, not
prompt paraphrase invariance. The separate copy stress matrix already measures
noise and near-match distractors and finds those layouts substantially harder.

Raw top-two logits and per-case correctness are in
[`copy_margins_base_int8.jsonl`](copy_margins_base_int8.jsonl) and
[`copy_margins_rl_int8.jsonl`](copy_margins_rl_int8.jsonl). Their summary files
include the frozen dataset hashes and top-1 parity checks.

## Added distractors and noise

We also collected top-two scores on one frozen 96-case slice of the copy-stress
matrix (seed 2, all three distances, all four layouts, both repetition counts;
dataset SHA256
`de24284ad2de19452d7f56f7fa173654f79a419cbce478570e36e892674ffb60`). The
top-1 matched the previously saved oracle answer on all 96 contexts per model.

| Layout | Base correct | Base median margin, correct / wrong | RL correct | RL median margin, correct / wrong |
|---|---:|---:|---:|---:|
| Plain | 16/24 | 6.27 / 0.83 | 16/24 | 6.03 / 1.04 |
| Noise 8 | 14/24 | 5.72 / 0.86 | 16/24 | 5.64 / 1.67 |
| Noise 32 | 14/24 | 5.67 / 1.15 | 12/24 | 4.46 / 0.70 |
| Near-match distractors | 14/24 | 5.41 / 0.73 | 15/24 | 4.82 / 0.29 |

Across this slice, Base had 58/96 correct task answers and RL 59/96. Each
model had wrong answers with margins above 6 (Base 6.09, RL 7.21), so low
margin is informative on average but cannot exactly identify all copy errors.
These are descriptive strata from one seed, not a calibrated threshold test.
The raw stress top-two observations are in
[`copy_stress_margins_base_seed2.jsonl`](copy_stress_margins_base_seed2.jsonl)
and [`copy_stress_margins_rl_seed2.jsonl`](copy_stress_margins_rl_seed2.jsonl).
