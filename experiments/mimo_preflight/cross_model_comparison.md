# MiMo-7B results beside earlier Rosetta models

This is a descriptive comparison across saved Rosetta experiments. Claims are
empirical on each stated finite domain; the task protocols, prompts, tokenizers,
and model sizes differ across experiments.

## Guarded copy

| Model / run | Selected guard | Raw copy correct | Guarded result |
|---|---|---:|---|
| Qwen2.5-0.5B-Instruct, seed 1 | `(suffix 3, 2 sources)` | 137/144 | selected guard exact on 96/96 firing cases |
| MiMo-7B-Base int8, seed 7 | none | 86/144 | selected circuit abstains; strict diagnostic guard 12/16 correct, 4 wrong |
| MiMo-7B-RL int8, seed 7 | none | 122/144 | selected circuit abstains; strict diagnostic guard 14/16 correct, 2 wrong |

On the matched Base/RL set, RL was correct on 40 cases where Base was wrong;
Base was correct on 4 where RL was wrong. This suggests RL changed copy
behavior on the synthetic suite. The matched runs use identical token IDs
because the MiMo tokenizers match. The Qwen run used a different seed and
tokenizer, so compare its aggregate numbers only as context. RL int8 also has
three graph-family parity flips on the separate 32-context suite; f16 copy
confirmation remains open.

The Qwen fixed-guard generalization run got 280/288 positive cases right and
answered none of 54 negative controls; raw copy got 209/288 and answered 36/54
controls. The matched MiMo distance/distractor matrix is now complete; see
[`copy_stress_mimo_comparison.md`](copy_stress_mimo_comparison.md). On that
domain, RL improves raw copy by 9/288 over Base and the diagnostic guard gets
35 versus 27 cases right, but both guards make errors and neither certifies.
Unlike Qwen, these are not admitted selected circuits: Rosetta selected no
guard for either MiMo checkpoint.

An expanded search over 48 suffix/support combinations on a fresh paired
dataset also selected no guard. The [search report](rule_search_expansion.md)
shows that the best three-source candidates still had development errors and
causal-pair failures. The fixed diagnostic guard certified on Base's 16-case
test firing domain but failed on RL's (one mismatch); that test result did not
retroactively change admission.

The [post-hoc Base-to-RL transfer probe](guard_transfer_base_to_rl/REPORT.md)
froze the best Base development near-miss (suffix 7, three agreeing sources). It
certified on Base's 16 fired test cases and failed on RL's with one mismatch.
This is an explicitly non-admitted candidate, not a certified Base circuit;
the result is descriptive because the test summary had already been inspected.
The [top-1 margin diagnostic](copy_behavior_stability.md) finds lower median
margins for wrong copy answers in both models, but the ranges overlap; these
logits cannot directly serve as a standalone Datalog guard.

The [deterministic copy-control search](rule_search_control.md) certified the
expanded family on all 144 held-out positive cases and abstained on all 16
negative controls. This shows the copy circuit is expressible and certifiable
for the authored rule; it does not guarantee that MiMo follows that rule.

The new [successor-plurality search](successor_plurality_search.md) adds a
distinct rule concept and passes the authored single-conflict control, but
does not improve MiMo admission. Its suffix-9/three-source diagnostic fired on
24/48 rather than consensus's 16/48, while errors rose from 5 to 10 for Base
and 6 to 11 for RL; neither policy certified on the model-answer domain.

## Recursive graph rule

Each row below compares model answers with the same Souffle reachability rule
on the 200-case branched domain. `Mismatches` count failure of exact model/rule
equivalence. Paired success means both the reachable graph and its paired
edge-edited unreachable graph receive the correct answer.

| Model / prompt | Mismatches / 200 | Paired edge-edit success at depths 1–5 (20 pairs each) |
|---|---:|---|
| MiMo-7B-Base int8, plain | 61 | 18, 8, 5, 5, 7 |
| MiMo-7B-RL int8, plain | 94 | 3, 2, 1, 0, 0 |
| Qwen2.5-Coder-1.5B, few-shot | 74 | 13, 6, 3, 2, 2 |
| HRM-Text-1B, 8 stack calls, few-shot | 81 | 8, 5, 2, 4, 0 |
| LoopFormer, 4 steps, plain | 97 | 2, 1, 2, 2, 1 |

Base int8 shows the strongest graph-rule agreement in this table; RL int8 is
weaker than Base and the earlier Qwen run on paired edge edits. On the
stratified 40-case RL f16 follow-up, Base int8 has 12 rule mismatches, RL int8
has 18, and RL f16 has 19; RL f16 differs from RL int8 on just one answer in
that subset. The precision-matched subset therefore retains the Base/RL gap,
with only four paired trials per depth. Qwen's prompt is few-shot, unlike the
MiMo plain prompt. The model outputs are constrained yes/no choices from
top-token logits, not unrestricted generation. Full results are in
[`mimo_graph_recursion.md`](mimo_graph_recursion.md).

The earlier Pythia 70m and Llama 3.2 1b results on the smaller 40-case chain
domain were near chance with no paired edge-edit successes at any depth. That
domain differs from this branched set, so it is only a broad reference point.

## Family screen

No saved earlier `probe_families.py` results were found for direct comparison.
On the new 10-trial screen, both MiMo checkpoints pass the 80%/80% screen on
12/15 families. Base also passes coreference and set membership; RL falls below
threshold on their causal-follow scores. Both are poor on IOI name-mover
copying. Details and scoring limits are in [`families/README.md`](families/README.md).

## Precision and status

Base f16/int8 each match the official FP32 top-1 answers on the 32-context
parity suite. RL f16 matches 32/32; RL int8 matches 29/32, with all three
flips on graph prompts. The current matched copy runs use int8 for both models;
the graph comparison is repeated on RL f16 to separate that quantization
effect. The full MiMo-V2.6/Qwen3.5 pair is not locally available, so no claim
about hybrid attention is made here.
