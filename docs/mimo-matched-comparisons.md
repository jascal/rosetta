# MiMo matched comparisons: protocol and preflight

Status: **open**. MiMo-7B-Base was downloaded at a pinned revision and given a
[full-checkpoint preflight](../experiments/mimo_preflight/README.md) on
2026-09-26. Fieldrun f16 matched the official FP32 reference on all 32 frozen
contexts for both MiMo-7B-Base and MiMo-7B-RL. The Base int8 bundle also matched
32/32; RL int8 matched 29/32, with all failures in graph prompts, and f16
recovered them. The f16 bundles are admitted as empirical build-time oracles on
this finite domain. Native BF16 remains a separate precision condition. The
remaining Qwen3.5 and MiMo-V2.6 full checkpoints and bundles remain absent
locally. The existing fieldrun MiMo and Qwen3.5 architecture results are
tiny-fixture parity results, not full-checkpoint validation. The matched
[Base/RL int8 guarded-copy comparison](../experiments/mimo_preflight/matched_guarded_copy_int8.md)
selected no guard for either model. Base raw copy matched 86/144 held-out model
answers; RL matched 122/144. The strict diagnostic guard failed exact
finite-domain equivalence for both, and the selected circuits abstained on all
144 cases. This behavioral difference needs confirmation using RL f16, which
passed all 32 frozen parity contexts.
On graph recursion, [Base int8 matches the certified rule more closely than RL
int8](../experiments/mimo_preflight/mimo_graph_recursion.md): 61/200 versus
94/200 answer mismatches, with Base also following more paired edge edits at
each tested depth. The 40-case RL f16 subset closely matches RL int8, suggesting
the gap is not caused solely by RL int8 quantization. The cross-model summary
compares these findings with earlier Qwen2.5, Pythia, Llama, HRM, and LoopFormer
runs. The completed [copy stress comparison](../experiments/mimo_preflight/copy_stress_mimo_comparison.md)
finds a small RL raw-copy gain over Base (130/288 vs 121/288), but the current
fixed diagnostic guards fail exact equivalence for both. Near-match distractors
defeat raw copying for both, and neither model selected a certified guard.
An [expanded suffix/support search](../experiments/mimo_preflight/rule_search_expansion.md)
added 42 candidates on a fresh paired dataset and still admitted none. Base's
fixed diagnostic guard was exact on its 16 fired test cases, while RL had one
mismatch; development counterexamples prevented admitting either rule.
The best Base development near-miss, frozen and transferred unchanged to RL,
was a post-hoc exploratory probe using test summaries already inspected in
earlier runs (Base exact on 16 cases, RL one mismatch), not an independent
confirmation. A separate
[top-1 margin audit](../experiments/mimo_preflight/copy_behavior_stability.md)
finds wrong copy predictions generally have lower margins, although some
correct and incorrect ranges overlap; this is not directly usable as a
standalone Datalog guard.
The same 48-candidate search certified a deterministic copy-control rule on
all 144 held-out positive cases and abstained on 16 negative controls
([control report](../experiments/mimo_preflight/rule_search_control.md)). This
supports that copy is expressible in Datalog; the remaining MiMo problem is
finding a useful rule that matches model behavior.
The [successor-plurality experiment](../experiments/mimo_preflight/successor_plurality_search.md)
tested a genuinely different source-aggregation rule on a fresh group-split
corpus. Its authored control is certified, but normal admission still selects
no rule for either MiMo checkpoint; the fixed plurality guard adds coverage
and more answer errors than consensus.
The follow-up [bounded-recency search](../experiments/mimo_preflight/bounded_recency_search.md)
likewise admits no MiMo rule. Restricting sources to recent positions increases
forced firing, but near-match distractors make those predictions even less
faithful. This points to missing source-selection behavior in the candidate
rules; the authored control still verifies that the copy rules are expressible
and certifiable in Datalog.
The next [specificity-ranked search](../experiments/mimo_preflight/source_ranked_copy_search.md)
filters out the 12-token distractor when true sources match 16 tokens. It
improves the fixed diagnostic for RL from 9/16 to 12/16 correct on plain cases,
but not for Base, and no learned guard passes admission. Exact certificates
still fail on both checkpoints, so longest-match ranking remains a useful
candidate feature rather than an established replacement circuit.
The follow-up [source-attribution audit](../experiments/mimo_preflight/source_attribution_audit.md)
finds that most source-attributable answers match a longest-suffix source, but
only 22/48 Base and 28/48 RL answers match any prior successor in the tested
range. This suggests source selection explains only part of the mismatch; a
substantial remainder needs another next-token rule family.
Fieldrun already implements both conversion paths and their tiny parity gates
([MiMo-7B](../../fieldrun/docs/MIMO.md),
[MiMo-V2.6](../../fieldrun/docs/MIMO_V26.md)); this protocol does not call for
reimplementing them. The lm-sae RoPE and Qwen3-MoE reference kernels cover
related families, but no MiMo or Qwen3.5 full-checkpoint parity result is
recorded there.

## Questions and ordering

1. **Training changes to circuits.** Compare MiMo-7B-Base with MiMo-7B-RL,
   then Qwen3.5-9B with MiMo-V2.6-Distill-Qwen-9B. Within each pair, compare
   causal copy success, selected guard, frozen test coverage, and exact
   finite-domain equivalence. Also test each base-selected guard unchanged on
   its trained counterpart. The latter is a transfer test, distinct from
   reselecting a guard for each checkpoint.
2. **Long-range copy stress.** Vary source distance, number of agreeing
   repetitions, and near-match distractors. Report raw model copying and the
   faithfulness of one previously frozen guard by stratum. MiMo-7B uses full
   attention; the Qwen3.5-derived model has three linear-attention layers
   followed by one full-attention layer. This comparison is descriptive because
   training, tokenizer, size, and architecture differ.
3. **Reasoning rule transfer.** Use fresh graph paths and paired edge edits in
   the [recursion ladder](RECURSION_LADDER.md). Report paired success by depth
   and the Datalog rule's exact equivalence to *all* recorded model answers in
   the stated domain. A failed certificate is a result. This tests behavior,
   not the model's internal mechanism.
4. **Quantization stability.** After the full-precision oracle gate, freeze
   prompts, token IDs, candidate rules, and evaluation domains. Compare oracle
   top-1 flips and margins across f32/f16/int8/int4 bundles, then run the same
   Datalog certificate separately for each. Do not attribute a model-pair
   difference to quantization.

## Gate 1: full-checkpoint oracle validation

For each of the four checkpoints, pin the exact upstream revision and record
its config, tokenizer, chat template, weight hashes, and fieldrun bundle hashes.
Collect reference and fieldrun top-1 outputs on the *same token IDs*, at every
position of a fixed short-context suite, including copy and graph prompts.
Record top-1 agreement, differing positions, and reference/fieldrun top-1
margins. An unmatched position blocks that bundle as an oracle until diagnosed.
Check resident-server startup, memory use, and prediction latency on this host.
Short contexts come first on the Qwen3.5 path because current generation
recomputes the full context. The output of this gate is a per-checkpoint parity
report with the raw token IDs and predictions; a tiny fixture is insufficient.

The initial local preflight found about 122 GB free disk and 14 GiB physical RAM.
The MiMo-7B pair has since passed the finite 32-context parity gate above;
full FP32 bundles exceeded this machine's memory. The Qwen3.5 and MiMo-V2.6
checkpoint gates remain unmeasured. Fieldrun belongs only to
build-time reference collection: emitted `circuits.dl` must run in Soufflé alone.

## Freeze before model test answers

- Use group-disjoint development and test sequences for
  [`benchmark_guarded_copy.py`](../py/benchmark_guarded_copy.py). Freeze the
  candidate guards, dataset, selected guard, and Datalog firing domain before
  collecting test references. Preserve `report.json`, `frozen.json`, and all
  certificate evidence. Compare both selected guards and cross-applied frozen
  guards on each matched pair.
- Run [`probe_families.py`](../py/probe_families.py) with identical text stimuli
  within each pair, tokenized separately when necessary. Freeze the succession
  prompt format on development data: the current probe chooses the better of
  two formats during scoring. Save trial-level predictions and paired-edit
  outcomes; its current aggregate console output is not an exact certificate.
  A top-k family detection rate is empirical and must not be labeled proved.
- Extend the [copy generalization matrix](copy-generalization.md) with
  independently varied distance, repetitions, and near-match distractors.
  Keep the query suffix and target identity controlled where possible. Freeze
  the guard and its firing domain before the first stress-test oracle call;
  report abstentions, errors, and exact `dl/equiv.dl` verdicts by stratum.
- Generate fresh graph cases, pin the prompt style and constrained yes/no
  scoring, and collect every case for every checkpoint. Preserve the complete
  response files. The recursion-ladder verifier rejects missing cases; null
  responses prevent a model-equivalence certificate.

Across architectures, token IDs need not mean the same text. Use common text
stimuli and each model's tokenizer for behavioral comparison. Exact circuit
equivalence remains a per-model, per-token-domain claim. Within a matched
checkpoint pair, verify tokenizer identity before reusing an identical token-ID
dataset or a frozen Datalog artifact.

## Sources and local entry points

- MiMo-V2.6's [model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/main/README.md)
  identifies Qwen3.5-9B as its supervised-fine-tuning source.
- The [MiMo-7B-RL config](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL/blob/main/config.json)
  and [MiMo-V2.6 config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/main/config.json)
  define the attention patterns used in the descriptive stress test.
- Rosetta's resident fieldrun oracle and Soufflé-only runtime are described in
  the [README](../README.md). Full-checkpoint parity belongs to fieldrun's
  reference comparison; Rosetta starts model measurements only after that gate.
