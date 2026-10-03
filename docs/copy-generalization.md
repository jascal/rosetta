# Fixed-guard generalization experiment

This experiment keeps the [previously certified guarded-copy circuit](guarded-copy-benchmark.md) unchanged and tests
it on new seeds, sequence lengths, source distances, and distractors. Both the guarded circuit and its unguarded-copy
comparison are copied **byte-for-byte** from the previous frozen artifacts. There is no new selection, training,
threshold adjustment, or successful-case filtering.

## Preregistered test matrix

The experiment uses seeds **2, 3, 4**, sequence lengths **8, 12, 24**, and two independent sequence groups for each
seed/length combination: **18 positive groups** in total. Vocabularies are disjoint across groups and exclude every
token used in both earlier Qwen datasets (seeds 0 and 1). Tokens remain sampled from `[200, 40000)`.

Each positive group supplies two or three previous copies of its nonce sequence, followed by the first half of that
sequence. Thus the query-prefix lengths are 4, 6, and 12 respectively; length comparisons change both sequence and
prefix length. Each combination has a base case and an intervention changing every earlier occurrence of the target
token, while leaving the final query suffix unchanged. Four layouts are fixed before any oracle reference:

| Layout | Construction |
|---|---|
| `plain` | Contiguous sequence copies, matching the earlier experiment's structure |
| `noise8` | Eight unrelated tokens between each source copy and the next copy/query |
| `noise32` | Thirty-two unrelated tokens between each source copy and the next copy/query |
| `near_match8` | Eight-token gaps containing the query's last two tokens followed by a decoy; the preceding token differs, so the fixed three-token guard should ignore the distractor |

The resulting **288 positive test cases** contain correlated variants of those 18 groups, not 288 independent
observations. Contexts range up to 180 tokens. Another **18 independent groups** supply 54 negative controls:
no-repeat contexts, one-source contexts, and conflicting-source contexts. The fixed guard should abstain on these.
Negative-control target values in the authored oracle are placeholders for scoring; abstention is the relevant check.

## Freeze and trusted checks

`py/benchmark_copy_generalization.py` verifies the source circuit hashes against the prior `frozen.json`, copies the
runtime programs, generates the complete dataset, and records the protocol. `dl/guard_domain.dl` computes each
candidate's firing domain using token facts alone. Only after `frozen.json` has recorded all artifact, harness,
dataset, protocol, and score-verifier hashes does the first oracle query occur. Completed observations are flushed
to `oracle-observations.jsonl` as they arrive.

Both circuits fire on all 288 positive cases. Any model disagreement therefore fails both the full-test and
frozen-firing-domain equivalence obligations. Negative controls remain separately visible as answered/abstained
counts; they are not silently removed from the evaluation report. Missing references or changes to frozen inputs
fail the evaluation.

`dl/holdout.dl` checks evaluation input integrity and counts outcomes. The appended `dl/holdout_strata.dl` counts
results by seed, length, layout, exposure count, and intervention kind, and emits counterexamples. These strata are
descriptive diagnostics, never inputs to circuit selection. Exact equivalence is separately decided by
`dl/equiv.dl`; all inputs, verifier sources, raw relation outputs, and hashes are retained for replay.

## Qwen result

On the fresh Qwen2.5-0.5B-Instruct run, the unchanged guard answered all 288 positive stress cases and got **280/288
correct** (97.22% precision). It abstained on all 54 negative controls. Both the full and firing-domain certificates
fail with `nmiss=8`: the guard fired on every positive case, so its 288-case firing domain is the complete test set.
This is **empirical** generalization data and an **open** residual, not a certified Qwen circuit on the full matrix.

The eight errors are all repeat cases with length 8; none occur on interventions. By layout, precision is 100% on
plain, 95.83% on noise8, 94.44% on noise32, and 98.61% on near_match8. Lengths 12 and 24 are 96/96; length 8 is
88/96. All failures appear in short-sequence noisy conditions in this sample. Target-token identity and query-prefix
length are confounded with that observation: the eight failures involve just two punctuation/newline targets, and
the paired lexical substitutions succeed. See the [controlled target diagnosis](copy-target-diagnosis.md).
The group-correlated matrix contains 18 sequence groups, so these counts do not support a population claim.

| Circuit | Positive test | Negative answers | Firing-domain certificate |
|---|---:|---:|---|
| Fixed guard | 280/288 | 0/54 | fails: 8 mismatches / 288 obligations |
| Raw copy | 209/288 | 36/54 | fails: 79 mismatches / 288 obligations |

The [full Qwen report](../reference/benchmarks/qwen25_05b_copy_generalization/REPORT.md) includes every
counterexample. No rule was changed after seeing them.

## Controls

The [authored copy control](../reference/benchmarks/copy_control_generalization/REPORT.md) checks that the unchanged
guard implements the task under these transformations: **288/288** positive cases correct, all **54** negative
controls abstained, and an exact certificate over the finite 288-case domain. Unguarded copy gets **216/288** right;
the 72 near-match distractor cases expose its one-token matching rule. This establishes instrument behavior on the
authored task, not LLM faithfulness.

The [constant-output control](../reference/benchmarks/constant_control_generalization/REPORT.md) deliberately disagrees
with the unchanged guard on all **288** positive cases and fails equivalence. Because this is a fixed-artifact
experiment, the runner must not replace that circuit with an abstaining alternative after seeing failures.

## Withheld pairs and the binding-blind baseline

This section adapts the generalization test of McCoy, Soulos, Linzen and Smolensky (2026, *The Emergent Symbolic
Structure of Artificial Neural Networks*, arXiv:2608.29530, §8). A copy target is a **(token, slot)** pair: the token
filling a left-to-right slot of the copied sequence. A pair is **known** if the guard's selection data
(`train`/`validation` of the prior experiment) showed that token in that slot; otherwise it is **novel**. Let **k** be
the number of novel pairs among the unconsumed slots `[prefix, length)`.

The **strong binding-blind baseline** knows every known pair and which tokens fill the unconsumed slots. It cannot bind a
novel token to its slot, so it places the novel tokens randomly. It predicts the queried slot correctly with
probability 1 if that pair is known, else `1/k` (the slot-wise form of the paper's `1/n!`). `dl/binding_baseline.dl`
computes k, the baseline and the per-k counts. It is appended to every strata verifier, so each report includes it.
Like the other strata, these are descriptive diagnostics, never selection inputs.

**The preregistered matrix above is all-novel by construction.** Vocabularies are disjoint from the selection data, so
`k = length − prefix` and k is confounded with length. Rescoring the recorded Qwen run uses its stored references and
makes no oracle queries (`--rescore`;
[report](../reference/benchmarks/qwen25_05b_copy_generalization_binding/REPORT.md)):

| k (= length − prefix) | Cases | Model = gold | Binding-blind expected hits |
|---:|---:|---:|---:|
| 4 | 96 | 88 | 24.0 |
| 6 | 96 | 96 | 16.0 |
| 12 | 96 | 96 | 8.0 |
| all | 288 | 280 | 48.0 |

**The `--protocol withheld-pairs` dataset varies k at a fixed length**, 12 with prefix 6. The first k unconsumed slots
get fresh tokens, so the queried slot is novel iff k ≥ 1. The rest reuse a token that the selection data showed in that
same slot. The design uses seeds 5–7, k = 0…6, two or three exposures, plain and `noise8` layouts, and paired
interventions: 168 positive and 21 no-repeat negative cases. Interventions substitute a fresh token, so design k = 0
interventions are measured at k = 1; Datalog counts k from the facts, not from the design label.

The selection bundle (`7bb34aa9…`) is not on this machine. This run therefore used the local Qwen2.5-0.5B-Instruct
fieldrun bundle (`13068ab3…`), with `--allow-bundle-mismatch` recorded as `same_bundle_as_selection: false`
([report](../reference/benchmarks/qwen25_05b_withheld_pairs/REPORT.md)):

| k | Cases | Model = gold | Binding-blind expected hits | Guard = model |
|---:|---:|---:|---:|---:|
| 0 | 12 | 12 | 12.0 | 12 |
| 1 | 36 | 36 | 36.0 | 36 |
| 2 | 24 | 24 | 12.0 | 24 |
| 3 | 24 | 24 | 8.0 | 24 |
| 4 | 24 | 24 | 6.0 | 24 |
| 5 | 24 | 24 | 4.8 | 24 |
| 6 | 24 | 24 | 4.0 | 24 |
| all | 168 | 168 | 82.8 | 168 |

The guard abstained on all 21 negatives.

**What this table says, and only this.** Everything below is scoped to bundle `13068ab3…`.
- **Proved over this finite domain:** the frozen guard artifact and raw copy each have an exact certificate over these
  168 positive cases, relative to **this bundle's** recorded references.
- **Empirical:** on this bundle the model agrees with gold on 168/168 at every k, while the binding-blind baseline
  falls from 1 to 1/6.

**What it does not say.**
- **It does not confirm the guard selection.** The guard was selected against bundle `7bb34aa9…`, and this is a
  different checkpoint. That confirmation is the pending re-run on the selection bundle.
- **It is weak evidence of binding.** Flat accuracy across k is what any position-copy rule produces, the guard's own
  rule included. It rules out a lookup over seen (token, slot) pairs, which would fall towards the baseline. It does
  not show systematic role-filler binding in the paper's sense.
- **What k measures.** k is a property of the *dataset*: novelty against the selection data file. That file is fixed
  and bundle-independent, so k stays well defined here. But it describes this bundle's inputs, not anything the
  selection process saw on this bundle.

The bundle-independent result of this section is the rescore above (280/288 against an expected 48). The same columns
apply unchanged to a lexicalized circuit (for example the train-only n-gram baseline), whose agreement would be
expected to fall with k.

```bash
python3 py/benchmark_copy_generalization.py /tmp/withheld --protocol withheld-pairs --oracle copy-control
python3 py/benchmark_copy_generalization.py /tmp/withheld-qwen --protocol withheld-pairs --oracle fieldrun \
  --port 8187 --bundle /path/to/bundle [--allow-bundle-mismatch]
python3 py/benchmark_copy_generalization.py /tmp/binding --rescore reference/benchmarks/qwen25_05b_copy_generalization
```

## Reproduce and replay

```bash
python3 py/benchmark_copy_generalization.py /tmp/copy-generalization --oracle copy-control
python3 py/benchmark_copy_generalization.py /tmp/constant-generalization --oracle constant-control

# Serve the same local Qwen bundle used to develop the fixed guard, then run:
fieldrun --bundle /path/to/bundle --serve 8187
python3 py/benchmark_copy_generalization.py /tmp/qwen-generalization --oracle fieldrun \
  --port 8187 --bundle /path/to/bundle
```

The model-bundle hashes must match the prior experiment. The default artifact source is
`reference/benchmarks/qwen25_05b_guarded_seed1`; output directories must be new. On this Mac, Soufflé preprocessing
requires `DEVELOPER_DIR=/Library/Developer/CommandLineTools`. The caller must ensure the indicated bundle is actually
served; file hashes record provenance, not server attestation.

The report's certificate entries contain relative evidence paths. Replaying a successful or failed verdict needs
only Python and Soufflé, with no model:

```bash
python3 py/certificate.py <run-directory>/certificate-evidence/<check-directory>
```

`report.json` includes per-condition counts and every counterexample's original instance ID, reference token,
circuit token, seed, group, length, layout, and case kind. `dataset.json` contains the corresponding token sequences.
The runtime artifact itself remains Soufflé-only. A clean certificate proves equality only over its stated finite
domain; empirical generalization measurements do not establish universal faithfulness.
