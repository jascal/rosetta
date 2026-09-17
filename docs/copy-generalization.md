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
