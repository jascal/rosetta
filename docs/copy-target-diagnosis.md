# Controlled diagnosis of the remaining copy failures

The [fixed-guard stress test](copy-generalization.md) produced eight model disagreements, all in length-8 sequence
variants with noise. Inspecting the recorded token IDs shows a more specific pattern: five errors copy token
11436 (`".\n\n`), and three copy token 34341 (` ],\r\n`). The model returns related punctuation/newline tokens instead.
Replacing these targets with token 2579 (`istory`) or token 1602 (` very`) makes the same paired contexts succeed.

This is **empirical discovery evidence**. Sequence length, query-prefix length, and target identity were confounded
in that experiment. It does not establish that short sequences cause the failures or that punctuation always fails.
The new experiment holds the Datalog guard fixed and crosses those factors to distinguish their effects.

## Result

**Empirical:** the unchanged guard matches Qwen on **180/192** diagnostic cases. Punctuation/newline targets match
85/96; lexical targets match 95/96. Among 96 paired punctuation-to-lexical substitutions, Datalog reports **11
recoveries, one regression, and 84 pairs correct on both sides**. Both newly chosen punctuation targets also fail
once, so the failures are not confined to the two original token IDs.

| Controlled factor | Condition | Correct / total |
|---|---|---:|
| Target category | Punctuation/newline | 85/96 |
| Target category | Lexical | 95/96 |
| Sequence length | 8 | 86/96 |
| Sequence length | 12 | 94/96 |
| Query-prefix length | 4 | 88/96 |
| Query-prefix length | 6 | 92/96 |
| Gap | 0 | 63/64 |
| Gap | 8 | 58/64 |
| Gap | 32 | 59/64 |

The exact certificate on the entire diagnostic domain **fails** with `ndomain=192`, `nmiss=12`, `nuncov=0`, and no
missing or invalid references. See the [Qwen report](../reference/benchmarks/qwen25_05b_target_diagnostic/REPORT.md)
and [retained verifier result](../reference/benchmarks/qwen25_05b_target_diagnostic/certificate-evidence/check-za8svp3s/certificate.json).
The original 96-case certificate remains scoped to its original recorded domain.

The lexical regression is case 81: the circuit copies `istory` (2579), while Qwen predicts `history` (18844), despite
length 12 and prefix length 6. Case 98 fails even with no gap. Case 156 replaces the copied `);\n` (317) with `;\n`
(280). These counterexamples rule out treating lexical-only targets, longer prefixes, longer sequences, or zero gaps
as sufficient conditions by themselves on the measured domain. No combination was selected after viewing outcomes.

**Open hypothesis:** context-sensitive formatting or word completion can override literal copying. The observed token
changes are consistent with that account; this experiment does not identify a model-internal mechanism. A useful next
probe would vary the token immediately preceding the query's suffix while keeping source targets and copy support
fixed, with fresh target tokens and backgrounds. Any resulting normalization detector needs independent holdout
evaluation before joining the runtime circuit library.

## Design

There are four fresh background groups: seeds 5 and 6 at sequence lengths 8 and 12. For every group, query-prefix
length is independently set to 4 or 6, and the gap between source copies/query is set to 0, 8, or 32. All contexts
have two prior copies. Each of those 24 background/prefix/gap cells is evaluated with all eight target tokens:

| Punctuation/newline target | Paired lexical target | Origin |
|---|---|---|
| 11436: `".\n\n` | 2579: `istory` | observed failure and its successful replacement |
| 34341: ` ],\r\n` | 1602: ` very` | observed failure and its successful replacement |
| 317: `);\n` | 23268: ` apple` | newly chosen targets |
| 698: `"\n` | 13551: ` garden` | newly chosen targets |

The 192 cases are correlated variants of **four** background groups. Non-target tokens exclude all tokens used in
the previous three Qwen datasets and all eight targets; the targets are deliberately shared across backgrounds to
measure target identity. Only the two copied-from successor positions change within a cell's target sweep. The query
suffix, noise tokens, source positions, and every other token stay fixed.

These target categories are declared experimental metadata. The runtime has no token-class lookup or filter, and
no new guard is selected. This is a controlled diagnostic experiment informed by known failures, **not** untouched
final-test evidence. Tokenizer pieces and tokenizer/model hashes are recorded in `protocol.json`.

## Trusted measurements

`py/diagnose_copy_targets.py` copies the existing frozen guard and freezes its artifact, all cases, and the entire
firing domain before requesting references. It then stages the references and paired edits for Datalog.
`dl/copy_target_diagnostic.dl` checks that every declared edit is a copied-from successor, every such successor is
edited, and no other token changes. It counts each pair as both correct, both wrong, recovered, regressed, or
unanswered. Counts by target, category, sequence length, prefix length, gap, and seed are also computed in Datalog.

`dl/equiv.dl` checks the entire 192-case diagnostic domain. A failed certificate remains a failed certificate; neither
target classes nor successful subsets are promoted into a runtime rule after observing the references. The
[authored control](../reference/benchmarks/copy_control_target_diagnostic/REPORT.md) certifies 192/192 and reports all
96 punctuation-to-lexical pairs as both correct. That validates the instrument on the authored copy task.

## Reproduce

```bash
python3 py/diagnose_copy_targets.py /tmp/copy-target-control --oracle copy-control
fieldrun --bundle /path/to/bundle --serve 8187
python3 py/diagnose_copy_targets.py /tmp/qwen-target-diagnosis --oracle fieldrun \
  --port 8187 --bundle /path/to/bundle
```

Use a new output directory. The Qwen bundle must match the prior experiment's recorded hashes. On this Mac,
Soufflé preprocessing requires `DEVELOPER_DIR=/Library/Developer/CommandLineTools`.
The report links retained inputs, query outputs, source snapshots and hashes under `certificate-evidence/`;
`python3 py/certificate.py <evidence-directory>` replays the recorded verdict without the model.
