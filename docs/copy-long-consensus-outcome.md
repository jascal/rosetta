# Beyond-12-token consensus: development family rejected

**Empirical**, fresh seeds 801/802, 408 contexts, pinned historical f32 Qwen
oracle. The [fixed family](copy-long-consensus-plan.md) tested repeated long
sources against a shorter conflicting decoy. It did not use target whitelists,
boundary guards or answers from a final test.

| Minimum match | Support | Train firing / wrong | Validation firing / wrong | `equiv.dl` mismatches / domain |
|---:|---:|---:|---:|---:|
| 16 | 2 | 164 / 15 | 164 / 30 | 45 / 328 |
| 16 | 3 | 80 / 7 | 80 / 14 | 21 / 160 |
| 24 | 2 | 136 / 7 | 136 / 22 | 29 / 272 |
| 24 | 3 | 64 / 3 | 64 / 10 | 13 / 128 |
| 32 | 2 | 80 / 7 | 80 / 2 | 9 / 160 |
| 32 | 3 | 32 / 3 | 32 / 0 | 3 / 64 |

All six firing-domain certificates fail with actual mismatches, complete
references and no uncovered obligations. Datalog's `lg_selected` is empty.
No final confirmation was activated and no final references were requested.
The strict candidate's clean validation result cannot override its failed
train gate or justify keeping only its known successful rows.

The reference-free query established positive incremental domains of 16–80
instances per split, depending on candidate. Unlike the previous unique-target
family, these candidates also fire on baseline-covered controls; both classes
remain in every certificate. The baseline answers 96 cases per split, making
zero train errors and ten validation errors. None of these data extends the
baseline's original finite certificate to this new matrix.

## Specific remaining failure

The strictest candidate's three errors all come from one train background
group (`s801-l64-g1-n0`) on unedited repeats with three true source copies.
The intended repeated successor is `ание`; the model instead emits competing
`}$`. Both 12- and 16-token decoys fail without gaps, and the 16-token decoy
also fails with eight-token gaps. The paired target substitutions succeed.
These are correlated variants, not three independent groups. They show that
even three long agreeing sources do not force this model to implement literal
long-context consensus.

The matrix has 384 positive variants and 24 independent-group controls, six
positive and six negative groups per split. Group vocabularies and both splits
are disjoint and exclude the previous research datasets. The original inputs,
every reference, paired edit, failed rule and counterexample are retained.

## What this changes

The baseline's 12-token horizon is a real algorithmic limitation, and the new
rule would cover cases outside it, but that alone does not establish faithful
conversion. More repeated evidence improves some results without closing the
semantic competition observed in the head-deletion study. Increasing match
length/support again, or filtering the Russian-fragment target after seeing
these errors, would not be an independent solution.

A useful next diagnostic is whether a context-independent next-token prior at
the final query token explains competing-source choices. Such prior observations
must be treated as new discovery inputs, not runtime model calls. Any later rule
would need a frozen model-free representation, fresh development and a separate
untouched final domain. This is a hypothesis, not evidence that the prior works.

## Replay

`reference/benchmarks/qwen25_copy_long_consensus/` contains the complete record.
Proposals remain hypotheses, not certified replacement artifacts.

```bash
python3 py/audit_copy_long_consensus.py
```

Nine Datalog queries replay model-free with unchanged verdicts and relations,
including all six failed equivalence checks. Two synthetic tests independently
check the generated all-source edits, disjoint inputs, baseline-horizon contrast,
positive admission, and rejection of every rule after a deliberate firing-case
error. The original goal remains unachieved: incremental certified coverage is
still zero.
