# Short-suffix development: all six candidates rejected

**Empirical**, pinned f32 Qwen2.5-0.5B-Instruct; fresh seeds 401/402, one source,
short query prefixes, varied lengths/gaps/distractors. No candidate was admitted,
and no final-test references were collected. No incremental certified coverage
was gained. The [registered development family](copy-prefix-development-plan.md)
was neither expanded nor retuned after collection.

The matrix has **624 contexts**, 312 per split, including 576 positive variants
and 48 controls. Each split has 12 positive and 12 independent negative groups.
Variants within a group are correlated. All vocabularies and groups are
disjoint across splits; the generated vocabularies exclude the historical copy
and native-trace discovery datasets recorded in `protocol.json`.

Every candidate's firing domain is a baseline abstention domain: the token-only
query found zero overlap and zero conflicts with run B's unchanged behavior-only
composite. This confirms the proposed rules would add coverage *if faithful*.

| Suffix | Minimum context length | Train firing / wrong | Validation firing / wrong | `equiv.dl` mismatches / total |
|---|---:|---:|---:|---:|
| 1 | 0 | 144 / 36 | 144 / 46 | 82 / 288 |
| 1 | 16 | 112 / 26 | 112 / 34 | 60 / 224 |
| 1 | 32 | 64 / 17 | 64 / 20 | 37 / 128 |
| 2 | 0 | 96 / 17 | 96 / 22 | 39 / 192 |
| 2 | 16 | 80 / 12 | 80 / 18 | 30 / 160 |
| 2 | 32 | 40 / 4 | 40 / 7 | 11 / 80 |

All six finite firing-domain certificates **fail**. Each has complete references
and zero uncovered obligations; the failures are actual mismatches. These are
development checks, not untouched final-test results. Causal source-substitution
successes occur in multiple groups, but do not rescue a candidate with errors.
Datalog's `cp_selected` relation is empty.

The baseline itself also fails on this new matrix: 18 wrong of 108 answered
train cases and 24 wrong of 108 answered validation cases. Its original finite
certificates remain valid on their own domains. This matrix supplies additional
counterexamples to broader faithfulness; it does not invalidate those finite
proofs. Baseline-covered three-token cases and conflicting-source controls are
reported, rather than being removed to improve the comparison.

## Interpretation and next uncertainty

Short-suffix token equality and context length do not determine this model's
continuation accurately enough for certification. The strongest candidate's
11 mismatches include token variants (`apture` versus `capture`, `_STEP`
versus `Step`, and ` artifacts` versus ` art`), as well as unrelated outputs.
These are descriptive examples from development data, not a fitted
normalization rule. Literal copying, approximate lexical reconstruction and
ordinary language continuation remain different candidate computations.

Combined with the layer-0 exchange null, these results leave the mechanism
open. A next study needs a computation hypothesis explaining the counterexamples,
with new development/confirmation boundaries. Excluding these wrong rows,
whitelisting successful targets, or collecting a final test for a rejected rule
would not satisfy the goal.

## Artifacts and checks

`reference/benchmarks/qwen25_copy_prefix/` retains the protocol, generated inputs,
excluded token set, model journal, reference-free domains, Datalog selection,
descriptive strata, all mismatch lists, and six failed `dl/equiv.dl` checks.
Standalone files under `proposals/` are **rejected hypotheses**, not delivered
faithful replacements. They require only token facts at runtime, but fail
model equivalence on the stated development domains.

```bash
python3 py/audit_copy_prefix.py
```

This checks frozen development inputs/sources and replays the nine retained
Datalog queries without the model, preserving failed verdicts too. Four new
synthetic-control tests cover admission, missing references, independent-group
requirements, exclusions and the runtime guards. Synthetic tests establish
instrument behavior, not model faithfulness.
