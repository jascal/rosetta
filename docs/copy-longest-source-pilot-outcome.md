# Source-selection pilot: a signal, not a faithful unguarded rule

**Empirical discovery**, four historical background groups and 32 correlated
contexts, pinned f32 Qwen. The long-source position, short distractor length and
paired target substitutions were fixed in the [pilot plan](copy-longest-source-pilot-plan.md).

| Distractor suffix | Source order | Cases | Model follows long source | Baseline answers / wrong |
|---|---|---:|---:|---:|
| 1 token | long first | 8 | 8 | 8 / 0 |
| 1 token | long last | 8 | 6 | 8 / 2 |
| 3 tokens | long first | 8 | 8 | 0 / 0 |
| 3 tokens | long last | 8 | 5 | 0 / 0 |

On three-token distractors placed after the long source, the model follows the
long source in all eight cases, despite the other successor being more recent.
A most-recent three-token source rule agrees with the model in zero of these
eight. This is evidence of contextual source selection beyond recency on these
inputs, not a universal mechanism claim.

The six-token minimum / three-token winning-margin rule fires on all 16
three-token-distractor cases and makes three errors. The twelve-token minimum
fires on eight and makes one error. All those firing cases are baseline
abstentions. The unguarded hypotheses therefore fail equivalence; no production
rule or final confirmation is admitted from this pilot.

The order dependence is post-hoc discovery evidence. In this construction,
"long first" also means that the winning matched context begins at input
position zero. That suggests an **input-boundary anchor**, but order and source
position are confounded. It does not justify filtering this pilot to its eight
successful rows and calling the result a new faithful circuit. A boundary guard
needs new development inputs and untouched confirmation inputs.

All model answers, paired edits, baseline outputs, recency comparisons and
candidate mismatches are retained under
`reference/benchmarks/qwen25_copy_longest_pilot/`. The report's `certified()` is
input/instrument integrity only. Positive finite-domain equivalence and fresh
generalization are still **open**.
