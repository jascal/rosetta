# Fresh source-selection development: all candidates rejected

**Empirical**, pinned f32 Qwen2.5-0.5B-Instruct; seeds 601/602, disjoint new groups
and vocabularies. No candidate qualifies and **no final-test references were
collected**. The conditional confirmation protocol was not activated.

The [fixed family](copy-longest-development-plan.md) crossed minimum matching
lengths 6/12/24 with unguarded/boundary-anchored variants. Source matching,
guards, causal admission, comparison to baseline and final selection were all
Datalog. The 684-case matrix retained both source orders, one-token leading
padding, zero/eight-token gaps, three distractor lengths, target substitutions,
and no-repeat/tied-source controls. Each split contains nine positive and nine
independent negative groups; variants within groups are correlated.

| Minimum match | Boundary guard | Train firing / wrong | Validation firing / wrong | `equiv.dl` mismatches / domain |
|---:|---|---:|---:|---:|
| 6 | absent | 324 / 80 | 324 / 42 | 122 / 648 |
| 6 | required | 108 / 28 | 108 / 18 | 46 / 216 |
| 12 | absent | 324 / 80 | 324 / 42 | 122 / 648 |
| 12 | required | 108 / 28 | 108 / 18 | 46 / 216 |
| 24 | absent | 216 / 31 | 216 / 28 | 59 / 432 |
| 24 | required | 72 / 8 | 72 / 15 | 23 / 144 |

Every candidate firing case is a baseline abstention, as established by the
reference-free query. All six equivalence checks have complete references,
zero uncovered obligations and actual mismatches. They **fail**. Minimum 6 and
12 have identical domains on this matrix and do not constitute independent
successful mechanisms. Datalog's `ls_selected` relation is empty.

## What replicated and what did not

The [historical pilot](copy-longest-source-pilot-outcome.md) showed 8/8 correct
long-first conflicting-source cases. That was a real finite observation, but it
did not support the boundary generalization:

| Order/alignment | Model follows long source / positive cases |
|---|---:|
| long first, boundary aligned | 170 / 216 |
| long last | 181 / 216 |
| long first, one leading padding token | 175 / 216 |

The reversed and padded conditions remain in every report. There is no evidence
in these aggregates for the pilot's claimed direction of a boundary advantage.
These correlated descriptive strata are not population estimates or causal
proofs of equality. The model follows the long source on 526/648 positive
variants overall; no-repeat/tie controls are reported separately, with their
authored gold treated as a scoring placeholder, not an obligation to copy.

The true computation still includes errors and alternative continuations that
longest exact matching does not explain. More structural guards selected from
these same failed rows would require a new hypothesis and new data. This study
does not admit a rule, rescue the pilot subset, or establish an anatomical copy
mechanism. Incremental **certified** coverage remains zero.

## Integrity, evidence and next uncertainty

`reference/benchmarks/qwen25_copy_longest/` retains inputs, the exact exclusion
set, model journal, reference-free domains, selector outputs, all mismatch
lists, descriptive strata, six finite-domain certificates and replay results.
The initial input protocol is also retained: a synthetic test caught an
ungrounded negated aggregate in the selector. That implementation error was
fixed before the first model request; the candidate family, inputs and admission
policy were unchanged. `protocol.json` records the predecessor hash and fix.

```bash
python3 py/audit_copy_longest.py
```

Nine queries replay without the model, preserving every verdict and relation.
The standalone programs in `proposals/` remain development hypotheses, not
faithful replacements. The final-domain/runtime checks and confirmation plan
are prospective, tested on authored controls; no real-model final success is
claimed for them.

The next useful distinction is whether wrong continuations arise from selecting
the wrong source or from transforming the right source token into a different
output. This calls for source-resolved observation and causal tests of attention
and its output, rather than another rule selected solely by input shape.
