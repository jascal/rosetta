# Corrupted-query pilot: no faithful incremental coverage

**Empirical discovery result**, 24 corrupted contexts on four historical run-C
background groups. These are correlated variants, not a fresh final-test domain.
The [pilot protocol](copy-corruption-pilot-plan.md), inputs and four approximate
alignment hypotheses were fixed before any new model request.

| Query corruption | Cases | Model = copy target | Baseline answers | Baseline wrong |
|---|---:|---:|---:|---:|
| Final token | 8 | 1 | 0 | 0 |
| One token before final | 8 | 8 | 8 | 0 |
| Three tokens before final | 8 | 8 | 8 | 0 |

All 24 model references were collected from the pinned historical f32 bundle
and executable. Datalog verified exactly one declared query-token change per
case, the unchanged context positions, and the paired source-target edits.
All references are complete. The report retains every candidate prediction,
mismatch, abstention, baseline overlap and paired causal success.

The six-token / two-source approximate circuit fires on all 24 cases and makes
seven errors, all on final-token corruptions. Its eight incremental cases are
precisely those final-token corruptions: **seven mismatches out of eight**.
The twelve-token / two-source circuit fires on the 12 long-sequence variants,
with three mismatches; its four incremental cases contain those three errors.
Three-source hypotheses abstain everywhere because the pilot has two source
copies. Empty domains cannot certify.

This is a negative result for the proposed extension. Keeping only earlier-token
corruptions would duplicate already-covered behavior. Keeping only the one
successful final-token case would select the domain using model answers.
Neither satisfies the goal. No final test or production rule is admitted.

## What the result changes

Long-context approximate alignment does not explain away the failures of exact
suffix matching. In these cases the model tolerates query corruption *before*
the final token but rarely tolerates changing the final token itself. Together
with the layer-0 same-final-token exchange null, this supports investigating an
exact final-token anchor plus contextual source selection. It does not prove a
particular head or anatomical mechanism.

The next computational question is how the model selects between **conflicting
earlier successors of the same final token**. A uniquely longer exact context
match could identify the source while the current baseline abstains because its
short matches conflict and its longest-match route requires repeated target
tokens. This is a proposed new computation hypothesis, not a result of this pilot.
It needs its own fixed inputs and tests; it must not become a new guard on
known successful pilot cases.

Evidence: `reference/benchmarks/qwen25_copy_corruption_pilot/`. `report.json` and
its retained query inputs/outputs are the source of the table. The query's
`certified()` relation means instrument integrity, **not model equivalence**.
