# Native traces as a prior for standalone circuit discovery — run B

Status: **open; amended before any run-B model observation.** The original
registration is commit `8135116`; the pre-collection repair is documented in
[`native-trace-runb-repair.md`](native-trace-runb-repair.md). Run A is closed and
its frozen evidence and zero incremental certified coverage remain unchanged.

## Why run B exists

Run A fitted features to distinguish correct from incorrect strict-copy outputs
on positive copy inputs only. Its fresh training majority signature was uniform,
including all negative controls. Population mismatch is a plausible contributor,
not an established causal explanation. Run B instead fits positive-copy versus
negative-control labels on a fresh select split. This changes the feature's task:
it tests whether task-class information improves rule discovery, not whether it
predicts correctness by construction. Both controls and exact admission remain.

The changes below are registered together. A positive outcome would support
this combined design on its stated domain; it would not isolate which change
helped. No run-B answers have been collected or used to choose this amendment.

## Change 1 — a negative-bearing selection split, disjoint from signing

**Feature selection runs on train; the trace signature is computed on
validation.** These are disjoint rows. Nothing observed on validation may
influence which neuron is selected, its threshold, or its polarity.

The selection negative class is redefined to include negative controls:

```datalog
% classes for feature selection, on the selection split only
label(I,1) :- sample(I,"select",_), pos(I,_).
label(I,0) :- sample(I,"select",_), neg(I,_).
```

where `pos` marks a positive copy context and `neg` a negative control. Both
classes must be nonempty and each must have at least **8** instances; otherwise
that layer/channel contributes no vote and the shortfall is reported. This
replaces run A's rule-agreement labelling. Rule agreement is still computed and
retained, but only as a reported diagnostic, never as a selection label.

Run B's train and validation splits each carry 18 negative controls by
construction, so the selection class is negative-bearing by design rather than by
luck.

## Change 2 — a plain signed-margin prior

For chosen feature `f`, define `m(I,f) = P_f * (V(I,f) - T_f)` and
`S(I) = sum_f m(I,f)`. Threshold and polarity are fitted on select only.
A sign row is trace-good exactly when `S(I) > 0`. A zero margin contributes
zero; negative margins retain their actual magnitudes. With no chosen features,
`S(I)=0` and the non-degeneracy gate stops the run.

**Pre-collection amendment:** remove the proposed `|gate|/mean|gate|` multiplier.
Raw preactivation magnitude is not SwiGLU openness: `SiLU(g)=g/(1+exp(-g))`
suppresses strongly negative gates even when `|g|` is large. An MoE routing
weight times an expert activation is a different quantity. Product observations
already include SiLU gating. There is no extra gate weight in run B.
Gate observations and their select means remain descriptive evidence, and the
actual gate channel is recorded for every feature, including value features.

Margins use their own raw channel units without normalization. Consequently
higher-scale channels can dominate the sum. This is a declared limitation and
fixed design choice, not a claim of causal attribution or comparable feature
units. No post-collection rescaling, weighting, or threshold adjustment is allowed.
Equal-magnitude margins reproduce a majority only in the absence of ties.

Feature selection, margins, sums, non-degeneracy checks, and ranking are Datalog.
Python stages observations and serializes the resulting relations.

## Non-degeneracy gate — run A's missing precondition

Before the trace term is used to rank anything, assert on the signing split:

- the signature is **not uniform** (both `trace_good` and its complement are
  nonempty), and
- `trace_good` does not cover every negative control. All-negative-bad is
  permitted; the condition does not require variation within that stratum.

If either assertion fails, the run **stops and is reported as degenerate**. It
does not proceed to ranking, does not fall back to behavior-only, and does not
substitute a different rule. A degenerate run is a completed, publishable null.

This precondition is the direct lesson of run A and is the single most important
line in this document.

## Model, scope, and oracle gate

Unchanged from run A: Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, f32 fieldrun bundle, greedy decoding.

Run B introduces **no new oracle**. It runs the same validated `0948cbe`
executable and the same f32 bundle as run A, so run A's upstream FP32/eager
preflight *is* run B's gate rather than being a separate one. This is stated
because run B differs from run A only in selection and voting, never in what
produces the answers — a different oracle would confound the comparison the
experiment exists to make.

The gate is therefore attested, not re-run: `py/run_native_trace2.py`'s
`model_gate` hashes the binary, both bundle files, and the preflight
`report.json` and `reference.json`, and refuses to collect unless they equal what
the preflight recorded. `tests/test_native_trace2_oracle.py` checks the same
identities, and additionally checks the refusal path so the attestation cannot
pass vacuously.

Verified on the pinned copy used for run A — binary
`54de8fc8fd5acec0…67ea62`, bundle `3a2fb8e5…f77f8d` / `5c013a3a…aaa12`,
preflight report `6d313afd…d3150`, reference `e58ed6ead6…da3b9c` — all matching
run A's recorded `model.json`.

## Datasets and information boundaries

Fresh generation only. **No run-A input, reference, trace, domain, or rule may be
regenerated, reused, or refrozen.** Run A's `frozen.json` must keep hashing to
`451c1650f98c973310f960154c905afc92ae72542c14c6b8b61d04ee347e327e` for the whole
of run B; run B has its own manifest with its own digest.

Seeds **201 / 202 / 203** designate select / sign / test. They are disjoint from
run A's 101/102/103. Same shape as run A: lengths 8/12/24, two groups per length,
exposures 2/3, layouts plain/noise8/noise32/near_match8, paired all-source target
substitutions; 96 positive and 18 negative controls per split. Exclude every token
used in the historical Qwen copy datasets, then exclude earlier split
vocabularies, then exclude run A's vocabularies. Group identity, input hashes, and
vocabularies are disjoint across select/sign/test **and** across runs A and B.
Generate and hash all inputs before collecting any run-B reference.

Inputs are generated by `py/run_native_trace2.py`, not hand-assembled, and are
committed at `reference/benchmarks/qwen25_native_trace_b/` (`dataset.json`
sha256 `80bce2e349f8b67e3edc12b1fc89b94d5386b9f97d44fee78cbec116dead0634`).
The generator excludes 6,418 tokens: every token in the six historical Qwen copy
datasets **and** every token in run A's own frozen `dataset.json` and
`discovery.json`, so no run-B input shares an embedding with the completed run.
`tests/test_native_trace2_inputs.py` regenerates the committed dataset bit for
bit and fails closed on any shared vocabulary, group, or excluded token.

Information boundary: select and sign references may inform admission. Test
references are forbidden before artifact/domain freeze. No test traces.

## Candidate vocabulary and equal budgets

Unchanged: the same 1,344 token-derived candidates (24 bases × (1 + 10 + 45)
guard compositions) in the same order, the same top-12 proposal budget, ties by
candidate ID, and no replacement of rejected candidates. All three arms receive
the identical universe and proposal budget.

Behavior score `B = sign_correct - 4 * sign_wrong`, computed on the **sign**
split. Trace score `= B + sum_{I where candidate fires} S(I)`, using thresholds
and polarities fitted on the disjoint select split. The shuffled arm uses the
identical formula with signature rows permuted by seed 204. Scores and their
relative weight are fixed.

Note the split discipline this implies: the select split fits feature
*parameters* and is never used to rank; the sign split computes the signature and
ranks but never influences a threshold or polarity; test is untouched
until after freeze. A trace value is therefore never summed on a row whose
feature parameters were fitted from it.

## The shuffled control is mandatory

Run A's shuffled arm is what made the degeneracy visible instead of leaving a
uniform prior looking like weak-but-real signal. It is retained, unweakened, for
run B. Specifically preregistered:

- If trace and shuffled scores come out elementwise identical, report the
  identity. With continuous margins it need not imply constant signatures.
- A run-B result where the trace arm beats shuffled is not sufficient on its own;
  the trace arm must also beat **behavior-only** on certified coverage.
- If trace fails to separate shuffled, no reranking, threshold adjustment, or
  feature substitution may be performed to create a separation.

## Intervention diagnostic

Unchanged in spirit and now a genuine contrast: for every selected unit, zero
its post-SwiGLU product at the last position on the first eight **positive** and
eight **negative-control** sign-split rows by ID, plus an opposite-class patch
and an identity patch, with a same-layer control neuron `(n+1) mod width`. Run A
did not use this contrast — its "incorrect" class was 7 positive contexts.
Run B's is a real negative-control population. Still a diagnostic, never an
admission criterion; local causal influence only, never universal necessity or a
unique mechanism. Nulls preserved. The sign collection records the logit of the
input generator's `expected` target; interventions read both that target and the
unmodified greedy answer. For negative controls the generated target is only a
readout, not an assertion about the correct model answer. The diagnostic checks
identity equality and completeness against the Datalog plan before freeze.
A degenerate selection stops before diagnostic/freeze/final and retains a
terminal report plus replayable selection evidence.

## Admission, runtime, and certification

Unchanged: zero errors on both select and sign, firing in at least two
independent groups per split, correct paired-substitution following in at least
two groups per split, controls included in the zero-error check and reported
separately. Freeze every candidate, composite, wrapper, program hash, feature
parameters, selection evidence, inputs, and firing domains before the first test
oracle query.

**Runtime independence is absolute and unchanged.** Token facts and an explicit
instance domain in; decisions and explicit abstentions out; Souffle alone. No
neural observations, weights, fieldrun, Python, or `whole.dl` at runtime. The
residual must never fall back to the model. Empty selections emit an explicit
abstaining artifact.

Check every admitted individual and each composite through `dl/equiv.dl` on its
frozen firing domain and on the full test domain including negative controls.
Retain all mismatches and raw verdicts; never remove a failure after seeing
references. A clean firing-domain certificate proves only that stated finite
subset. An empty firing domain is not a proof.

## Reporting and completion

Endpoint is unchanged: positive incremental **finite-domain certified** coverage
for the trace composite over **both** controls. Raw agreement and errors are
reported independently of that strict metric.

Per arm report proposed/admitted, firing and abstention, correct/wrong,
positive/negative strata, individual and composite certificates, program bytes,
rule counts, incremental certified-domain sets versus both controls, and
observation/search cost. Additionally report the run-B non-degeneracy assertions
and their values, the per-feature descriptive gate means, and the signature distribution on
the sign split by stratum.

Aggregation and set comparison are Datalog. Use `proved` / `empirical` / `open`
with explicit domain limits. Hash and retain replayable selection, domain and
equivalence evidence including failures. The replay audit must run model-free and
report `frozen_hashes_unchanged: true` for run B's own manifest.

Before any real collection, test with authored controls that discriminate all
three ranking arms, that make the non-degeneracy gate **fail** when it should,
that exercise rejected candidates, and that inject final-test errors; also test
standalone execution from the emitted program and input facts alone.

**Reporting a second null is a success, not a failure.** Run B is complete when
the outcome — positive or null — is committed and published with its
counterexamples. Do not alter the protocol after final-test answers to change the
verdict.
