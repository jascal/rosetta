# Native traces as a prior for standalone circuit discovery — run B

Status: **open, preregistered before any run-B model observation or any run-B
implementation.** Written after run A completed and after its outcome was known;
the reasons are recorded in [`native-trace-diagnosis.md`](native-trace-diagnosis.md).
Run B is a separate experiment with new inputs. Run A is closed, its frozen
evidence is immutable, and its result — zero incremental certified coverage —
stands unchanged.

## Why run B exists

Run A's trace arm was non-degenerate in principle and degenerate in fact. Its
feature-selection negative class was the 7 positive copy contexts where the
suffix-3 rule errs, because the discovery population contained no negative
controls at all, while every evaluation split contains 18. The decision
threshold was calibrated between two classes drawn only from positive copies, so
on a negative control it extrapolated off-manifold — and labelled all 18/18
train negative controls trace-good. A uniform signature made the trace term
collapse to a firing count, which made the trace and shuffled scores elementwise
identical.

Run B therefore differs from run A in exactly two pre-registered respects. Any
other change to the protocol is out of scope.

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

## Change 2 — a magnitude-bearing, gate-weighted vote

Run A counted agreeing features and discarded magnitude
(`N*2 > D`, `dl/native_trace_select.dl:42`). Run B sums signed margins.

For feature `f` on instance `I`, with selected channel value `V`, threshold `T`
and polarity `P`:

```datalog
% raw margin, in the feature's own channel units
margin(I,F,P*(V-T)) :- trace(I,F,V), feature(F,T,P).

% gate weight: the SwiGLU gate is the real gate on this unit, so a unit whose
% gate is closed should not speak as loudly as one whose gate is open.
gweight(I,F,W) :- trace(I,F,_,G), feature(F,T,P), gate_norm(F,G,W).
gate_norm(F,G,W) :- feature_gate(F,M), W = abs(G) / M, M > 0.0.

% aggregate signed margin; trace-good strictly above zero
signature_sum(I,S) :- signature(I,0), S = sum W : { margin(I,_,_), gweight(I,_,W) }.
trace_good(I) :- signature(I,0), signature_sum(I,S), S > 0.0.
```

`feature_gate(F,M)` is the mean `|gate|` over the **selection** split for that
feature, so the weight is ~1 for a typical instance and suppresses a unit whose
gate is closed. `M <= 0` makes that feature contribute nothing, and that is
reported rather than silently dropped. This mirrors fieldrun's `w_router · act`
(`fieldrun` PR #145): score a gated unit by what actually reaches the output, not
by its raw magnitude.

This is a strict generalisation, not a different rule: `sign(margin)` is run A's
vote, and the aggregate reduces to a majority when all margins have equal
magnitude. A feature at exactly its threshold contributes zero margin.

All of the above is **Datalog**. Python stages facts and serialises. No numeric
or statistical decision moves into Python.

## Non-degeneracy gate — run A's missing precondition

Before the trace term is used to rank anything, assert on the signing split:

- the signature is **not uniform** (both `trace_good` and its complement are
  nonempty), and
- it is **not constant across the negative-control stratum** — specifically,
  `trace_good` does not cover every negative control.

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
split. Trace score `= B + Σ_f (margin_f · gweight_f)` summed over candidate
firings on that same sign split, using the features' thresholds, polarities and
gate means fitted on the disjoint select split. The shuffled arm uses the
identical formula with signature rows permuted by seed 204. Scores and their
relative weight are fixed.

Note the split discipline this implies: the select split fits feature
*parameters* and is never used to rank; the sign split computes the signature and
ranks but never influences a threshold, polarity or gate mean; test is untouched
until after freeze. A trace value is therefore never summed on a row whose
feature parameters were fitted from it.

## The shuffled control is mandatory

Run A's shuffled arm is what made the degeneracy visible instead of leaving a
uniform prior looking like weak-but-real signal. It is retained, unweakened, for
run B. Specifically preregistered:

- If trace and shuffled scores come out elementwise identical, that is the
  degeneracy signature from run A and is reported as such.
- A run-B result where the trace arm beats shuffled is not sufficient on its own;
  the trace arm must also beat **behavior-only** on certified coverage.
- If trace fails to separate shuffled, no reranking, threshold adjustment, or
  feature substitution may be performed to create a separation.

## Intervention diagnostic

Unchanged in spirit and now a genuine contrast: for every selected unit, zero
its post-SwiGLU product at the last position on the first eight **positive** and
eight **negative-control** sign-split rows by ID, plus an opposite-class patch
and an identity patch, with a same-layer control neuron `(n+1) mod width`. Run A
could not form this contrast — its "incorrect" class was 7 positive contexts.
Run B's is a real negative-control population. Still a diagnostic, never an
admission criterion; local causal influence only, never universal necessity or a
unique mechanism. Nulls preserved.

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
and their values, the per-feature gate weights, and the signature distribution on
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