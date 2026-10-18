# Run B: native traces as a discovery prior, Qwen2.5-0.5B-Instruct

**Question.** Do native-neuron activation traces improve discovery of certified,
standalone Datalog circuits, versus behavior-only and shuffled-trace controls
under equal proposal budgets?

**Answer. No.** Run B fixes the degeneracy that voided run A, and the traces
still add **zero** incremental certified coverage over either control. What
changes is that the null is now interpretable: the circuits certify, the
selection is non-degenerate, and the trace arm simply does not beat the
controls.

Run A's null was cheap to dismiss — the trace label was uniformly good on every
training instance, so the trace and shuffled arms were the same object and the
gate refused to rank. Run B removes that artifact and still finds no benefit.
That is the substantive finding.

## Status of the claims

| Claim | Tag | Scope |
|---|---|---|
| Each of 31 circuits is equivalent to the model on its firing domain | `proved` | Per-circuit finite domain, 86–102 instances, `nmiss=0 ∧ nuncov=0` |
| No circuit produces a wrong answer on the full test split | `proved` | 114 test instances, `nmiss=0` everywhere |
| Trace adds no certified coverage over behavior or shuffled | `proved` | `incremental("behavior",0)`, `incremental("shuffled",0)` |
| Admissions, coverage, signature separation, intervention flips | `empirical` | 114+114+114 contexts on one pinned checkpoint |
| That the selected neurons *are* the copy mechanism | `open` | Contradicted locally; see diagnostic |

No unconditional claim about this model or checkpoint is made anywhere below.

## What was run

Pinned oracle, unchanged from run A: `fieldrun-0948cbe`
(`sha256 54de8fc8…`) against the f32 bundle `model.fieldrun.bin`
(`sha256 3a2fb8e5…`) for Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Both digests re-verified against the
retained run-A preflight before collection, so no new upstream validation was
required or performed.

Three disjoint splits of 114 contexts each — 96 positive copy, 18 negative
control — from seeds 201/202/203, with excluded token 6418 kept out of every
historical dataset. Dataset digest `80bce2e3…`, hypotheses digest `d1365940…`.
The select split fits feature parameters only; sign ranks; test is untouched
until after the freeze.

Nine features, one neuron per layer 7/15/23 × gate/value/product, each with its
own fitted threshold and polarity.

## The gate now passes

Run A's failure was `degenerate_uniform()`: the majority trace label was good on
every training instance. Run B relabels features by copy-versus-control class,
and the signature separates cleanly:

| Stratum | Rows | trace_good | Signature range | Mean |
|---|---|---|---|---|
| Copy (positive) | 96 | 69 | [−5.91, 16.62] | **+3.64** |
| Control (negative) | 18 | 2 | [−15.52, 7.89] | **−7.29** |

`degenerate`, `degenerate_uniform`, and `degenerate_all_negatives` are all empty;
`non_degenerate()` holds. The signed-margin sum `S(I) = Σ_f P_f·(V(I,f) − T_f)`
implemented in `dl/native_trace_select2.dl` does what the repair specified:
negative margins keep their magnitude, zero contributes zero, and the gate
channel is retained descriptively without ever reweighting a margin.

## Ranking under a fixed budget

Twelve proposals per arm, no replacements:

| Arm | Proposed | Admitted |
|---|---|---|
| 0 — behavior-only | 12 | **12** |
| 1 — trace | 12 | **12** |
| 2 — shuffled-trace control | 12 | **4** |

All eight rejections fall in the shuffled arm, each carrying real errors
(2 select / 5 sign, or 1 / 2). The trace arm shares ten of twelve with
behavior-only, dropping `169 179` and promoting `953 961` — so the signed-margin
term demonstrably moves the ranking, it just does not move it somewhere useful.
The shuffled control's four survivors are `169 171 179 196`, and `169 179` are
precisely the two the trace term demoted.

## The diagnostic contradicts the mechanism reading

576 registered sign-split interventions, plan and observations agreeing exactly
(144 each):

| Class | Interventions | Flips |
|---|---|---|
| identity | 144 | 0 |
| target zero | 144 | **0** |
| adjacent-neuron control zero | 144 | 0 |
| opposite-class swap | 144 | 1 |

Zeroing the selected neuron changes the model's answer **zero times in 144
attempts**. Identity is exact, and the adjacent-neuron control behaves like the
target — both are inert. Only the swap class moved anything, once, and a swap
substitutes a different instance rather than ablating cleanly.

So the selected neurons discriminate the classes well enough to reorder a
ranking, yet ablating them does not move the model. Whatever the signature
measures, it is not shown to be the copy mechanism. This is a dissociation, and
it is the strongest reason not to read the trace arm's ranking advantage as
mechanistic.

## Final result on the held-out test split

The freeze was committed and pushed as `f6b7030` **before** the first test
observation; `frozen.json` records that no test-oracle reference was requested
when the manifest was written. All 31 recorded artifact digests re-verified
against their files.

**All 31 circuits are proved equivalent over their firing domains** —
`nmiss=0 ∧ nuncov=0` on 86–102 instances each. Zero wrong answers on any of the
114 test instances, for any arm. Coverage profiles:

| Answered / total | Artifacts |
|---|---|
| 102 / 114 | 6 |
| 96 / 114 | 23 |
| 86 / 114 | 2 |

`counterexample`, `wrong`, `extra`, and `improved` are all empty.

And yet:

```
incremental("behavior", 0)
incremental("shuffled", 0)
improved() = {}
```

The trace arm certifies **nothing the controls do not already certify**, and it
is strictly *worse* in raw coverage — 96 answered against 102 for both controls,
because it abstains on all 18 negative controls where the others answer 6.

Full-domain checks deliberately fail: `ndomain=114, nmiss=0, nuncov=12` (or 18).
That is abstention, not error, and it is the honest boundary of the result.

## What this means for rosetta

Run A established that the native-trace prior, as first implemented, bought
nothing — but its gate tripped, so the experiment could not have found anything.
Run B is the fair version, and it still buys nothing.

The methodological gain is real and separate: this is the first nonempty
model-equivalence domain in the program. Run A produced `ndomain=0` everywhere
and three abstaining artifacts; run B certifies 31 circuits over domains of
86–102 instances with no wrong answers. Whatever the traces contribute, the
minimization loop itself now demonstrably certifies on this model.

Two things follow. The residual must keep abstaining rather than fall back to
the model — runtime remains souffle-only, and the 12–18 uncovered instances are
handled by that Datalog abstention, exactly as the runtime-independence
invariant requires. And there is no evidence here for widening the native-trace
prior; if anyone revisits it, the diagnostic result — strong ranking signal,
zero causal effect — is the obstacle to explain, not the ranking.

## Evidence and reproduction

Study directory `reference/benchmarks/qwen25_native_trace_b/`.

```bash
.venv/bin/python py/audit_native_trace2.py reference/benchmarks/qwen25_native_trace_b
```

Replays all 108 retained queries with the model absent: **108 checks, 108
verdicts preserved, frozen hashes unchanged, no missing evidence, 3 standalone
arms executing from `circuits.dl` and token facts alone, runtime OK**. The
standalone executions compare both decisions *and* abstentions against the
frozen predictions.

`python3 py/verify_threx.py` remains **CERTIFIED 25/25**.

Observation cost, excluding startup, conversion, and upstream validation:

| Phase | Requests | Seconds |
|---|---|---|
| select | 114 | 30.83 |
| sign | 114 | 31.83 |
| diagnostic | 576 | 134.31 |
| test | 114 | 33.50 |

The three sealed input files were byte-verified unchanged after the run. No
preregistered threshold was altered, no rule was substituted, and no candidate
was reranked after seeing any test observation.

## Relation to run A and to fieldrun

Run A remains valid as a recorded negative result on its own terms; this
document does not retract it. Run B supersedes it only as the fair test of the
hypothesis. The upstream instrumentation is
[fieldrun PR #144](https://github.com/jascal/fieldrun/pull/144); the original
sibling checkout was never modified.