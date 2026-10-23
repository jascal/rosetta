# Run C — is the trace signature the copy mechanism?

**Question.** Run B certified 31 circuits over nonempty firing domains and found
that the native-trace prior adds **zero** incremental certified coverage over
either control. It also left a dissociation: the nine selected neurons separate
copy contexts from negative controls well enough to reorder the ranking, yet
zeroing one of them changed the model's answer **0 times in 144 attempts**.

**Answer. The signature is not the copy mechanism — and it is not off the model's
path either.** Run C holds the nine coordinates fixed from run B's frozen
evidence, ablates them as a group and cumulatively, writes other instances'
values into them, and does all of it on fresh inputs. Across **2,394
interventions on 114 fresh contexts, not one copy-row answer changed** — not when
all nine were zeroed at once, not when a control instance's values were written
over a copy row's. Yet those same coordinates move the model's logits about
**twenty times** more than layer-matched random neurons do.

So the honest answer is a dissociation, not a mechanism: the coordinates are
causally connected to the output and carry instance-specific content, and they are
still not where the copy answer is computed.

## Status of the claims

| Claim | Tag | Scope |
|---|---|---|
| No arm flips any copy-row answer | `empirical` | 114 fresh contexts, one pinned checkpoint, 2,394 interventions |
| The identity instrument is exact | `empirical` | 114/114 rows reproduce their answer bit-for-bit; max \|Δlogit\| = 0.0 |
| The treated coordinates move the reference logit far more than layer-matched random neurons | `empirical` | copy rows; mean \|Δlogit\| 0.149 vs 0.008 at k=9 |
| The copy answer is computed outside these nine coordinates | `empirical` | one model, one task, one position |
| That the copy computation is redundant, or where it lives | `open` | not established; see "What this does not show" |

Run C produces **no circuit and no equivalence certificate**, so it claims no
`proved` result. The program's `proved` results remain run B's 31 finite-domain
certificates.

## What was run

Pinned oracle, unchanged from runs A and B: `fieldrun-0948cbe`
(`54de8fc8fd5acec0…67ea62`) against the f32 bundle `model.fieldrun.bin`
(`3a2fb8e5…f77f8d`) for Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The retained run-A preflight still
hashes to those identities exactly, so it is the gate; no new upstream validation
was performed or needed.

Fresh inputs, generated and committed before the first model request: seed 301,
lengths 8/12/24, two groups per length — **114 contexts, 96 positive copy and
18 negative controls**, in 12 groups over 6 matched group keys. The vocabulary
excludes **8,872 tokens**: the six historical Qwen copy datasets, run A's frozen
inputs, and run B's. The model answers **95 of the 96 copy rows correctly**
unpatched.

**The treatment was fixed in advance**: the nine `(layer, neuron)` coordinates of
run B's frozen `features.json`, digest-pinned to `fe06adfa…`. Nothing was fitted,
selected or re-ranked on run-C data. The plan — 2,394 requests over 114 contexts
and 13,338 patches — was computed in Datalog and **frozen and committed before the
first intervention request** (`f6b7030`-style boundary: `frozen.json` hashes the
inputs, the treatment, the plan, the collection and every source).

Arms, all at the last position, all patching the post-SwiGLU product:

| Arm | Requests | What it does |
|---|---|---|
| `zero` k=1..9 | 1,026 | zero the product of the first k registered coordinates (layer-stratified: Z_3 is one neuron per layer) |
| `rand` k=1..9 | 1,026 | zero k non-target neurons with the **same per-layer counts** (seed 205) |
| `mean` | 114 | replace all nine with their corpus mean (scale-preserving removal of instance information) |
| `ident` | 114 | write each row's own values back — the instrument control |
| `write_control_into_copy` | 96 | write a matched-group control instance's nine values into a copy row |
| `write_copy_into_control` | 18 | write a matched-group copy instance's nine values into a control row |

## The instrument is exact

The `ident` arm reproduced every row's unpatched answer, with a maximum
`|Δlogit|` of **exactly 0.0** across all 114 rows. The intervention pipeline is
therefore not perturbing the model by accident, and the zero arms measure what
they claim to measure.

## The result: no copy-row answer ever changed

**Zero copy-row flips, in every arm.** The copy answer stayed correct on 95 of
96 copy rows under all 21 arms that touch a copy row, and the one copy row the
model already got wrong stayed wrong, unchanged, under every arm.

| Arm | Copy rows | Answer flips | Still correct |
|---|---|---|---|
| `zero` k=1..5 | 96 | **0** | 95/96 |
| `zero` k=6..9 | 96 | **0** | 95/96 |
| `rand` k=1..9 | 96 | **0** | 95/96 |
| `mean` | 96 | **0** | 95/96 |
| `write_control_into_copy` | 96 | **0** | 95/96 |
| `write_copy_into_control` | 18 | **0** | — |
| `ident` | 96 | 0 | 95/96 |

Both registered necessity arms fail on the copy rows:

- **Necessity.** Zeroing all nine treated coordinates simultaneously does not
  change the copy answer on any of the 96 copy rows. Nor does any prefix of that
  set (k=1..8). Nor does the corpus-mean substitution, which removes the
  instance-specific content while keeping the marginal scale. Nor does overwriting
  a copy row's nine coordinates with a matched control instance's values.
- **Sufficiency.** No write arm made a row answer its donor's answer
  (`flip_to_donor` is empty for both write arms). A negative here is *not*
  evidence against the signature on its own — a control context contains no copy
  source to be moved — but nothing supports sufficiency either.

### The one flip, and why it is not evidence

Exactly one answer changed anywhere in the experiment: **control row 32**
(12240 → 1015), and it changed under the treated ablation **and** under the
layer-matched random control, to the same token, at the same doses (k≥6 in both).
The identity arm left it alone. A perturbation that moves the model the same way
whether the coordinates are the treated ones or random ones of the same layers is
a generic perturbation effect on a low-margin context, not a fact about the
treated coordinates. The registered rule asks about copy-row answers for exactly
this reason.

### The interesting part: the coordinates do move the model

The copy rows' answers do not change, but the logits do. Mean |Δlogit| on the
row's own reference token, copy rows:

| Arm | Mean \|Δlogit\| | Max |
|---|---|---|
| `ident` | 0.00000 | 0.00000 |
| `rand` k=1..8 | 0.0008 – 0.0078 | ≤ 0.042 |
| `zero` k=1..8 | 0.0011 – 0.0159 | ≤ 0.070 |
| `mean` k=9 | 0.0476 | 0.187 |
| **`zero` k=9** | **0.1486** | **0.430** |
| **`write_control_into_copy`** | **0.1431** | **0.444** |

Two things follow, and they point in opposite directions:

1. **The treated coordinates are causally connected to the output.** At matched
   doses (k=1..8) the treated set moves the reference logit about **twice** as
   much as layer-matched random neurons of the same layers; at k=9 the gap is
   **19×** (0.149 vs 0.008). They are not inert coordinates.
2. **Almost all of that effect is one coordinate.** The dose-response is flat
   through k=8 (mean 0.0011 → 0.0159) and then jumps **9×** at k=9. The only
   difference between `zero` k=8 and `zero` k=9 is the single coordinate
   **(layer 23, neuron 1792)** — the layer-23 product coordinate. Its removal
   moves the copy rows' reference logits far more than the other eight combined.

And still: **not one copy answer changed.** Writing a completely different
instance's values into all nine coordinates of a copy row moves its reference
logit as much as deleting them (0.143 vs 0.149) and does not change its answer.
The copy computation is implemented somewhere else, robustly enough that nine
coordinates carrying genuinely instance-specific information — one of them
powerful enough to move the logits an order of magnitude more than its
neighbours — are not required to produce it.

That is the cleanest statement of the dissociation: **the signature is real, and
it is not the mechanism.**

## The registered verdict, and an implementation defect in it

The preregistration fixed the decision rules before collection:

- H2 (redundancy) iff some `Z_k` with k ≥ 3, or `mean`, or `W_cd` flips ≥ 1
  copy-row answer → **not met** (zero copy-row flips everywhere).
- H1 (off-path) iff *no* arm flips any copy-row answer, while run B's signature
  still separates the classes → **met**. Run B's separation is established in its
  own retained evidence (copy mean **+3.64**, control mean **−7.29**), and run C
  measured zero copy-row flips in all 20 registered decision arms on copy rows (and in
the identity control).
- H-SUF iff some write arm makes a row answer the donor's answer → **not met**.

**The frozen summary query's `h1_supported` relation is nevertheless empty, and
that is a defect in my implementation, not in the data.** The module tested
`!any_flip()` — no flip on *any* row — where the preregistration asks about
*copy-row* answers. The single control-row flip is therefore enough to withhold
the label under the implemented rule but not under the registered one. The same
imbalance makes the module's `relevant_not_sufficient` fire, where the
preregistration reserved that label for a write arm.

Every count the registered rule needs is already in the frozen `summary`
relation: `nflipcopy` is **0 in every one of its 21 arm rows**, covering all 2,394
interventions. The registered verdict
is one step away from the frozen evidence and needs no new query.

**I did not patch the frozen module.** It is hashed in `frozen.json`, and the
archive replays byte-for-byte only while it stays unchanged; the frozen sources
are recoverable at the freeze commit. A defect fix belongs to a future study,
where it can be registered before collection instead of after.

## What this does not show

- **Not** that the copy mechanism is redundant. Run C shows these nine coordinates
  are not necessary for the copy answer on this model. Redundancy elsewhere —
  other neurons, other layers, other mechanisms — remains possible and untested.
- **Not** where the copy computation lives. Run C ruled nine coordinates out; it
  did not point at anything. A layer-sweep ablation of whole MLPs, or of the
  attention path, is the natural next probe and is **not** part of this run.
- **Not** anything beyond one checkpoint, one task family, one position (the
  answer position), greedy decoding, and the last-position product channel. The
  copy source lives at an earlier position; an intervention there is a different
  experiment.
- **Not** a claim about native-neuron interpretability in general. It is a claim
  about nine specific coordinates, chosen by one particular discovery pipeline,
  on one model.

## What this means for rosetta

Run A said: the native-trace prior, as first implemented, bought nothing.
Run B said: even a fair version of it buys nothing — the trace arm certifies
nothing the controls do not already certify, and is strictly worse in raw
coverage. **Run C says why the bridge did not close, and it is not a calibration
problem.** The signature genuinely discriminates the classes and genuinely
influences the model's internals; it is simply not the computation the model
performs to copy. Threshold statistics from this bridge cannot substitute for
that, because the bridge points at the wrong coordinates.

For the minimization loop itself the news is better than in runs A and B: run B
already produced the program's first nonempty certified domains (31 circuits,
86–102 instances, zero wrong answers), and run C did not disturb them — it
worked on a fixed treatment and never touched a circuit.

The practical consequence is that a *better bridge* has to be built from
interventions that are **necessity-and-sufficiency** tests on the candidate
mechanism, not from ranking statistics over a fixed predicate library. Run C is
a working instrument for that: a treatment fixed in advance, a frozen plan, an
exact identity control, layer-matched random controls, and a Datalog verdict.

## Evidence and reproduction

Study directory `reference/benchmarks/qwen25_native_trace_c/`.

```bash
.venv/bin/python py/run_native_trace3.py evaluate reference/benchmarks/qwen25_native_trace_c
```

Re-runs the summary query against the retained intervention journal with the
model absent, and rewrites `summary.json` and `report.json` (neither is in the
frozen manifest; the frozen files are untouched). The plan query replays on its
own:

```bash
.venv/bin/python py/certificate.py reference/benchmarks/qwen25_native_trace_c/certificate-evidence/plan/check-<id>
```

Both retained queries replay as `certified: true`: the plan query re-derives all
2,394 requests and 13,338 patches from the frozen inputs and the fixed
treatment, and the summary query re-derives every count, flip and verdict
relation from the retained intervention journal.

Cost, excluding server startup and bundle load:

| Phase | Requests | Seconds |
|---|---|---|
| collection (unpatched) | 114 | 39.2 |
| interventions | 2,394 | 801.4 |

The preregistration, the inputs, the freeze and the plan are committed in order
on `research/native-trace-run-c`; the freeze commit precedes the intervention
journal, and the journal is not in the frozen manifest.

## Relation to runs A and B

Run A remains valid on its own terms; this document does not retract it. Run B
remains the program's `proved` result. Run C is the mechanism probe that
neither of them was: with the treatment fixed from run B and the inputs fresh,
it separates "the signature is off-path" from "the signature is on-path but
redundant" — and the answer is that the coordinates are on-path and still not
the mechanism. The upstream instrumentation is
[fieldrun PR #144](https://github.com/jascal/fieldrun/pull/144); the original
sibling checkout was never modified.
