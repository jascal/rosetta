# Native traces as a discovery prior: Qwen2.5 experiment

**Completed; empirical result: zero incremental certified coverage.** All three
arms admitted zero candidates and abstain on all 114 final inputs. This is the
negative outcome allowed by the unchanged [preregistration](native-trace-prereg.md).
The evidence does not support a native-trace advantage for this fixed search.
Freeze commit: `a59f553`, created before the first final oracle request.

## Motivation and relation to fieldrun

Goodfire's [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/)
argues for studying native neurons and their connections directly, including signed
SwiGLU regimes and interventions. It presents a working hypothesis; interpretable
neurons do not by themselves establish exact behavioral replacement. For rosetta,
the useful implication is a testable search prior: can native observations help
find token-derived rules that survive independent equivalence checks?

Fieldrun's recent [native activation probes](https://github.com/jascal/fieldrun/pull/139)
and [blind-max search](https://github.com/jascal/fieldrun/pull/140) supply related
build-time investigation tools. Blind-max searches from a unit's output-projection
vocabulary and compares against corpus activations. Its ratio depends on the search
and corpus; it is not an equivalence certificate. Subsequent changes record the
[search configuration](https://github.com/jascal/fieldrun/pull/142) and distinguish
[unsupported scorers from deliberately declined routed layers](https://github.com/jascal/fieldrun/pull/143).
Our [instrumentation PR](https://github.com/jascal/fieldrun/pull/144) adds gate,
value, and product observations plus position-specific product replacements to
dense RoPE models. The experiment is pinned to its validated `0948cbe` executable;
the PR also incorporates current upstream work at merge `4c61d9d`.

**Open:** native mechanisms may suggest useful symbolic predicates. The claim to
test is improved independently certified coverage, not neuron interpretability
alone. The experiment uses traces to rank a fixed predicate library; it does not
invent unrestricted new predicate definitions or reconstruct an entire mechanism.
No native measurements enter an emitted runtime artifact.

## Protocol and oracle

Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, f32 fieldrun bundle, greedy next-token
answers. Historical copy contexts are discovery inputs only; all 288 were queried
afresh. Seeds 101/102/103 supply train/validation/test, 114 inputs each, with
disjoint groups and token vocabularies. Each split contains 96 positive and 18
negative-control inputs. Natural-language preflight contexts are separate.

**Empirical:** the upstream FP32/eager validation passes on 12 contexts and 108
interventions: all top-1 answers match, 525,312 gate/value/product coordinates and
selected logits meet `abs(error) <= .002 + .002*abs(reference)`. Maximum native
absolute error is 0.000236786. This supports this finite validation suite; it is
not a proof that all fieldrun answers equal upstream. The final equivalence
queries are against the pinned fieldrun references.

Model/checkpoint hashes, dependency versions, comparison details, and raw arrays
are retained in [native_trace_preflight](../reference/benchmarks/native_trace_preflight/).
The [method notes](../reference/benchmarks/qwen25_native_trace/METHOD_NOTES.md)
record the uncaptured converter executable digest and pre-freeze implementation
changes. The resulting bundle and actual probe executable are hashed and validated.

## Development result

**Empirical:** the strict suffix-3/two-source copy rule agrees with 281/288 freshly
queried discovery answers; seven disagree. Datalog selected one neuron for each
layer 7/15/23 and channel gate/value/product, nine total. On the fresh training
set, however, their majority vote labels all 114 inputs trace-good. Shuffling
whole trace rows therefore changes no majority labels. Trace and shuffled ranking
are identical; their score is behavior score plus firing count in this run.
This is an observed failure of this prior to discriminate the training inputs.

Each arm receives the same 1,344 candidates and 12 proposal slots. Every proposed
candidate has at least one train error and at least one validation error, so none
passes admission. All proposal scores, group/paired-substitution support, and
counterexamples are retained in the Datalog selection evidence. The additional
trace term changes the proposals relative to behavior-only but admits no rule.
No rejected rule was replaced or relaxed.

## Final outcome and certificate scope

| Arm | Proposed | Admitted | Final decisions | Correct / wrong | Abstain | Certified coverage |
|---|---:|---:|---:|---:|---:|---:|
| Behavior | 12 | 0 | 0/114 | 0 / 0 | 114/114 | 0/114 |
| Trace | 12 | 0 | 0/114 | 0 / 0 | 114/114 | 0/114 |
| Shuffled trace | 12 | 0 | 0/114 | 0 / 0 | 114/114 | 0/114 |

Both incremental certified-domain sets are empty. All arms abstain on the 96
positive and 18 negative-control final inputs. No final mismatch counterexample
exists because no artifact answers; this must not be reported as perfect
accuracy. Development counterexamples are in `selection.json`'s `failure_detail` and the
retained `certificate-evidence/selection/*/outputs/failure_detail.csv`. For example,
candidate 171 predicts token 4833 instead of model token 51185 on training
instance 102, and 21287 instead of 58263 on validation instance 171.

Each composite has a 3,815-byte `circuits.dl` with 34 shared library rule lines
and no selected candidate facts, plus its runtime wrapper. These are abstaining
artifacts; the library size is not evidence of recovered or compressed behavior.
There are no admitted individual artifacts to certify.

**Proved:** no nonempty model-equivalence domain was established. All six
`dl/equiv.dl` checks fail: the three firing domains are empty (`ndomain=0`), and
each full domain has `ndomain=114`, `nmiss=0`, `nuncov=114`. There are no missing or
invalid references. The empty-domain rejection prevents vacuous certification.
The evidence preserves all six failures. This does not claim that copy behavior
is impossible to capture or that native neurons are uninformative in general.

## Intervention diagnostic

**Empirical**, restricted to the selected neurons and discovery contexts:

| Intervention | Requests | Greedy prediction flips |
|---|---:|---:|
| Target-neuron zero | 135 | 1 |
| Adjacent-neuron zero control | 135 | 0 |
| Identity | 135 | 0 |
| Opposite-class swap | 126 | 1 |

The zero flip is discovery instance 10, layer 7, neuron 604; the swap flip is
instance 10, layer 23, neuron 3070. All identity patches preserve the checked
logits exactly. These sparse effects establish neither necessity nor a unique
copy mechanism. The logit changes, including cases without flips, are retained
in `diagnostic-summary.json` and its replayable query. Signed regime counts cover
all 288 discovery cases for each selected unit; e.g. layer 7/neuron 604 has
50 `++`, 57 `+-`, 149 `-+`, and 32 `--` gate/value signs. A positive product alone
would conceal which same-sign regime produced it.

## Cost and checks

Datalog aggregates observed request durations (seconds spent in each local API
request; excludes startup, conversion, upstream validation, and symbolic queries):

| Stage | Model requests | Request seconds |
|---|---:|---:|
| Discovery | 288 | 87.125 |
| Train + validation | 228 | 65.448 |
| Diagnostic | 531 | 149.932 |
| Final | 114 | 33.238 |

The fixed search evaluates 1,344 candidates per arm and proposes 12 each. Trace
feature selection runs nine queries over 4,864 neurons × 288 observations each.
The original combined selection query took 0.743 seconds; a reporting-only rerun
took 0.908 seconds and preserved every original decision relation. Separate arm
query durations and original feature-query durations were not recorded. Trace
collection is additional cost, so equal proposal budgets do not imply equal
wall time or equal information. Raw traces and query facts are retained in full.

- Rosetta: 181 tests pass, including 11 native-trace tests with discriminating
  synthetic controls, final-error injection, freeze tampering, and independent runtime checks.
- The threx reference remains `CERTIFIED 25/25`.
- Fieldrun merged head: 14 library + 144 binary tests pass; PR #144's full CI
  build, test, and faithfulness matrix passes.
- All retained study queries replay with identical output relations and unchanged
  verdicts, including failed certificates. Three isolated runtime checks use only
  emitted Datalog and token/domain facts. See `replay-audit.json`.

## Implications for rosetta

**Empirical:** this particular discovery-class threshold prior did not transfer
as a discriminating signal to training inputs, and it yielded no certified
runtime behavior. A seven-case discovery error class, a small synthetic copy
domain, one checkpoint, and a fixed predicate library limit the conclusion.
The shuffled control is uninformative about context-specific trace alignment
when every majority label is the same; it still exposes that degeneracy clearly.

**Open:** the unresolved bridge is turning a mechanistic explanation into an
independently certified, token-only computation. Goodfire's argument motivates
investigating native mechanisms directly; that investigation is not the same thing
as a circuit that survives `dl/equiv.dl`. This run measures one bridge — a
discovery-class threshold prior ranking a fixed predicate library — and that
bridge did not close. **The promising direction is a better bridge, not more
threshold statistics from the same one.** What this experiment does establish for
that work is the instrument and the test: native observations are validated
against upstream, the whole selection/admission/certification path is Datalog and
replayable without a model, and a null result is reported with its counterexamples
rather than replaced by a positive one.

A separately preregistered follow-up could test richer signed/conditional
mechanisms, different layers or positions, or token predicates suggested by
intervention contrasts. It must use new evaluation inputs and stronger trace
controls. Those are future hypotheses, not retroactive repairs to this experiment.
Native probes and fieldrun's blind-max work are useful build-time instruments;
rosetta should retain its independent Datalog admission/certification boundary
and Souffle-only runtime. Neither a high activation nor a plausible neuron
explanation licenses a runtime model fallback or an unconditional proof claim.

## Evidence and reproduction

The study is retained in
[reference/benchmarks/qwen25_native_trace](../reference/benchmarks/qwen25_native_trace/).
`protocol.json` pins initial inputs; `frozen.json` pins final sources, observations,
selection, emitted artifacts, and firing domains before final model requests.
`report.json` records finite-domain verdicts and Datalog summaries. The directory
includes failed checks as well as successful queries. A `certified` flag on a
selection/report query certifies that query's integrity conditions, not model
replacement; only the `equiv-*` checks address replacement.

Replay an individual retained query, without the model:

```bash
python3 py/certificate.py reference/benchmarks/qwen25_native_trace/certificate-evidence/<stage>/<check-directory>
```

A failed equivalence check intentionally exits 1. Its evidence remains intact.
Every evidence directory contains its own Datalog program, facts, outputs,
Souffle version, and SHA-256 manifest.

To recollect in a new directory, copy the committed `dataset.json`,
`discovery.json`, `hypotheses.dl`, and `protocol.json` there; use these exact inputs
rather than generating against a changing set of historical datasets. Obtain the
pinned checkpoint, convert to f32, build the recorded fieldrun revision, and run
`scripts/validate_mlp_probe.py` as documented in fieldrun's
[`docs/NATIVE_MLP_PROBES.md`](https://github.com/jascal/fieldrun/blob/feat/native-mlp-traces/docs/NATIVE_MLP_PROBES.md).
Use a fresh `--preflight` directory for recollection: the driver verifies that its
binary and bundle match that gate's hashes. A rebuilt binary need not have the
original executable hash; recollection is a separate run with its own provenance.

Then run these phases in order, supplying `--binary`, `--bundle`, and `--preflight`
to model phases:

```text
discovery → features → development → select → diagnostic → freeze → final → evaluate
```

Each phase is `python3 py/run_native_trace.py <phase> <new-study-directory>`.
Commit the frozen manifest before `final`. Python performs observation I/O,
fact staging, and artifact serialization; Datalog computes features, proposals,
admission, routing, effects, and certificate/coverage summaries.

To run an emitted artifact independently, copy just its `circuits.dl` and
`run.dl` into a fresh directory. Supply `facts/tok.facts` as tab-separated
`instance,position,token` rows and `facts/domain.facts` as one instance ID per
line, then run:

```bash
mkdir outputs
souffle run.dl -I . -F facts -D outputs
```

`cdecide.csv` contains decisions; `abstain.csv` contains explicitly unanswered
instances. No Python, fieldrun, weights, traces, or `whole.dl` are needed.
