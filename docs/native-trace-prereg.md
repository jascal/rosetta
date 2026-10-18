# Native traces as a prior for standalone circuit discovery

Status: **open**, preregistered before instrumentation, new model observations,
or implementation of the experiment. Existing published copy failures motivate
this experiment and are discovery data, never final evaluation evidence.

## Question and success criterion

Does native-neuron information improve certified coverage under a fixed symbolic
search budget? Compare behavior-only, trace-assisted, and shuffled-trace arms.
Success requires positive incremental **finite-domain certified** coverage for
the trace-assisted composite over both controls. A failed composite certificate
contributes zero certified coverage; an empty firing domain is not a proof.
Report raw agreement and errors independently of this deliberately strict metric.
Zero improvement is a completed experiment, not a reason to change the protocol.

## Model, scope, and oracle gate

Use Qwen/Qwen2.5-0.5B-Instruct, revision
`7ae557604adf67be50417f59c2c2f167def9a775`, converted from the cached upstream
checkpoint to an f32 fieldrun bundle. Pin checkpoint, tokenizer, bundle, source,
binary and toolchain hashes in the run manifest. This is one checkpoint and one
precision condition; historical int8 answers are discovery labels only. Re-query
discovery contexts with the pinned f32 oracle. Never pool their old references
with new answers.

Before discovery, validate all gate-preactivation, value and post-SwiGLU product
coordinates at the last token of layers 7, 15 and 23 against upstream Transformers
FP32/eager hooks on a frozen 12-context preflight suite (four natural strings,
four repeated sequences and four distractor sequences). Require matching top-1
on all contexts and per-coordinate `abs(error) <= 0.002 + 0.002*abs(reference)`.
Retain raw arrays, maxima, failures, versions and input IDs. Zero and replacement
interventions on fixed units (7,0), (15,17), (23,31) at the last position must
also match upstream top-1 and selected logits at that tolerance. Compare an
identity patch to an unmodified pass. A gate failure stops model discovery until
diagnosed; any numerical-protocol amendment must precede new discovery data and
be recorded. Tiny fixture and cached/naive agreement do not replace this gate.

Fieldrun changes belong on a separate branch and PR. Work starts from `374d0f6`;
the sibling checkout's uncommitted blind-corpus work is independent and untouched.

## Datasets and information boundaries

Discovery contexts: all 288 positive rows of
`reference/benchmarks/qwen25_05b_copy_generalization/dataset.json` (including its
known failures). They may be traced, intervened on and inspected. The fixed
suffix-3/two-source consensus rule supplies a copy prediction for labeling
discovery rows by agreement with freshly collected model answers.

Generate fresh sequences with `benchmark_copy_generalization.generate`:
lengths 8/12/24, two independent groups per length, exposures 2/3, layouts
plain/noise8/noise32/near_match8, and paired all-source target substitutions.
Seeds 101, 102 and 103 designate train, validation and test respectively.
Use each seed's negative groups in that same split. Exclude every token used in
the historical Qwen copy datasets, then exclude earlier split vocabularies.
Each split has 96 positive and 18 negative contexts. Group identity, input hashes,
and vocabularies must be disjoint between discovery/train/validation/test.
Generate and hash all inputs before collecting new study references. Labels such
as expected target, layout and exposure count never enter runtime rules.

Train and validation references may inform admission. Test references and test
traces are forbidden before artifact/domain freeze; no test traces are needed.
Historical failures and all preflight inputs remain outside final evaluation.

## Fixed candidate vocabulary and equal budgets

All arms receive the same token-derived candidate universe:

- Consensus copy: suffix lengths 1/3/6/9, support 1/2/3 (12 bases).
- Nearest-source copy: suffix lengths 1/3/6/9 (4 bases).
- Longest matching suffix, capped at 12, nearest source breaking ties (1 base).
- Unique-plurality copy: suffix lengths 3/6/9, winning support 2/3 (6 bases).
- Repeat the final token (1 base).

Each base is composed with zero, one, or two distinct guards from this ordered
list: context length <=32, >32, <=64, >64; nearest occurrence of the predicted
token is <=16, <=32, or >32 positions behind the final token; predicted-token
frequency >=2 or >=3; predicted token differs from final token. Guards use only
tokens. Include inconsistent combinations (they simply never fire). There are
24 * (1+10+45) = 1,344 candidates, in the listed base/guard order.

All arms compute training predictions for this same universe, then Datalog ranks
and proposes exactly the top 12 candidates with nonzero training firing (or all
if fewer exist). Ties use candidate ID. No further proposals replace rejected
candidates. All arms thus have the same universe, training observations, and
12-candidate validation/admission budget. Trace collection is additional measured
build-time cost, not a claim of equal wall time or equal information.

Behavior score B = training correct count - 4 * training wrong count.
Trace-assisted score = B + training trace-good firing count - 4 * training
trace-bad firing count. Shuffled-trace uses this same formula with shuffled
signatures. These additive scores and their relative weight are fixed.

## Trace prior and intervention diagnostic

Collect gate-preactivation, value and product for all units at layers 7/15/23
on discovery inputs. For each layer/channel, Datalog selects one neuron maximizing
`(mean_correct - mean_wrong)^2 / (variance_all + 1e-12)`; ties use neuron ID.
Both agreement classes must be nonempty and the score positive; otherwise that
layer/channel contributes no vote. Threshold is the midpoint of the two class
means, with polarity pointing toward the correct class. This yields at most nine
features, selected entirely on discovery. A train row is trace-good when a strict
majority of available features votes correct; with no features the trace term is
zero in both trace arms. Equality at a threshold votes wrong. Preserve signed
gate/value/product observations and sign-regime summaries.

Collect only those features on train inputs. Shuffle whole signature rows using
seed 104, globally within train, preserving their marginal distribution and
inter-feature dependence while breaking correspondence to contexts. Serialize
the permutation as evidence. Datalog applies thresholds, votes, scores, candidate
ranking, admission and routing; Python performs generation, observation I/O,
fact staging and serialization only.

Diagnostic, never an admission criterion: for every selected unique unit, zero
its post-SwiGLU product at the last position on the first eight correct and eight
incorrect discovery cases by ID. Also patch from the opposite-class case paired
by rank, and perform identity patches. Repeat zeroing for a same-layer control
neuron `(n+1) mod width`. Record predictions and logits for the unmodified answer
and copy answer. Datalog summarizes effects, identity consistency, and sign
regimes. This tests local causal influence, not universal necessity or a unique
mechanism. Preserve null results. Missing a class makes this diagnostic explicitly
unavailable rather than inventing examples.

## Admission, runtime and final certification

A proposed candidate is admitted only if it has zero train/validation errors,
fires in at least two independent groups in each split, and correctly follows
paired target substitutions with changed outputs in at least two groups per
split. Missing references or ambiguous outputs invalidate admission. Controls
are included in the zero-error check; they are not automatically labeled wrong
when the model agrees with a candidate. Report their firing separately.

Serialize all admitted candidates and a first-match router in proposal rank
order. The behavior-only composite is the baseline. Each arm is evaluated alone;
there is no inherited priority layer that can hide another arm. Freeze every
individual candidate artifact, composite, runtime wrapper, program/source hash,
selected-feature parameters, selection evidence, all inputs and their Datalog
firing domains before the first test oracle query. A single frozen manifest
records this boundary. Verify it again before certification.

Runtime takes token facts (and an explicit instance domain to represent empty
contexts), emits decisions and explicit abstentions, and uses Souffle alone.
There are no neural observations, model weights, fieldrun, Python, or whole.dl
dependencies. Empty selections emit an explicit abstaining artifact.

Check every admitted individual and each composite through `dl/equiv.dl` on its
frozen firing domain and on the full test domain, including negative controls.
Retain all mismatches and raw verdicts; never remove failures after seeing
references. A clean firing-domain certificate proves only that stated finite
subset. Failure and abstention remain visible; passing individuals cannot rescue
a failed composite by retrospective rerouting.

## Reporting and completion

Report per arm: proposed/admitted counts, firing and abstention, correct/wrong,
positive/negative strata, individual/composite certificates, program bytes,
rule counts, incremental certified-domain sets versus behavior and shuffled
controls, and observation/search cost. Aggregation and set comparisons are Datalog
queries. Use proved/empirical/open labels with explicit scope. Hash and retain
replayable selection, domain and equivalence evidence, including failed checks.

Before the real run, test with authored reference/trace controls that discriminate
the three ranking arms, exercise rejected candidates, and inject final-test errors;
also test standalone execution with only the emitted program and input facts.
Run repository reference checks. Publish the fieldrun instrumentation PR and a
rosetta outcome with reproducible commands and limitations. Never alter frozen
rules after final-test answers, even if all three arms certify zero coverage.
