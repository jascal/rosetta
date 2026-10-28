# Run C — is the trace signature the copy mechanism? (necessity and sufficiency)

Status: **registered before any run-C model request.** Written after runs A and B were
closed and published (PRs [#62](https://github.com/jascal/rosetta/pull/62),
[#63](https://github.com/jascal/rosetta/pull/63)); run A and run B evidence are immutable
and are not touched by this experiment.

## The question this run exists to answer

Run B certified 31 circuits over nonempty firing domains — the first such domains in the
program — and found that the native-trace prior adds **zero** incremental certified
coverage over either control. It also produced a dissociation:

- the nine selected neurons separate copy contexts from negative controls well
  (mean signature **+3.64** vs **−7.29**), and the signed-margin term demonstrably moves
  the ranking (the trace arm drops `169 179` and promotes `953 961`);
- yet zeroing the selected neuron changes the model's answer **0 times in 144
  attempts**, the adjacent-neuron control is equally inert (**0/144**), and only a
  cross-instance swap moved anything (**1/576**).

Run B's own outcome names the obstacle: *"strong ranking signal, zero causal effect."*
Whatever the signature measures, it is not *shown* to be the copy mechanism. **Run C
tests that directly**, with a treatment fixed in advance and no parameter fitting of any
kind.

Two explanations are consistent with run B, and they are different claims:

- **H1 — off-path.** The signature measures something correlated with the class (a
  nuisance of the copy/control populations, or a class-membership statistic) rather than
  the copy computation. Then removing the coordinates leaves the model untouched, and so
  does overwriting them.
- **H2 — redundancy.** The copy computation is implemented redundantly, so removing
  nine neurons (or two adjacent ones) is underpowered. Then *some* group-level
  intervention must move the model, even if no single-neuron ablation does.

Run B cannot separate them, because run B only ever ablated one or two neurons at a
time.

## Registered hypotheses and decisive readouts

All three are registered before collection. Every readout is a Datalog relation.

| ID | Hypothesis | Decisive readout |
|---|---|---|
| **H-NEC** | The nine coordinates are *necessary* for the copy answer | Some arm flips ≥ 1 copy-row answer |
| **H-SUF** | The signature is *sufficient* to produce the copy answer | A write arm makes a row answer the donor's answer |
| **H-INST** | The coordinates carry *instance-specific* information | Overwriting a copy row with another instance's values changes its answer |

Arms, in the order they are read:

- **necessity** — `Z_k` for k = 1..9: zero the product of the first k coordinates of the
  registered order (§ Treatment). `Z_9` is the maximal treatment.
- **random control** — `R_k` for k = 1..9: zero k *non-target* neurons with the **same
  per-layer counts** as `Z_k` (so `Z_k` and `R_k` are layer-matched), drawn with seed 205.
- **scale-preserving control** — `mean`: replace all nine coordinates with the corpus
  mean of that coordinate over the evaluation split. This removes instance-specific
  information while preserving marginal scale, guarding against "zero is
  out-of-distribution".
- **identity control** — `ident`: write each row's own nine values back. Must reproduce
  the unpatched answer exactly; any deviation invalidates the instrument.
- **write** — `W_cc`: write a copy row's nine values into a control row; `W_cd`: write a
  control row's nine values into a copy row.

Decision rules, fixed now:

- **H2 (redundancy) is supported** iff some `Z_k` with k ≥ 3, or `mean`, or `W_cd`
  flips ≥ 1 copy-row answer.
- **H1 (off-path) is supported** iff *no* arm (`Z_1..Z_9`, `R_1..R_9`, `mean`, `W_cd`)
  flips any copy-row answer, while the run-B signature still separates the classes.
- **H-SUF is supported** iff some write arm makes a row's answer equal the donor's
  answer.
- A write arm that flips a row but *not* to the donor's answer is registered as
  "coordinates are causally relevant, not sufficient" — a third outcome, not a failure.

**A negative write result is not evidence for H1.** A control context contains no copy
source, so injecting a copy instance's signature cannot manufacture a copy answer there;
and a copy context may compute the copy answer redundantly. `W_cd` is the informative
direction; `W_cc` is retained because it is cheap and because a positive result there
would be strong.

## Treatment (fixed in advance, from run B's frozen evidence)

The nine coordinates are taken verbatim from run B's committed
`reference/benchmarks/qwen25_native_trace_b/features.json` (sha256
`fe06adfa53badd035a84c735f9c98af19ca014aee63619037a8231172d6aea57`, inside run B's
frozen manifest `293e4088…`) and are **not** refitted, re-selected, or re-ranked on
run-C data. The driver refuses to build a plan unless that file still hashes to this
digest. Registered order (layer-stratified, so that `Z_3` is one neuron per layer and
the dose-response is not confounded with depth):

| r | layer | channel | neuron |
|---|---|---|---|
| r1 | 7 | gate | 1898 |
| r2 | 15 | gate | 788 |
| r3 | 23 | gate | 412 |
| r4 | 7 | value | 3 |
| r5 | 15 | value | 680 |
| r6 | 23 | value | 1650 |
| r7 | 7 | product | 1835 |
| r8 | 15 | product | 992 |
| r9 | 23 | product | 1792 |

`Z_k = {r1..rk}`. Interventions patch the **post-SwiGLU product** at the last position
(`hidden[position, neuron]`), which is the same instrument run B used, so run C's null or
non-null is directly comparable to run B's 0/144.

## Inputs

Fresh generation, one evaluation split, seeds **301**, lengths 8/12/24, two groups per
length, exposures 2/3, layouts plain/noise8/noise32/near_match8, plus 18 negative
controls — **114 contexts: 96 positive copy, 18 negative controls**. Vocabulary is
disjoint from the six historical Qwen copy datasets, from run A's frozen inputs, and
from run B's inputs. Group identity, input hashes and vocabularies are disjoint from
every prior dataset.

**No fitting happens in run C.** There is no select split, no sign split, no threshold,
no polarity, no candidate ranking and no admission: the treatment is a fixed nine-element
set and the readout is a causal effect. Test/evaluation leakage is therefore structurally
impossible — the only thing learned from data is the effect measurement itself, and the
intervention plan is frozen before the first intervention request.

Donor rule for the write arms: the donor is the row in the **matched group** (same seed,
same generator length, same group index) and the opposite class, preferring the same
context length, taking the lowest id; if no same-length counterpart exists, the lowest-id
counterpart in the matched group is used. The rule is a Datalog relation, not a host-side
choice.

## Model, scope, and oracle gate

Unchanged from runs A and B: Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, f32 fieldrun bundle, greedy decoding.

Run C introduces **no new oracle**: the same validated `0948cbe` executable
(`54de8fc8fd5acec0…67ea62`) and the same f32 bundle (`3a2fb8e5…f77f8d` /
`5c013a3a…aaa12`) as runs A and B. Run A's retained upstream preflight
(`reference/benchmarks/native_trace_preflight/`) is the gate, and the driver refuses to
collect unless the binary, both bundle files, the preflight report and the preflight
reference all hash to what the preflight recorded.

## Readouts

All in Datalog (`dl/native_trace_runc_summary.dl`), over the intervention results and the
unpatched references:

- `flip(i, arm)` — the patched answer differs from the row's **own** unpatched greedy
  answer (the reference), i.e. the intervention moved the model;
- `answer_correct(i, arm)` — for copy rows, the patched answer equals the generated
  copy target;
- `flip_to_donor(i, arm)` — for write arms, the patched answer equals the donor's
  reference answer;
- `logit_delta` — the change in the logit of the row's own reference token, and of the
  generated target;
- `summary(arm, n, n_flip, n_correct, n_flip_to_donor, max_abs_logit_delta)` — per arm.

`flip` is relative to the model's own unpatched answer, not to the generated target: a
model that already gets a copy row wrong cannot be "flipped away from correct" there.
The two are reported separately and never conflated.

## Claim discipline

Everything here is **`empirical`**, over one pinned checkpoint, one pinned executable,
one finite 114-context set, greedy decoding, and interventions at the last position only.
Run C produces **no circuit and no equivalence certificate**, so it claims no `proved`
result; the `proved` results of the program remain run B's 31 finite-domain
certificates.

Not claimed, at any point:

- that the selected neurons are *the* copy mechanism, or that they are not;
- anything about a mechanism beyond local causal influence at the last position;
- anything about other checkpoints, temperatures, positions, or task classes;
- that a null result here generalizes to other discovery bridges.

A null result is a completed, publishable outcome, with its counterexamples retained.

## Frozen boundary and evidence

The inputs, the treatment, the plan and every source file are hashed and frozen **before
the first intervention request**. Every intervention observation, every failed check and
every counterexample is retained. All plan, summary and cost aggregation is Datalog;
Python stages facts, observes the build-time model, and serializes relations.

Stop rules: if the plan query reports invalid inputs, or if the identity control fails to
reproduce the unpatched answers, the run stops before reporting and says so.

## Cost estimate

Per context: 9 `Z` + 9 `R` + 1 `mean` + 1 `ident` + 1 write = 21 requests, over 114
contexts — about **2,400 requests**, roughly 10 minutes of model time at run B's observed
per-request cost.

## Completion

Run C is complete when the outcome — positive, null, or a mixed dissociation — is
committed and published with its counterexamples. No protocol element may be altered
after any intervention observation to change the verdict.
