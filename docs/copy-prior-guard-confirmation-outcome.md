# Fixed prior-guard confirmation: failed

**Empirical:** the unchanged 32-token/two-source candidate failed its independent
confirmation. All 612 planned references were collected. There are 13 mismatches
among 240 firing cases, including 13 among the 96 cases beyond the frozen
behavior-only baseline. No rows, candidates or guards were changed after answers.

| seed | contexts | firing | incremental | mismatches | successful incremental causal groups |
|---|---:|---:|---:|---:|---:|
| 921 | 306 | 120 | 48 | 9 | 3 |
| 922 | 306 | 120 | 48 | 4 | 3 |

`dl/equiv.dl` fails on both the complete frozen firing domain (13/240 mismatches)
and the complete frozen incremental domain (13/96). Both have zero uncovered,
missing-reference and invalid obligations. The full 612-case check retains 372
uncovered obligations and the same 13 mismatches. The baseline answers 288 cases
with zero measured errors; 144 are outside this standalone candidate's domain.
The candidate is not a baseline coverage superset.

The failures expose a specific remaining behavioral gap: the model can choose
the shorter competing source even when two or three longer sources agree and
the frozen one-token lexical prior favors their successors. In case 242 the
rule predicts token 34294 (`"-`) while the model emits the decoy 35679
(`Internet`). Editing every long-source successor to token 36496 (` servlet`)
in case 243 leaves the model's answer at `Internet`. Cases 600/601 similarly
predict `sy`/`SUCCESS`, while the model emits the decoy ` attempt`.
These examples are retained counterexamples, not instructions for a new guard.

**Proved over the finite runtime comparison domain:** relocated execution in
Souffle alone reproduces every frozen decision and abstention on all 612
contexts. The artifact is 92,713 bytes, including 3,240 categorical lexical
facts, with no model, weights or Python dependency at runtime. All eight retained
queries replay with identical relations and verdicts, including the failed
equivalence queries. Executability and reproducibility do not establish model
equivalence.

The clean 54-case development certificate remains valid over its stated domain;
it did not generalize without error to this independently frozen population.
This population was explicitly conditioned on the fixed lexical guard and is
not a natural-language coverage estimate. The intrinsic alphabet was known to
the build-time feature extraction; full contexts and context vocabularies were
fresh and disjoint between seeds and from earlier full-context studies.

The goal of fresh, positive incremental **certified** coverage remains **open**.
The trial is closed as a failure; option 6 is not substituted and no successful
subset is promoted to confirmation.

Evidence: [frozen protocol](../reference/benchmarks/qwen25_copy_prior_guard_trial/protocol.json),
[Datalog summary](../reference/benchmarks/qwen25_copy_prior_guard_trial/summary.json),
[equivalence verdicts](../reference/benchmarks/qwen25_copy_prior_guard_trial/certificates.json),
[standalone runtime check](../reference/benchmarks/qwen25_copy_prior_guard_trial/runtime.json),
[model-free replay](../reference/benchmarks/qwen25_copy_prior_guard_trial/replay-audit.json).
