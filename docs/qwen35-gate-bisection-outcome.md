# Qwen3.5 gate bisection outcome

**Empirical:** the frozen search completed 26 new forwards across five split
rounds and one terminal query. All requested/applied site checks passed. The
initial three failed reverse roots were retained without further search. The
successful forward roots reduced as follows:

| target context | terminal sufficient set | outcome |
|---|---|---|
| 0 | neuron603 | donor-gate/target-up substitution transfers the long-source answer |
| 28 | seven neurons | neither tested half transfers; stop at the successful parent |
| 54 | neuron1765 | donor-gate/target-up substitution transfers the long-source answer |

Both children were observed before each branch decision. When both succeeded,
the fixed left-first rule was used and the unchosen successful child remained
retained. A failing split does not prove the seven-neuron set irreducible;
nonconsecutive or smaller combinations could still succeed. The singleton
results establish sufficiency for those specific donor/target contrasts, not
necessity or generality across contexts.

This supersedes the failed overlap hypothesis (neurons162/381 were insufficient
individually and jointly). It identifies more precise causal candidates, but no
token-only computation or runtime guard has yet been recovered. The user goal
still requires a frozen standalone Datalog artifact and clean, fresh, disjoint,
nonempty incremental equivalence certificate. These intervention checks must not
be described as that certificate.

The [preregistered search policy](qwen35-gate-bisection.md) remains unchanged;
results are kept in this separate document because the policy is byte-frozen.
Evidence: [terminal outcome](../reference/benchmarks/qwen35_gate_bisection/outcome.json)
and [model-free adaptive replay](../reference/benchmarks/qwen35_gate_bisection_replay.json).
The independent audit checks all six root witnesses, every new argmax archive,
every planner/site query, and exact correspondence of each round's reference
snapshot to earlier observations only.

**Open:** test whether the singleton effects transfer across the other
source-choice contrasts, then derive and test a token-computable predictor of
the relevant gate change. A context-specific donor activation is a build-time
probe, never an allowed runtime input to the final Datalog circuit.

The [completed cross-context test](qwen35-gate-cross.md) finds no extension of
either singleton effect beyond its discovery case, and the joint intervention
rescues only those same two cases. The repeat hashes agree exactly. This rejects
a simple shared-neuron rule; the context qualification is essential.
