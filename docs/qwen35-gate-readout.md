# Qwen3.5 causal-unit pairwise readout analysis

**Empirical decomposition:** all 12 singleton gate substitutions move the
unnormalized pairwise margin toward the donor's answer under the stored-weight
linear readout calculation. Only two substitutions change the global answer,
consistent with their retained full-vocabulary references. The cross-context
nulls therefore must not be interpreted as showing no donor-directed signal;
the signal can be too small to overcome the remaining readout margin.

For final-MLP neuronN and vocabulary tokenT, Datalog computes
`sum_h down(h,N) * (1+norm_weight(h)) * head(T,h)`. Qwen3.5's final RMSNorm uses
`1+weight`, verified against the installed official implementation. The RMS
inverse scale is positive and common across vocabulary tokens within one state,
so it does not affect the sign of that state's pairwise margin. The contribution
change is `(patched_product-original_product) * (coefficient(wanted)-coefficient(old))`.

Only source successors and actual observed answers are selected, by Datalog,
for coefficient readout. There are20 neuron/token coefficients over the two
neurons and ten tokens. Source weights are read and staged at build time; all
dot products, margin changes, classifications, and consistency checks are
Datalog. No new model forward was requested.

The calculation uses double arithmetic on widened stored values. It is not a
bit-exact reproduction of the FP32 down projection, residual addition, final
normalization, or full-vocabulary logits. The check requires complete dimensions
and verifies that an observed transfer has positive predicted pairwise change;
it does not certify a replacement function or infer a global answer from a
positive contribution. All12 predicted contributions are positive, with no
sign conflict. The retained singleton argmax witnesses still establish only two
actual transfers.

**Open:** predict the relevant gate-product contribution from token facts,
without whole-model activations at runtime. The existing isolated-query probes
already contain these two unit observations, so a causal partial-readout
predictor can be tested on known training data without new oracle calls. That
would still require a frozen standalone rule/guard and fresh disjoint final
confirmation before any new coverage is certified.

The weight facts in this analysis are build-time evidence, explicitly not the
user's requested runtime artifact. No claim of token-only execution, positive
incremental coverage, or whole-model equivalence follows from this decomposition.

[Readout relations](../reference/benchmarks/qwen35_gate_readout/readout.json),
[model-free replay](../reference/benchmarks/qwen35_gate_readout_replay.json).

The [completed isolated two-unit predictor](qwen35-partial-prior.md) reduces
four-token source-preference errors to44/144 versus58 for the full-logit prior,
but its fixed guarded domain has17/80 mismatches. This does not provide an
admissible circuit. All predictor counterexamples and three replayable queries
are retained, with no new model forwards.
