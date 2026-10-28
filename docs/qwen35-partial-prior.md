# Qwen3.5 two-causal-unit prior diagnostic

**Empirical:** the four-token query-history kernel derived from the two causal
units improves source-preference agreement on the known144-case training sample,
but still fails to supply a zero-error copying rule.

The fixed kernel sums isolated-query products at neurons603/1765 times their
final-readout token coefficients. Datalog computes both the coefficient dot
products and the kernel. No weights, units, history lengths or guard thresholds
are optimized against answers. The existing one/four/full-query probes supply
all observations, so there are zero new model forwards. The readout weights are
build-time evidence, not a claimed standalone runtime artifact.

| history | source disagreements /144 | fixed guard firings | guard mismatches |
|---|---:|---:|---:|
| last1 token | 96 | 48 | 13 |
| last4 tokens | 44 | 80 | 17 |
| full query | 96 | 48 | 13 |

For comparison, the complete-logit four-token prior disagrees on58 cases and
has12 mismatches among56 guarded firings. The two-unit kernel's lower44 source
errors do not establish better certified coverage: its larger guarded domain
contains17 mismatches. These are overlapping, correlated contexts from three
vocabulary groups, not an independent generalization result. Other history
lengths are worse under this fixed kernel.

The guard remains unchanged: the unique prior-preferred source successor must
be the unique longest-source successor with suffix match>=24, competitor>=12,
and match-length margin>=4. Missing data fails; ties abstain. Per-context counts
use the separately tested explicit-instance aggregator, not the earlier
query-group aggregate.

All three Datalog queries replay unchanged without a checkpoint, model, or
Torch. All preferences, firings and counterexamples remain retained. The
predictor is not admitted; no final-test request or equivalence certificate is
promoted from this post-hoc diagnostic. This rejects the fixed isolated-query
two-unit kernel as a zero-error guard, while preserving its limited empirical
source-preference result.

[Protocol](../reference/benchmarks/qwen35_partial_prior/protocol.json),
[kernel](../reference/benchmarks/qwen35_partial_prior/kernel.json),
[counterexamples](../reference/benchmarks/qwen35_partial_prior/summary.json),
[per-context counts](../reference/benchmarks/qwen35_partial_prior/counts.json),
[model-free replay](../reference/benchmarks/qwen35_partial_prior_replay.json).
