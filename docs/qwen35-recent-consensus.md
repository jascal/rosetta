# Qwen3.5 recency/long-consensus candidate: rejected

**Empirical:** a single token-only computation was tested on the completed
300-context Qwen3.5 development inventory, whose answers were already known.
It combines the existing long-consensus detector with nearest matching-source
agreement: minimum suffix match32, at least two agreeing source occurrences,
a different successor matching at least12, a match-length advantage of at
least4, and agreement with the nearest source matching at least3.

The suffix detector can match beyond the literal final query into preceding
repeated material. Thus minimum match32 does not imply a32-token query or a
length64 source in this inventory; it describes the actual whole-context suffix
match. The candidate has no model, activation, lexical-prior or neuron-ID input.

| development split | token-only firings | model mismatches |
|---|---:|---:|
| train | 24 | 0 |
| validation | 24 | 10 |

`dl/equiv.dl` rejects the complete48-case firing domain: `nmiss=10`, `nuncov=0`,
with no missing-reference or invalid obligations. The token-only domain query,
failed equivalence query, and split-count query all replay unchanged. The
unmodified failed artifact and every mismatch are retained. No successful subset
is promoted, no final reference is requested for this rejected candidate, and no
claim of a fresh certificate is made.

The failure is concentrated in the validation length64 vocabulary group, despite
long-source support and nearest-source agreement. For example, context268
predicts119902 while the model returns46511; the paired all-source edit269
predicts123713 while the model still returns46511. Recency and repeated matching
alone do not explain the model's preference for the competing successor.

This is another failed extraction hypothesis, not the user's requested result.
A token-computable source-choice rule remains missing. The earlier MLP
interventions establish context-specific causal sensitivity, but do not justify
promoting this structural approximation to model equivalence.

[Artifact](../reference/benchmarks/qwen35_recent_consensus_development/circuits.dl),
[failed certificate and counterexamples](../reference/benchmarks/qwen35_recent_consensus_development/certificate.json),
[split counts](../reference/benchmarks/qwen35_recent_consensus_development/counts.json).
