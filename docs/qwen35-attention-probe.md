# Qwen3.5 full-attention probe: instrument only

The source-conditioned two-unit memory approximation failed. The next mechanism
measurement needs to distinguish upstream retrieval from final-MLP readout.
`py/qwen35_attention_reference.py` adds observations and substitutions to the
existing official-graph build-time oracle, without changing the frozen prior
reference/probe implementations.

The official eager full-attention module returns attention weights. Its output
is gated by `sigmoid(gate)` before `o_proj`. The probe records each requested
head's final-query attention row and post-gate vector immediately before that
projection. Scalar patches name layer, position, head, dimension and value.
Observations precede same-layer substitutions; later layers can see earlier
patches. Hooks are removed in `finally`. Invalid linear-attention layers,
head/dimension/position bounds, duplicate sites and nonfinite values are rejected.

The patch width is `num_attention_heads * head_dim`, which need not equal model
hidden size. This matters for actual Qwen3.5-4B (16 heads of width256 feed a
2560-wide residual). Gated DeltaNet layers are explicitly unsupported by this
probe and remain part of the official forward, not approximated or omitted.

**Empirical instrument test:** on a seeded tiny hybrid Qwen3.5 checkpoint,
attention rows match the direct resident official model; full output logits
match; identity substitution is bit-exact; source-position and last-position
substitutions match independently hooked resident-model outputs; a subsequent
unpatched call is bit-exact to baseline. The test passes.

**Open gate before mechanism use:** run the actual4B identity/zero/independent
substitution validation with retained full-score evidence, Datalog payloads and
verdicts. The tiny test is not actual4B parity, source-selection localization,
head sufficiency, standalone conversion or incremental coverage.

The probe is build-time only. Its attention/vector observations must not become
runtime inputs to the final token-only Datalog artifact, and no model fallback
is introduced.

The [actual4B validation](../reference/benchmarks/qwen35_attention_validation_replay.json)
now passes all four replayed queries, including five full-vocabulary comparisons
with zero tolerance/argmax failures. This closes the instrument gate for the
specified sites and smoke context; it does not establish copy mechanism or new
coverage. The [all-head profile](qwen35-attention-profile.md) is the next study.
