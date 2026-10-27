# Qwen3.5 head27/10 routing/value/gate factorization

**Open:** what part of the transferred post-gate head vector caused the two
observed answer transfers? The head-output exchange combines routing weights,
contextual value vectors and the query output gate. Its causal effect does not
by itself identify a source-retrieval algorithm.

This study fixes head27/10 and all six old/new directions before requests. Six
no-op captures retain the last-query attention row, raw gate scalars from the
head's query projection, value projections at every context position, and the
post-gate output vector. The grouped key/value index is `head // kv_groups`;
a tiny official-model test checks distinct KV groups and reconstructs their
outputs. Capture hooks must preserve all earlier full-score hashes exactly.

Five arms per direction give30 substitutions: reconstructed target, reconstructed
donor, donor routing with target values/gate, donor values with target
routing/gate, and donor gate with target routing/values. Datalog computes the
stable sigmoid and attention-weighted sums. All source positions are included
in the value factor; this is not yet a source-successor-only value replacement.
The target/donor contexts have aligned lengths and positions.

The policy and source hashes freeze before captures. After capture, Datalog
requires scalar reconstruction within `1e-5+1e-5*abs(original)` and freezes all
numerical payloads and capture bytes before substitutions. All12 reconstructed
full-vocabulary controls run first. They must match the exact target or prior
exact-head-donor scores within `.002+.002*abs(original)` and preserve global
argmax. Failure stops hybrid requests and invalidates interpretation. Applied
FP32 scalar values allow the established `1e-7+1e-7*abs(request)` narrowing error.

Both directions and every failed intervention remain retained. No heads, cases,
thresholds or factors are changed in response to hybrid answers. A transfer
supports a factor's empirical sufficiency only conditional on the other target
model state. It supplies neither a token-only predictor nor a fresh incremental
Datalog certificate. No model activation is an allowed final runtime input.

[Protocol](../reference/benchmarks/qwen35_attention_factor/protocol.json).

```bash
.venv/bin/python py/run_qwen35_attention_factor.py capture
.venv/bin/python py/run_qwen35_attention_factor.py prepare
.venv/bin/python py/run_qwen35_attention_factor.py run
.venv/bin/python py/run_qwen35_attention_factor.py evaluate
python3 py/audit_qwen35_attention_factor.py
```

## Completed causal development outcome

All six captures preserve their prior complete score hashes. Scalar head-vector
reconstruction passes. All12 complete-vocabulary reconstructed controls pass
before the18 hybrid interventions run. The final FP32 patch/argmax summary also
passes.

| substitution | donor-answer transfers /6 directions |
|---|---:|
| reconstructed target | 0 |
| reconstructed donor | 2 |
| donor routing, target values/gate | 2 |
| donor values, target routing/gate | 2 |
| donor output gate, target routing/values | 0 |

Both routing-only and value-only substitutions transfer cases0 and54 to their
long-source answers. Neither transfers case28 or any reverse direction. All
other hybrid outputs remain at their target baseline answer; there are no
additional off-target changes. The donor output gate alone transfers none.

**Empirical:** the donor attention mixture is sufficient for the two rescues
while holding target values and output gating fixed. Contextual value changes
are independently sufficient for the same rescues, holding target routing and
gating fixed. This is evidence for alternative sufficient paths, not a unique
routing-only algorithm or proof that the output gate is irrelevant in general.
The value factor includes every context position, so it must not be described
as a source-local value kernel yet.

The next retrieval factorization should distinguish query-driven from key-driven
routing changes and test whether source-successor keys alone suffice. Only then
can local key/query kernels be justified for a standalone token computation.
All such kernels remain open; no activation, weight file or model may be a
runtime dependency of the final circuit. No fresh incremental equivalence
certificate is claimed by these interventions.

[Summary](../reference/benchmarks/qwen35_attention_factor/summary.json),
[replay report](../reference/benchmarks/qwen35_attention_factor_replay.json).

The [completed Q/K factorization](qwen35-attention-qk.md) reproduces both routing
rescues using only the donor competing-successor key in head27/10's computation,
with target query, other keys, values and gate retained. Query-only transfers
one case. This localizes a sufficient routing operand but does not make its
contextual key representation token-computable.
