# Qwen3.5 source-directed head donor exchange

**Open:** do the consistently shifting full-attention heads causally transfer the
source-choice change? The preceding all-head profile was observational. This
study retains the same six known development contexts and both directions.

Datalog selects the heads with positive shifts in all three source-choice pairs
among the four frozen profile leaders: layer27 heads10 and12. A seed2300 random
head order selects the same number of controls in the same layer, excluding only
the selected heads. The frozen controls are27/1 and27/14. Control27/1 was also a
profile leader; it is not excluded or replaced on that basis. Controls measure
this fixed random comparison, not a guaranteed irrelevant/null head population.

For every directed contrast, exchange each selected head separately, both
selected heads jointly, each random head separately, and both random heads
jointly. Also run a joint selected-head identity control. These seven arms per
context give42 forwards. Every scalar in the256-dimensional post-gate head vector
at the last query position is substituted immediately before output projection.
All other positions and heads retain the target's own computation.

The full payloads, cases, controls, prior links and source hashes are frozen
before model requests. Datalog computes global argmaxes from all vocabulary
scores, verifies every requested/applied scalar and site count, and requires
all six identity controls to match the prior full score hashes exactly with
zero nonidentity scalar changes. Both heads and every failed substitution stay
retained; no answer-dependent head replacement is permitted.

A donor-answer transfer supports causal sufficiency only in that context,
conditional on the remainder of the target model. Even all forward transfers
would not establish a token-only retrieval algorithm, necessity, a shared
symbolic rule, or fresh incremental coverage. The final runtime still must use
only token facts and Datalog with explicit abstention and no model fallback.

[Protocol](../reference/benchmarks/qwen35_attention_exchange/protocol.json),
[payloads](../reference/benchmarks/qwen35_attention_exchange/payload.json).

```bash
.venv/bin/python py/run_qwen35_attention_exchange.py run
.venv/bin/python py/run_qwen35_attention_exchange.py evaluate
python3 py/audit_qwen35_attention_exchange.py
```

## Completed causal development outcome

All42 forwards complete. The summary passes, all six identity score hashes are
exact with zero changed scalar sites, and every nonidentity arm changes all256
(single) or512 (joint) requested scalars. The random controls therefore receive
nonzero interventions even though their answers do not change. All42 complete
argmax witnesses and both Datalog queries replay unchanged without a model.

| exchange | changed answers /6 | donor-answer transfers /6 |
|---|---:|---:|
| selected27/10 | 2 | 2 |
| selected27/12 | 1 | 1 |
| selected joint | 2 | 2 |
| random27/1 | 0 | 0 |
| random27/14 | 0 | 0 |
| random joint | 0 | 0 |
| identity | 0 | 0 |

All transfers are old-to-new. Head27/10 transfers cases0 and54 to their
long-source answers. Head27/12 transfers case54 only, and the selected joint
transfers cases0 and54. There is no rescue in case28 or any reverse direction.
The selected joint does not extend coverage beyond27/10 alone. The random27/1
head's observational shift does not yield an answer transfer in this test.

**Empirical:** these results support a causal role for27/10's gated head output
in two selected source-choice changes, relative to this fixed matched random
comparison. They do not prove that the attention routing alone causes the effect:
the exchanged vector combines attention weights, source values and the query
output gate. Necessity, fresh generalization, the third contrast and a symbolic
copying rule remain open.

The next factorization should separate donor routing weights, donor source
values and donor output gates, with self/donor reconstruction controls, before
calling this a recovered retrieval computation. No model activation may enter
the eventual standalone Datalog runtime. No new incremental certificate is
claimed here.

[Summary](../reference/benchmarks/qwen35_attention_exchange/summary.json),
[model-free replay](../reference/benchmarks/qwen35_attention_exchange_replay.json).

The [completed factorization](qwen35-attention-factor.md) reproduces both forward
rescues with donor routing alone and, separately, donor contextual values alone.
Donor output gating alone transfers none. All scalar and full-vocabulary
reconstruction controls pass. Neither routing nor values are uniquely necessary
from this test; a token-only kernel remains open.
