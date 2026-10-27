# Qwen3.5 sparse gate-change screen

**Open:** can a small subset of final-MLP gate changes reproduce the whole-width
routing transfers? The preceding factorization established three forward gate
rescues and zero reverse transfers, with valid reconstruction controls. This
study retains all six directions without filtering failed directions.

The Datalog planner ranks neurons separately for each target by absolute change
from its original FP32 MLP product to the donor-gate/target-up product, with neuron
ID breaking ties. This is a magnitude heuristic, not an attribution to the
output or evidence of causality by itself. The planner chooses the top32,128,512
neurons. Matched controls use those same budgets from a fixed seed2200 random
neuron order, shared across contexts. A final arm patches the complement of the
top128. Seven arms in six directions yield 42 forwards.

Only selected products change. All other products remain those computed by the
original target model. No neuron, budget or context is added in response to new
answers. All subsets, numeric payloads and source hashes are frozen before new
forwards. Prior full-width gate and reconstruction results remain the reference
controls; the same `1e-7 + 1e-7*abs(request)` FP32 narrowing check applies.

A positive result would identify an empirical sufficient subset on these chosen
contexts. Random-subset transfers or complement transfers may indicate that the
state change is distributed or redundant. A null result rejects this magnitude
selection heuristic at these budgets, not all possible sparse gate circuits.
No result alone establishes a token-computable rule, fresh generalization, or
positive incremental certified coverage.

[Protocol](../reference/benchmarks/qwen35_sparse_gate/protocol.json),
[all subsets and payloads](../reference/benchmarks/qwen35_sparse_gate/payload.json).
Commands:

```bash
.venv/bin/python py/run_qwen35_sparse_gate.py run
.venv/bin/python py/run_qwen35_sparse_gate.py evaluate
```

The reused collection function is hashed by this study and consumes only the
frozen sparse payloads. Complete-vocabulary argmaxes and all patch checks are
Datalog. Every score archive and both planner/summary queries are retained and
replayed at evaluation, with failed summaries preserved.

## Completed outcome

**Empirical:** all 42 forwards complete, the applied-patch summary passes, and
all 42 complete-vocabulary argmax witnesses plus both Datalog queries replay
unchanged without a model.

| subset | donor-answer transfers / 6 directions |
|---|---:|
| top32 | 3 |
| random32 | 0 |
| top128 | 3 |
| random128 | 0 |
| top512 | 3 |
| random512 | 0 |
| complement of top128 | 2 |

All transfers are old-to-new, with zero reverse transfers. Thus 32 changes per
context, about 0.35% of the 9,216-neuron width, suffice for each of the three
selected forward rescues. The selected neuron identities differ by context;
this is not yet one shared 32-neuron circuit. The complement succeeds in two
cases, so the top-ranked subset is not uniquely necessary. One fixed random
order at each budget is a control, not a statistical null distribution.

A post-hoc Datalog overlap query finds 80 distinct neurons across the three
forward top32 sets: 66 appear once, 12 twice, and two in all three sets. The
shared neurons are **162 and381**. This is descriptive selection for a subsequent
intervention, not proof that either neuron is sufficient or necessary. The next
causal test should exchange those two gate factors individually and jointly,
and compare with the remaining top32 neurons, before claiming a shared core.

[Summary](../reference/benchmarks/qwen35_sparse_gate/summary.json),
[model-free replay](../reference/benchmarks/qwen35_sparse_gate_replay.json),
[overlap evidence](../reference/benchmarks/qwen35_sparse_gate_overlap/overlap.json).
No token-only runtime artifact or fresh incremental certificate has yet resulted.

The [completed shared-neuron test](qwen35-shared-gate.md) rejects sufficiency of
162/381, individually and jointly (zero transfers). The remaining 30 changes
retain all three forward rescues. Thus the shared-ID hypothesis failed; overlap
must not be promoted to a circuit discovery.
