# Qwen3.5 final-MLP gate/up factorization

**Open:** which factors of the final-layer MLP vector carry the observed routing
transfer? The preceding frozen study found three old-to-new donor transfers at
layer31 and none in reverse. This study retains all six directions, including
failed transfers, with no further selection.

For each target context, four forwards replace all 9,216 last-position products:
reconstructed target `SiLU(target_gate) * target_up`, reconstructed donor
`SiLU(donor_gate) * donor_up`, donor gate with target up, and target gate with
donor up. Datalog computes the stable SiLU activation and every substitution.
All payloads, context bytes, source hashes and prior evidence links were frozen
before any new forward. There are 24 forwards.

The reconstructed target and donor products are compared to their earlier
exact-product reference forwards, across the entire vocabulary. Interpretation
requires identical argmax and logit differences no larger than
`0.002 + 0.002 * abs(original)`. This controls numerical error from Datalog double
arithmetic versus the original FP32 SiLU and multiplication. Hybrid transfer
counts are invalid if either reconstruction control fails in any case. The old
exact-product evidence and all new failures remain retained.

Before any new forward, an amendment corrected requested/applied patch checking:
FP32 storage necessarily narrows double-valued Datalog products. The check permits
`1e-7 + 1e-7 * abs(requested)` error. The original protocol is retained beside the
amended protocol with its hash and rationale. No context, mode, patch value,
reconstruction tolerance or outcome was changed. Tests demonstrate acceptance of
ordinary FP32 rounding and rejection of a materially wrong patch.

A donor-gate or donor-up transfer is an empirical intervention outcome on these
six contexts. It does not identify a complete copy algorithm or provide a
standalone Datalog replacement. An eventual extracted rule must still freeze its
runtime guard and pass fresh disjoint confirmation with positive incremental
coverage.

[Protocol](../reference/benchmarks/qwen35_mlp_factor/protocol.json),
[original protocol](../reference/benchmarks/qwen35_mlp_factor/protocol-original.json).
Commands:

```bash
.venv/bin/python py/run_qwen35_mlp_factor.py run
.venv/bin/python py/run_qwen35_mlp_factor.py evaluate
```

During evaluation, the original reconstruction query's correlated maximum was
grounded by every score row, causing quadratic work. That evaluation was
interrupted before any reconstruction report completed. The frozen query remains
unchanged. An evaluation amendment uses a separate query that first projects the
distinct comparison sides, then computes exactly the same maxima. Tests compare
all output relations against the frozen query for passing and failing controls.
No model forward, reference answer, patch, tolerance or outcome was changed.
Resume evaluation with:

```bash
.venv/bin/python py/evaluate_qwen35_mlp_factor_linear.py
python3 py/audit_qwen35_mlp_factor.py
```

The amended verifier itself is retained in each reconstruction certificate.

## Completed outcome

**Empirical:** all 24 forwards complete and all 12 reconstruction controls pass
full-vocabulary logit tolerance and exact argmax checks. Applied FP32 values pass
the amended narrowing check. The summary reports:

| substitution | donor-answer transfers / 6 directions |
|---|---:|
| reconstructed target | 0 |
| reconstructed donor | 3 |
| donor gate, target up | 3 |
| target gate, donor up | 1 |

All three donor-gate transfers are the same old-to-new contrasts as the full
product donor. The donor-up transfer occurs in one of those three groups. None
transfers the old answer in reverse. Thus donor gate changes are sufficient for
all three tested rescues while retaining the target up projection. This does not
show that the up projection is irrelevant, or that a gate vector alone computes
the answer: the target residual, value vector, projection and readout remain.

This is still a whole-width intervention on deliberately selected development
contrasts. **Open:** isolate a small causal subset of gate changes and identify a
token-computable rule for their effect. No new standalone circuit or incremental
finite-domain certificate has yet been produced.

[Summary](../reference/benchmarks/qwen35_mlp_factor/summary.json),
[retained replay audit](../reference/benchmarks/qwen35_mlp_factor/replay-audit.json).

The independent model-free audit also replays all 24 argmax archives and all 14
other queries unchanged. See the
[closed-evidence replay report](../reference/benchmarks/qwen35_mlp_factor_replay.json).

The [completed sparse screen](qwen35-sparse-gate.md) reproduces all three forward
gate rescues using top32 changes, versus zero with matched random subsets. Two
neurons (162,381) are shared across the successful top32 sets; their individual
or joint sufficiency remains untested. This is a causal candidate to test, not
yet an extracted symbolic circuit.
