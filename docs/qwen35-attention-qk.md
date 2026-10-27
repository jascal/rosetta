# Qwen3.5 query/key routing factorization

**Open:** is head27/10's routing transfer driven by its query, contextual keys,
or source-successor keys specifically? The preceding study showed two routing-only
rescues with target values and output gating held fixed; it did not locate the
routing change within Q/K computation.

This study fixes the same six directed contrasts. Six captures retain actual
normalized/rotated Q/K at the official eager attention boundary, its scaling and
mask row, plus the already validated values, raw output gate and attention row.
The capture wrapper delegates to the original kernel unchanged and restores it
in `finally`. A tiny official-model test checks distinct grouped KV heads,
reconstructed attention probabilities, unchanged scores and cleanup.

Datalog computes Q/K dot products, the captured scale/mask, stable softmax, and
the target-value/gate head vector. Six modes per direction give36 substitutions:
reconstructed target routing, reconstructed full donor routing, donor query only,
all donor keys only, donor source-successor keys only, and donor competing-source
keys only. All other keys remain target in the two restricted modes. Source
positions follow suffix matches>=12; the unique longest successor is long and
a different successor is competing. Positions are aligned, not remapped; masks,
scales and context lengths must agree across each pair.

All modes retain target contextual values and target output gate. Thus the
reconstructed donor control is compared to the preceding **routing-only**
intervention, not to the full donor head. Target reconstruction is compared to
the target baseline. The policy/source hashes freeze before capture; numerical
payloads and capture bytes freeze before substitutions. Probability and vector
reconstruction tolerances are `1e-5+1e-5abs`. All12 full-vocabulary controls must
pass `.002+.002abs` and exact argmax before any hybrid runs. Applied FP32 scalar
checks retain the established narrowing tolerance. No factor is selected or
changed based on intervention outcomes.

These are build-time activation-based causal tests, not a standalone token
predictor or fresh final equivalence proof. Source-key sufficiency would motivate
local key/query kernel extraction; its failure would remain evidence against
that simplification. No captured tensor or model fallback is allowed in the
final runtime product.

[Protocol](../reference/benchmarks/qwen35_attention_qk/protocol.json).

```bash
.venv/bin/python py/run_qwen35_attention_qk.py capture
.venv/bin/python py/run_qwen35_attention_qk.py prepare
.venv/bin/python py/run_qwen35_attention_qk.py run
.venv/bin/python py/run_qwen35_attention_qk.py evaluate
python3 py/audit_qwen35_attention_qk.py
```

## Completed causal development outcome

All six captures reproduce prior full-score hashes exactly. Datalog probability
and vector reconstruction passes. All12 whole-vocabulary reconstruction controls
pass before the24 hybrids run. The final scalar/argmax summary passes, and all36
argmax witnesses plus15 queries replay unchanged without a model.

| routing substitution | donor-answer transfers /6 |
|---|---:|
| reconstructed target | 0 |
| reconstructed full donor route | 2 |
| donor query, target keys | 1 |
| all donor keys, target query | 2 |
| donor source-successor keys only | 2 |
| donor competing-successor key only | 2 |

The key-only modes all transfer cases0 and54; query-only transfers case0. There
is no transfer in case28 or any reverse direction. Every other hybrid output
remains at its target baseline answer, with no additional off-target change.

The repaired Datalog site-count predicate rejects an empty context set and
requires exactly one long and exactly one competing successor position in every
context. Re-running it on the unchanged frozen facts certifies 1/1 for all six
contexts. The earlier predicate checked only nonzero counts; its stored data
were 1/1, but its certified bit alone did not enforce uniqueness. The strict
[revalidation](../reference/benchmarks/qwen35_key_window_repair/site-counts.json)
now supplies that finite-data check. Therefore the last mode changes one key
vector in head27/10's reconstructed routing, holding its query, all other keys,
all values and output gate fixed. Equivalently, one competing-source dot-product
logit changes before softmax redistributes attention. The unchanged target
values still supply the recovered long-source answer in the two successful
contrasts.

This is a **head-local edge intervention**: the counterfactual key is used only
to reconstruct27/10's output. The model shares KV projections across query heads;
this test does not globally mutate that shared key for the other heads. It does
not establish necessity, universal source preference, or a token-computable key
function. Both old and new keys are contextual model states, still forbidden as
inputs to the final standalone runtime.

The subsequent [frozen key-window test](qwen35-key-window-outcome.md) recovers
all six captured keys with the full-prefix proxy, but none with windows of 4 or
16 using either fixed filler. All donor-window argmaxes match the intervention;
without key matches those are negative recovery results. **Open:** a token-only
key kernel. Query locality remains unproved; no fresh incremental `dl/equiv.dl`
certificate has resulted from these interventions.

[Summary](../reference/benchmarks/qwen35_attention_qk/summary.json),
[replay report](../reference/benchmarks/qwen35_attention_qk_replay.json),
[verified source-position counts](../reference/benchmarks/qwen35_attention_qk_source_counts/counts.json).
