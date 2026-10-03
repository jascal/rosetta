# Binding-blind baseline for a recorded run

Run: `reference/benchmarks/qwen25_05b_copy_generalization` (references recorded; no new oracle queries).
**Empirical.**

## Guarded circuit

k = novel (token, slot) pairs among the unconsumed slots, relative to the guard's selection data. Chance is the
expected gold-token hits of the strong binding-blind baseline (1 if the queried pair is known, else 1/k).

| k | Cases | Model = gold | Chance hits | Guard answered | Guard = model |
|---:|---:|---:|---:|---:|---:|
| 4 | 96 | 88 | 24.0 | 96 | 88 |
| 6 | 96 | 96 | 16.0 | 96 | 96 |
| 12 | 96 | 96 | 8.0 | 96 | 96 |
| all | 288 | 280 | 48.0 | 288 | 280 |
