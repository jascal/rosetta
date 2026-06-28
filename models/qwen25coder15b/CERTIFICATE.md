# qwen25coder15b · certificate (T-parameterized — the canonical artifact)

`circuits.dl` carries top-K logits (incidence) per rule; the runtime computes `softmax(logits/T)` in souffle
at a queried `.input temp` (T=0 = the argmax collapse). Build-time logits from the cached logits (cache-only — no oracle).


- domain: 300 decision windows (W=8)
- range: T ∈ [0.7, 1.0], ε = 0.02
- rules: 296 (n-gram, top-K mean 19.9)

| T | contexts | max TV | verdict |
|---|---|---|---|
| 0.7 | 300/300 | 0.0046 | CERTIFIED |
| 0.85 | 300/300 | 0.0061 | CERTIFIED |
| 1.0 | 300/300 | 0.0100 | CERTIFIED |

**CERTIFIED across the range** — souffle cdist vs the model's own softmax(logits/T). Runtime: `souffle run.dl`.
