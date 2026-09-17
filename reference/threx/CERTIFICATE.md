# threx · historical temperature measurements (`empirical`)

The separate [composed-circuit certificate](COMPOSED_CERTIFICATE.md) has fresh Datalog evidence for its exhaustive
25-case argmax domain. The temperature measurements below remain historical.

`circuits.dl` carries top-K logits (incidence) per rule; the runtime computes `softmax(logits/T)` in souffle
at a queried `.input temp` (T > 0; use the crisp emitter for T=0). Build-time logits from whole.dl.
`circuits.symbols.dl` is an uncertified token-string rendering; some idioms may be omitted.

- domain: 300 decision windows (W=8)
- tested temperatures: T ∈ {0.7, 0.85, 1.0}, ε = 0.02
- rules: 161 (1 compose, 2 select-gate + 158 n-gram, top-K mean 4.2)

| T | contexts | max TV | verdict |
|---|---|---|---|
| 0.7 | 300/300 | 0.0100 | historical Python check passed |
| 0.85 | 300/300 | 0.0083 | historical Python check passed |
| 1.0 | 300/300 | 0.0099 | historical Python check passed |

**empirical** — historical Python comparisons against supplied reference logits at the three listed temperatures.
These measurements have not been replayed with `dl/equiv_dist.dl`; no interval or full-model tail bound is established.
Runtime: `souffle run.dl` with T > 0. See [certificate scope](../../docs/certificates.md).
