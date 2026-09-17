# pythia160m · historical unified-cover measurements (`empirical`)

`circuits.dl` combines distributional n-grams with structural point-mass circuits; the runtime computes `softmax(logits/T)` in souffle
at a queried `.input temp` (T > 0; use the crisp emitter for T=0). Build-time logits from the cached logits (cache-only — no oracle).
`circuits.symbols.dl` is an uncertified token-string rendering; some idioms may be omitted.

- domain: 300 decision windows (W=8)
- tested temperatures: T ∈ {0.7, 0.85, 1.0}, ε = 0.02
- rules: 299 (no idioms + 299 n-gram, top-K mean 54.2)

| T | contexts | max TV | verdict |
|---|---|---|---|
| 0.7 | 300/300 | 0.0060 | historical Python check passed |
| 0.85 | 300/300 | 0.0076 | historical Python check passed |
| 1.0 | 300/300 | 0.0100 | historical Python check passed |

**empirical** — historical Python comparisons against supplied reference logits at the three listed temperatures.
These measurements have not been replayed with `dl/equiv_dist.dl`; no interval or full-model tail bound is established.
Runtime: `souffle run.dl` with T > 0. See [certificate scope](../../docs/certificates.md).

## Argmax leg — structural circuits (398 circuit-behavior instances)

**398/398 match the model at argmax** (`empirical`; historical comparison, no retained Datalog certificate).

| circuit | mechanism | frame-gated | historically matching instances |
|---|---|---|---|
| induction | induction | no | 135 |
| succession | succession | no | 13 |
| ioi | once_appearing | yes | 69 |
| transitivity | once_appearing | yes | 34 |
| modus_ponens | once_appearing | yes | 5 |
| temporal | once_appearing | yes | 69 |
| spatial | once_appearing | yes | 67 |
| syllogism | once_appearing | yes | 6 |

**empirical** — these structural counts were reported by the earlier build. They are retained as historical
measurements and have not been replayed with the current Datalog verifier. No combined certificate is claimed.
