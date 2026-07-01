# pythia160m · certificate (unified — T-distributional n-gram + argmax circuits)

`circuits.dl` is ONE cover: the natural-corpus n-gram rules carry top-K logits and the runtime computes
`softmax(logits/T)` at a queried `.input temp` (certified across the T-range by total-variation distance);
the structural circuits are frame-gated point-mass rules routed above/below the n-gram (a circuit predicts a
token, so it is certified at the **argmax** collapse, not by TV). `circuits.symbols.dl` is the legible twin.

## Distributional leg — n-gram cover, T ∈ [0.7, 1.0], ε = 0.02 (300 natural windows, W=8)

| T | contexts | max TV | verdict |
|---|---|---|---|
| 0.7 | 300/300 | 0.0060 | CERTIFIED |
| 0.85 | 300/300 | 0.0076 | CERTIFIED |
| 1.0 | 300/300 | 0.0100 | CERTIFIED |

## Argmax leg — structural circuits (398 circuit-behavior instances)

**398/398 match the model at argmax** → CERTIFIED.

| circuit | mechanism | frame-gated | argmax-certified instances |
|---|---|---|---|
| induction | induction | no | 135 |
| succession | succession | no | 13 |
| ioi | once_appearing | yes | 69 |
| transitivity | once_appearing | yes | 34 |
| modus_ponens | once_appearing | yes | 5 |
| temporal | once_appearing | yes | 69 |
| spatial | once_appearing | yes | 67 |
| syllogism | once_appearing | yes | 6 |

**CERTIFIED** over the stated domain (natural windows ∪ circuit-behavior stimuli). Runtime: `souffle run.dl`.
