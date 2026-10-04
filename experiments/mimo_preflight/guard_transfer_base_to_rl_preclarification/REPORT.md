# Frozen Base-to-RL copy-rule transfer

**Empirical, finite-domain probe.** This rule was selected from Base development data by a relaxed near-miss ranking. It failed normal Rosetta admission and is not a certified selected circuit.

Frozen rule: suffix length `7`, minimum agreeing sources `3`. Base development: 2 answer mismatches, 1 causal-pair failures, coverage 14.

| Model | Test answers correct / 144 | Fired | Wrong | Abstained | Exact on frozen firing domain? |
|---|---:|---:|---:|---:|---|
| Base | 16/144 | 16 | 0 | 128 | True |
| RL | 15/144 | 16 | 1 | 128 | False |

The rule and input token IDs are identical across the two evaluations. A clean certificate is scoped to the frozen cases where the rule fires; it does not establish arbitrary-context equivalence.
