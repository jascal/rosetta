# Specificity-ranked copy-source search

**Empirical finite-domain search and certificate results.** This experiment
tests whether a source with a longer suffix match is more likely to explain a
copy than a nearer, weaker match. The new Datalog rule finds the maximum
matching suffix length across candidate sources, keeps sources at that
specificity, then selects the nearest of those sources inside a frozen
recency horizon. A minimum-support setting counts equally specific sources in
that horizon. The search varies suffix limits 1–16, support 1–3, and horizons
16/32/64/128 (192 candidates).

The corpus uses 102 cases: 24 development, 48 test, and 6 off-domain controls,
with plain, 32-token-noise, and 12-token near-match distractor layouts. It is
fresh and token-disjoint from earlier corpora. SHA256:
`af9a6422c8da8e25cdb94cfa38e4fe2127c0391ddb4822bdae8d9473bb84c840`.
Checkpoint test references were queried only after candidate selection and
firing domains were frozen. The three prior policies reuse the same
checkpoint references by case ID.

## Control

The fixed diagnostic `(maximum suffix 16, minimum support 1, horizon 32)`
predicts the correct answer on all 16 plain test cases and abstains on the
noise and near-match layouts. Its 16-case firing domain is exactly certified.
The normal search selects `(suffix limit 13, support 3, horizon 128)`, which
is also exact on its 8-case test firing domain. This verifies the ranked rule
and the longest-match distractor distinction in Datalog.

## MiMo results

No learned consensus, plurality, recency, or ranked guard passes the normal
development admission criteria for either checkpoint. The fixed diagnostics
are not selected replacements:

| Checkpoint | Diagnostic | Test fired | Correct | Wrong | Layout detail | Exact certificate |
|---|---|---:|---:|---:|---|---|
| Base int8 | Consensus, suffix 9 / support 3 | 16 | 7 | 9 | plain 3/8; noise 4/8; near-match abstains | failed (9 misses) |
| Base int8 | Bounded recency | 32 | 8 | 24 | plain 6/16; near-match 2/16 | failed (24 misses) |
| Base int8 | Specificity-ranked, suffix 16 | 16 | 6 | 10 | plain 6/16; noise and near-match abstain | failed (10 misses) |
| RL int8 | Consensus, suffix 9 / support 3 | 16 | 9 | 7 | plain 6/8; noise 3/8; near-match abstains | failed (7 misses) |
| RL int8 | Bounded recency | 32 | 13 | 19 | plain 12/16; near-match 1/16 | failed (19 misses) |
| RL int8 | Specificity-ranked, suffix 16 | 16 | 12 | 4 | plain 12/16; noise and near-match abstain | failed (4 misses) |

Specificity ranking removes the near-match errors produced by bounded recency:
the true source matches 16 suffix tokens while the distractor matches 12.
For RL it also improves the fixed diagnostic from 9/16 to 12/16 correct,
with the same plain-only firing domain. Base does not improve (6/16 versus
7/16 under consensus). This suggests that RL more often follows the
longest-match preference on this small test sample, but the exact certificate
fails for both models and the broad candidate search still finds no admitted
guard. The result is a model-behavior difference, not yet a faithful circuit.

## Artifacts

- [Ranked Datalog policy](../../dl/copy_guard_ranked.dl)
- [Ranked-source test](../../tests/test_guarded_copy.py)
- Fresh dataset: `copy_guard_stress_seed39_len32.json`; generation manifest is
  `copy_guard_stress_seed39_len32.json.manifest.json`.
- [Deterministic control run](ranked_search_control_seed39/)
- [Base int8 run](ranked_search_base_int8/)
- [RL int8 run](ranked_search_rl_int8/)
