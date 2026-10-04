# Bounded-recency copy-guard search

**Empirical finite-domain search and certificate results.** This adds a Datalog
candidate family that limits matching source positions to the most recent
16, 32, 64, or 128 token positions before applying the unanimous-successor
rule. For each horizon, it searches suffix lengths 1–12 and minimum supports
1–3 (144 candidates). The selection and certification gates are unchanged.

All three policies ran on the same frozen 102-case dataset
(`copy_guard_stress_seed38_len32.json`, SHA256
`1cd1bd8504a8b0992eae9050abddc009d956776241f5805b7f16d18014f32d34`). For
each checkpoint, one set of development references drove selection, then the
test and off-domain references were queried after that candidate freeze. The
same checkpoint references were reused by case ID for the other two policies.
Base and RL comparisons are paired but use their own oracle answers. The
`recent` rule and horizon facts are build-time inputs compiled into a
standalone Datalog circuit.

## Deterministic control

On the authored copy-rule control, all policies selected a rule. The
expanded-search plurality guard covered 48/48 held-out cases and was certified
with zero mismatches. Consensus covered 32/48, abstaining on the distractor
layout. Bounded recency selected a conservative rule that covered 8/48. Its
fixed diagnostic guard did make all 16 near-match predictions incorrectly;
recency by itself does not identify which repeated source is relevant.

## MiMo results

No candidate was admitted for Base or RL under consensus, plurality, or
bounded recency. The fixed diagnostic guards show what the policy does when
forced to fire; they are not selected or certified replacements.

| Checkpoint | Policy | Diagnostic cases fired | Correct | Wrong | Exact certificate |
|---|---|---:|---:|---:|---|
| Base int8 | Consensus | 16 | 6 | 10 | failed (10 misses) |
| Base int8 | Plurality | 24 | 6 | 18 | failed (18 misses) |
| Base int8 | Bounded recency | 32 | 8 | 24 | failed (24 misses) |
| RL int8 | Consensus | 16 | 7 | 9 | failed (9 misses) |
| RL int8 | Plurality | 24 | 9 | 15 | failed (15 misses) |
| RL int8 | Bounded recency | 32 | 6 | 26 | failed (26 misses) |

The layouts explain the apparent coverage gain. The fixed recency guard fires
on the near-match layout and is wrong on 15/16 Base cases and 16/16 RL cases
there. It is also less faithful on plain examples than consensus. Restricting
the source window therefore increases firing while preserving the key failure:
the rule cannot select the source that the model treats as relevant. A recent
decoy can dominate even when older exact matches support the intended answer.

This is evidence against a *simple* stale-match explanation. It does not prove
that all source-selection rules fail, and the cross-model difference is
descriptive. The deterministic control confirms that the Datalog mechanism is
expressible and certifiable. The next useful search should add explicit
source-ranking signals (for example, longest suffix match, source recency
conditioned on match specificity, or a learned discrete source class) and
require those signals to be frozen before held-out references are inspected.
Any such rule must still pass the same development gate and an exact
finite-domain certificate before it is called faithful.

## Artifacts

- [Datalog recent-source rule](../../dl/copy_guard_recent.dl)
- [Control run](recency_search_control_seed38/)
- [Base int8 run](recency_search_base_int8/)
- [RL int8 run](recency_search_rl_int8/)
- Frozen dataset and generation manifest are alongside this report.
