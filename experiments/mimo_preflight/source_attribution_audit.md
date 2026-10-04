# Frozen copy-source attribution audit

**Empirical finite-domain diagnostic.** After the specificity-ranked search,
this Datalog audit asks where each frozen model answer appears among prior
successors whose preceding suffix matches the current query. It records the
longest match per source position up to 16 tokens, the globally best match
length, the nearest source, and the best-specificity source matching the
model's answer. It does not select a replacement rule and does not query the
model; it reuses the fresh seed-39 Base/RL reference files.

Both runs use dataset SHA256
`af9a6422c8da8e25cdb94cfa38e4fe2127c0391ddb4822bdae8d9473bb84c840`.
The Base references SHA256 is
`842c2e14a4e98ed0893b403739160b52266bf359a2577cdbaa6ba577bc89d8c2`; RL is
`bb4e5928cc76130151f97927677b16e3f22e2cb63efd642579392b086357d805`.

## Test results

| Layout | Model | Any prior successor | Nearest source | Best-specificity source | No source match |
|---|---|---:|---:|---:|---:|
| Plain | Base int8 | 6/16 | 6/16 | 6/16 | 10/16 |
| Plain | RL int8 | 12/16 | 12/16 | 12/16 | 4/16 |
| 32-token noise | Base int8 | 7/16 | 7/16 | 7/16 | 9/16 |
| 32-token noise | RL int8 | 7/16 | 7/16 | 7/16 | 9/16 |
| 12-token near match | Base int8 | 9/16 | 2/16 | 7/16 | 7/16 |
| 12-token near match | RL int8 | 9/16 | 1/16 | 8/16 | 7/16 |

Across the three layouts, an answer matched some source successor in 22/48
Base cases and 28/48 RL cases. Of those attributable answers, 20/22 Base and
27/28 RL matched a globally best-specificity source. Only two Base answers
and one RL answer matched the recent 12-token distractor successor. In
contrast, 26/48 Base and 20/48 RL answers matched no source successor within
the 1–16 token suffix range.

This changes the diagnosis from “the model usually picks the wrong copy” to a
split account. When the answer is source-attributable, it usually follows the
longest match; in the near-match cases, the long true source is favored over
the closer decoy. A large share of the remaining errors are not explainable
by selecting among these copy sources at all. They require a different
next-token computation or a broader candidate family. Source attribution is
descriptive: matching a source successor alone does not certify a circuit or
guarantee the answer is correct for every context.

## Artifacts

- [Datalog attribution query](../../dl/copy_source_attribution.dl)
- [Driver and JSON report writer](../../py/audit_copy_source_attribution.py)
- [Base case-level attribution](source_attribution_base_seed39/attribution.json)
- [RL case-level attribution](source_attribution_rl_seed39/attribution.json)
- [Preceding specificity-ranked experiment](source_ranked_copy_search.md)
