# Copy target diagnosis

controlled diagnosis informed by previous failures; not untouched final-test evidence

Oracle: `fieldrun`. **empirical**: 180/192 correct; 12 wrong.
Exact certificate on entire diagnostic domain: `False`.

| Factor | Value | Correct / total | Wrong |
|---|---|---:|---:|
| seed | 5 | 94/96 | 2 |
| seed | 6 | 86/96 | 10 |
| length | 8 | 86/96 | 10 |
| length | 12 | 94/96 | 2 |
| prefix | 4 | 88/96 | 8 |
| prefix | 6 | 92/96 | 4 |
| gap | 8 | 58/64 | 6 |
| gap | 0 | 63/64 | 1 |
| gap | 32 | 59/64 | 5 |
| target | 11436 | 20/24 | 4 |
| target | 2579 | 23/24 | 1 |
| target | 34341 | 19/24 | 5 |
| target | 1602 | 24/24 | 0 |
| target | 317 | 23/24 | 1 |
| target | 23268 | 24/24 | 0 |
| target | 698 | 23/24 | 1 |
| target | 13551 | 24/24 | 0 |
| token_class | punctuation_newline | 85/96 | 11 |
| token_class | lexical | 95/96 | 1 |
| target_origin | discovery | 86/96 | 10 |
| target_origin | new_target | 94/96 | 2 |

## Paired punctuation-to-lexical substitutions

| Original token | Replacement token | Outcome | Pairs |
|---|---|---|---:|
| 317 | 23268 | recovered | 1 |
| 317 | 23268 | both_correct | 23 |
| 698 | 13551 | recovered | 1 |
| 698 | 13551 | both_correct | 23 |
| 11436 | 2579 | recovered | 4 |
| 11436 | 2579 | regressed | 1 |
| 11436 | 2579 | both_correct | 19 |
| 34341 | 1602 | recovered | 5 |
| 34341 | 1602 | both_correct | 19 |

All pairs edit only the copied-from successors; Datalog checks the declared edits and counts outcomes.
Target labels are experimental metadata, not runtime filters. Correlated variants share 4 background groups.
This diagnoses a hypothesis; it does not select or certify a replacement guard on fresh holdout data.
