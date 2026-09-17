# Copy target diagnosis

controlled diagnosis informed by previous failures; not untouched final-test evidence

Oracle: `copy-control`. **empirical**: 192/192 correct; 0 wrong.
Exact certificate on entire diagnostic domain: `True`.

| Factor | Value | Correct / total | Wrong |
|---|---|---:|---:|
| seed | 5 | 96/96 | 0 |
| seed | 6 | 96/96 | 0 |
| length | 8 | 96/96 | 0 |
| length | 12 | 96/96 | 0 |
| prefix | 4 | 96/96 | 0 |
| prefix | 6 | 96/96 | 0 |
| gap | 8 | 64/64 | 0 |
| gap | 0 | 64/64 | 0 |
| gap | 32 | 64/64 | 0 |
| target | 11436 | 24/24 | 0 |
| target | 2579 | 24/24 | 0 |
| target | 34341 | 24/24 | 0 |
| target | 1602 | 24/24 | 0 |
| target | 317 | 24/24 | 0 |
| target | 23268 | 24/24 | 0 |
| target | 698 | 24/24 | 0 |
| target | 13551 | 24/24 | 0 |
| token_class | punctuation_newline | 96/96 | 0 |
| token_class | lexical | 96/96 | 0 |
| target_origin | discovery | 96/96 | 0 |
| target_origin | new_target | 96/96 | 0 |

## Paired punctuation-to-lexical substitutions

| Original token | Replacement token | Outcome | Pairs |
|---|---|---|---:|
| 317 | 23268 | both_correct | 24 |
| 698 | 13551 | both_correct | 24 |
| 11436 | 2579 | both_correct | 24 |
| 34341 | 1602 | both_correct | 24 |

All pairs edit only the copied-from successors; Datalog checks the declared edits and counts outcomes.
Target labels are experimental metadata, not runtime filters. Correlated variants share 4 background groups.
This diagnoses a hypothesis; it does not select or certify a replacement guard on fresh holdout data.
