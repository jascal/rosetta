# Expanded copy-guard search: suffix length and source support

**Empirical, finite-domain search experiment.** We expanded the fixed search
from six guards (suffix lengths 3/6/9, support 2/3) to 48 guards (suffix
lengths 1–12, support 1–4). The Datalog selection rule was unchanged: a
candidate must make no development errors, pass all causal-pair checks, and
have minimum group coverage in both train and validation. Candidates and
firing domains were frozen before untouched test references were collected.

The paired Base/RL runs used identical data, seed 19, and int8 fieldrun
bundles. The dataset SHA256 is
`73d6f38ffe62f9727ca2fc03bab8edc72760d50a68d7dca1033e70c864232bff`.
Token IDs used by all earlier guarded-copy and copy-stress runs were excluded.

## Results

| Measurement | Base int8 | RL int8 |
|---|---:|---:|
| Expanded candidates admitted | none | none |
| Raw copy, held-out | 122/144 | 113/144 |
| Fixed suffix-9/three-source guard fired | 16/144 | 16/144 |
| Fixed guard correct on firing domain | 16/16 | 15/16 |
| Exact certificate on fixed guard's firing domain | certified | failed (one mismatch) |

The best-coverage candidates with three-source support used suffix lengths
7–9. Each fired on 14 development cases; Base had two mismatches and one
causal-pair failure, while RL had three mismatches and two pair failures.
The broader parameter grid therefore did not find a new admitted circuit.
Candidates with four-source support had no development coverage.

The Base development failures are informative: one repeated context had three
identical copies of a random token sequence, yet the model predicted `41257`
instead of the copied token `16701`. Its paired intervention also failed to
track the replacement target. The issue there is not insufficient repetition
or a suffix that is too short. It is a model answer that violates the proposed
copy relation. The Base diagnostic guard happened to be exact on its 16 fired
test cases, but using that test result to admit it would be leakage; its
development failures correctly kept the selected artifact abstaining. RL
retained one error on the same guard's test domain.

This experiment changes only suffix length and support count. It does not yet
search new concepts such as bounded recency, source ranking, or uncertainty
features. The next search expansion should add such a distinct hypothesis and
evaluate it under the same frozen-development/untouched-test protocol.

The replayable reports are in [`search_base_int8_retry/`](search_base_int8_retry/)
and [`search_rl_int8/`](search_rl_int8/). `search_base_int8_interrupted/`
contains the first run, which stopped before requesting any test references
because of an output-path bug; the successful rerun fixed path normalization.

## Base-to-RL transfer

As a follow-up, the best Base development near-miss was chosen by a frozen
ranking: fewest development answer errors, fewest causal-pair failures, highest
coverage, then shortest suffix and lowest support. This selected suffix 7 with
three agreeing sources (2 development errors, 1 pair failure, 14 development
answers). The same exact Datalog program and firing domain were applied to
both models on the paired test set. It certified for Base on 16/16 fired cases
and failed for RL with one mismatch (15/16). See
[`guard_transfer_base_to_rl/`](guard_transfer_base_to_rl/). This is a transfer
probe of a non-admitted hypothesis, not a Base-selected certified circuit.
Test references had already been collected and their summary inspected during
the preceding search, so this is post-hoc descriptive evidence rather than an
independent confirmatory holdout. The Soufflé certificate is still exact for
the saved Base answers on the stated firing domain; it proves no generalization
claim.

## Reproduction

Add `--expanded-search` to `py/benchmark_guarded_copy.py` to select the 48
hypotheses. The default remains the original six-candidate family, preserving
the established benchmark behavior. Reproduce Base and RL on the same seed
and repeatable `--prior` datasets recorded in each run's `protocol.json`.
