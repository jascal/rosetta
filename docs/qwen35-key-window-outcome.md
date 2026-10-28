# Qwen3.5 frozen competing-source key windows

**Empirical**, 2026-10-26. This is the one study frozen in
[the policy](qwen35-key-window-policy.md), on the same six contrasts and one
competing-source position per context. Absolute positions were preserved.
Sources and policy froze before the 30 feature captures; numerical payloads
froze before the 72 substitutions. No windows, fillers, tolerances or contrasts
were changed after an outcome.

The repaired site-count query certifies exactly one long and one short position
for each context: 0, 1, 28, 29, 54 and 55. It rejects empty input and counts other
than one. This checks the frozen source facts, not new model coverage.
The payload rejects missing reference probabilities/vectors and compares both
donor references to Q/K `donor_short_keys` (`6*I+5`). Captured-key replay and
full-prefix proxy arms have distinct IDs. All 24 reconstruction controls passed
the `1e-5 + 1e-5*abs(reference)` probability, vector and whole-vocabulary logit
checks, with exact argmax, before any local-window substitution was interpreted.

All counts below are out of six. Key match uses that same strict numerical band.
Self argmax compares to the captured target; donor argmax compares to the prior
head-27/10 `donor_short_keys` intervention. An unmatched transfer moves to the
donor answer while its proxy key fails the key comparison.

| Mode | Key matches | Self argmax matches | Donor-short argmax matches | Unmatched transfers | Recovery |
|---|---:|---:|---:|---:|---|
| Full prefix | 6 | 6 | 6 | 0 | empirical |
| Window 4, filler 0 | 0 | 4 | 6 | 2 | negative |
| Window 4, filler 1 | 0 | 4 | 6 | 2 | negative |
| Window 16, filler 0 | 0 | 5 | 6 | 2 | negative |
| Window 16, filler 1 | 0 | 5 | 6 | 2 | negative |

Every local-window donor arm matches the prior intervention's argmax, but none
recovers the captured competing-source key. Each local mode also produces two
transfers without a key match; those are negative results for recovery.
Same-source windows change the answer in two cases at width 4 and one at width
16, further showing why answer transfer alone is insufficient.

The full-prefix proxy recovers the captured key and both argmax controls on this
finite sample. **Open:** whether a token-only kernel can compute that key. No
kernel was extracted, and no additional localization family was started.
These captures, proxies and patches are build-time evidence. There is no new
`dl/equiv.dl` certificate or certified coverage; the intended runtime product
remains standalone Souffle. The study stops here.

Artifacts:

- [Strict site counts](../reference/benchmarks/qwen35_key_window_repair/site-counts.json)
- [Datalog outcome query](../reference/benchmarks/qwen35_key_window/summary.json)
- [Empirical summary](../reference/benchmarks/qwen35_key_window/outcome.json)
- [Independent model-free replay](../reference/benchmarks/qwen35_key_window_replay.json)

The outcome query's `certified` bit checks table completeness only. It is not a
model-equivalence certificate. Negative key-match reports retain their false
bits and complete expected relations.
