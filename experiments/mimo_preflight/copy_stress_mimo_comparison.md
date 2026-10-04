# MiMo-7B matched copy stress comparison

**Empirical finite-domain comparison.** Base and RL used the same int8
fieldrun setup, tokenizer, and frozen 342-case dataset (288 positive copy
cases plus 54 negative controls). The strict diagnostic guard was fixed before
test references. Rosetta selected no guard on either model's development data,
so this is a diagnostic rule comparison, not an admitted Rosetta circuit.

| Measure | Base int8 | RL int8 |
|---|---:|---:|
| Raw copy, positive cases | 121/288 | 130/288 |
| Strict guard fired | 48/288 | 48/288 |
| Strict guard correct among fired | 27/48 | 35/48 |
| Positive cases abstained by guard | 240/288 | 240/288 |
| Guard answers on negative controls | 0/54 | 0/54 |
| Exact certificate on firing domain | failed | failed |

The RL model gets 9 more positive cases right with raw copy and 8 more right
within the diagnostic guard's fixed firing domain. This is a small descriptive
gain on this dataset, not evidence that RL produced a generally faithful copy
circuit. Both diagnostic guards have false positives on their firing domains,
and zero coverage on the negative controls does not establish general
selectivity.

Raw accuracy by sequence length (8, 12, 24) was Base 44/96, 35/96, 42/96 and
RL 51/96, 30/96, 49/96. By layout (plain, noise-8, noise-32, near-match-8),
Base scored 47/72, 41/72, 31/72, 2/72; RL scored 52/72, 47/72, 31/72, 0/72.
The near-match distractors are especially disruptive to both models. The
diagnostic guard only fires on the longest sequences with three exposures; its
correct counts by layout were Base 8, 7, 6, 6 of 12 fired cases and RL 9, 9,
7, 10 of 12. It is therefore not a uniform distance-generalization result.

For context, the earlier Qwen2.5 fixed-guard experiment scored 280/288 positive
cases with its selected guard and answered none of 54 negative controls. That
guard was selected and certified on its own firing domain; it is not the same
guard or tokenizer as these MiMo runs. Treat the cross-model contrast as
descriptive, not a controlled model ranking.

Rosetta's result here is a detector-library boundary: raw behavior is
measurable, but the current guard candidates do not yield a certified
replacement for either MiMo checkpoint. The standalone selected artifacts
abstain. The failed diagnostic certificates are useful counterexamples for
improving candidate discovery or guard conditions; they must not be promoted as
proved circuits.

Raw reports and replayable certificate evidence are in
[`copy_stress_base_int8/`](copy_stress_base_int8/) and
[`copy_stress_rl_int8/`](copy_stress_rl_int8/). The two dataset hashes match:
`de24284ad2de19452d7f56f7fa173654f79a419cbce478570e36e892674ffb60`.
