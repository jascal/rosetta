# MiMo copy stress: fixed diagnostic circuits

Oracle source: `fieldrun`. **Empirical** measurements.
The strict guard was fixed at suffix length 9 and three agreeing sources. No guard was reselected after seeing test answers.

| Circuit | Positive correct / total | Wrong | Abstained | Negative answers | Firing-domain certificate |
|---|---:|---:|---:|---:|---|
| strict_guard | 27/288 | 21 | 240 | 0/54 | False |
| raw_copy | 121/288 | 167 | 0 | 36/54 | False |

## Strict guard by test stratum

| Factor | Value | Correct / total | Wrong | Abstained |
|---|---|---:|---:|---:|
| seed | 2 | 8/96 | 8 | 80 |
| seed | 3 | 12/96 | 4 | 80 |
| seed | 4 | 7/96 | 9 | 80 |
| length | 8 | 0/96 | 0 | 96 |
| length | 12 | 0/96 | 0 | 96 |
| length | 24 | 27/96 | 21 | 48 |
| layout | plain | 8/72 | 4 | 60 |
| layout | noise8 | 7/72 | 5 | 60 |
| layout | noise32 | 6/72 | 6 | 60 |
| layout | near_match8 | 6/72 | 6 | 60 |
| exposures | 2 | 0/144 | 0 | 144 |
| exposures | 3 | 27/144 | 21 | 96 |
| kind | repeat | 18/144 | 6 | 120 |
| kind | intervention | 9/144 | 15 | 120 |
| negative | no_repeat | 0/18 | 0 | 18 |
| negative | one_source | 0/18 | 0 | 18 |
| negative | conflicting_sources | 0/18 | 0 | 18 |

The test suite contains correlated variants within 18 sequence groups. Certificates are limited to the recorded finite domains.
See `report.json`, `frozen.json`, `references.json`, and certificate evidence for exact results.
