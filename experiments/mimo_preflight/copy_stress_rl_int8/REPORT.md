# MiMo copy stress: fixed diagnostic circuits

Oracle source: `fieldrun`. **Empirical** measurements.
The strict guard was fixed at suffix length 9 and three agreeing sources. No guard was reselected after seeing test answers.

| Circuit | Positive correct / total | Wrong | Abstained | Negative answers | Firing-domain certificate |
|---|---:|---:|---:|---:|---|
| strict_guard | 35/288 | 13 | 240 | 0/54 | False |
| raw_copy | 130/288 | 158 | 0 | 36/54 | False |

## Strict guard by test stratum

| Factor | Value | Correct / total | Wrong | Abstained |
|---|---|---:|---:|---:|
| seed | 2 | 8/96 | 8 | 80 |
| seed | 3 | 14/96 | 2 | 80 |
| seed | 4 | 13/96 | 3 | 80 |
| length | 8 | 0/96 | 0 | 96 |
| length | 12 | 0/96 | 0 | 96 |
| length | 24 | 35/96 | 13 | 48 |
| layout | plain | 9/72 | 3 | 60 |
| layout | noise8 | 9/72 | 3 | 60 |
| layout | noise32 | 7/72 | 5 | 60 |
| layout | near_match8 | 10/72 | 2 | 60 |
| exposures | 2 | 0/144 | 0 | 144 |
| exposures | 3 | 35/144 | 13 | 96 |
| kind | repeat | 18/144 | 6 | 120 |
| kind | intervention | 17/144 | 7 | 120 |
| negative | no_repeat | 0/18 | 0 | 18 |
| negative | one_source | 0/18 | 0 | 18 |
| negative | conflicting_sources | 0/18 | 0 | 18 |

The test suite contains correlated variants within 18 sequence groups. Certificates are limited to the recorded finite domains.
See `report.json`, `frozen.json`, `references.json`, and certificate evidence for exact results.
