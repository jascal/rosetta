# Fixed copy guard: generalization test

Oracle: `fieldrun`. **Empirical** measurements.
Circuits were copied unchanged from the prior experiment. Dataset and firing domains were frozen before all oracle queries.

| Circuit | Correct / all test | Wrong | Abstained | Negative answers | Exact firing-domain certificate |
|---|---:|---:|---:|---:|---|
| guarded | 280/288 | 8 | 0 | 0/54 | False |
| raw_copy | 209/288 | 79 | 0 | 36/54 | False |

## Guarded circuit by predeclared test condition

| Factor | Value | Correct / total | Wrong | Abstained |
|---|---|---:|---:|---:|
| seed | 2 | 91/96 | 5 | 0 |
| seed | 3 | 96/96 | 0 | 0 |
| seed | 4 | 93/96 | 3 | 0 |
| length | 8 | 88/96 | 8 | 0 |
| length | 12 | 96/96 | 0 | 0 |
| length | 24 | 96/96 | 0 | 0 |
| layout | plain | 72/72 | 0 | 0 |
| layout | noise8 | 69/72 | 3 | 0 |
| layout | noise32 | 68/72 | 4 | 0 |
| layout | near_match8 | 71/72 | 1 | 0 |
| exposures | 2 | 139/144 | 5 | 0 |
| exposures | 3 | 141/144 | 3 | 0 |
| kind | repeat | 136/144 | 8 | 0 |
| kind | intervention | 144/144 | 0 | 0 |
| negative | no_repeat | 0/18 | 0 | 18 |
| negative | one_source | 0/18 | 0 | 18 |
| negative | conflicting_sources | 0/18 | 0 | 18 |

Exact certificates apply only to their stated finite domain, relative to recorded references. A failure is retained
as a counterexample; no successful subset is certified after observing references. Strata are descriptive measurements,
not newly selected rules. Correlated variants share sequence groups. See `report.json` for raw counters and evidence.
