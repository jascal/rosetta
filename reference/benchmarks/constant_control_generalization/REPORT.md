# Fixed copy guard: generalization test

Oracle: `constant-control`. **Empirical** measurements.
Circuits were copied unchanged from the prior experiment. Dataset and firing domains were frozen before all oracle queries.

| Circuit | Correct / all test | Wrong | Abstained | Negative answers | Exact firing-domain certificate |
|---|---:|---:|---:|---:|---|
| guarded | 0/288 | 288 | 0 | 0/54 | False |
| raw_copy | 0/288 | 288 | 0 | 36/54 | False |

## Guarded circuit by predeclared test condition

| Factor | Value | Correct / total | Wrong | Abstained |
|---|---|---:|---:|---:|
| seed | 2 | 0/96 | 96 | 0 |
| seed | 3 | 0/96 | 96 | 0 |
| seed | 4 | 0/96 | 96 | 0 |
| length | 8 | 0/96 | 96 | 0 |
| length | 12 | 0/96 | 96 | 0 |
| length | 24 | 0/96 | 96 | 0 |
| layout | plain | 0/72 | 72 | 0 |
| layout | noise8 | 0/72 | 72 | 0 |
| layout | noise32 | 0/72 | 72 | 0 |
| layout | near_match8 | 0/72 | 72 | 0 |
| exposures | 2 | 0/144 | 144 | 0 |
| exposures | 3 | 0/144 | 144 | 0 |
| kind | repeat | 0/144 | 144 | 0 |
| kind | intervention | 0/144 | 144 | 0 |
| negative | no_repeat | 0/18 | 0 | 18 |
| negative | one_source | 0/18 | 0 | 18 |
| negative | conflicting_sources | 0/18 | 0 | 18 |

Exact certificates apply only to their stated finite domain, relative to recorded references. A failure is retained
as a counterexample; no successful subset is certified after observing references. Strata are descriptive measurements,
not newly selected rules. Correlated variants share sequence groups. See `report.json` for raw counters and evidence.
