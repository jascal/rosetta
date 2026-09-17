# Fixed copy guard: generalization test

Oracle: `copy-control`. **Empirical** measurements.
Circuits were copied unchanged from the prior experiment. Dataset and firing domains were frozen before all oracle queries.

| Circuit | Correct / all test | Wrong | Abstained | Negative answers | Exact firing-domain certificate |
|---|---:|---:|---:|---:|---|
| guarded | 288/288 | 0 | 0 | 0/54 | True |
| raw_copy | 216/288 | 72 | 0 | 36/54 | False |

## Guarded circuit by predeclared test condition

| Factor | Value | Correct / total | Wrong | Abstained |
|---|---|---:|---:|---:|
| seed | 2 | 96/96 | 0 | 0 |
| seed | 3 | 96/96 | 0 | 0 |
| seed | 4 | 96/96 | 0 | 0 |
| length | 8 | 96/96 | 0 | 0 |
| length | 12 | 96/96 | 0 | 0 |
| length | 24 | 96/96 | 0 | 0 |
| layout | plain | 72/72 | 0 | 0 |
| layout | noise8 | 72/72 | 0 | 0 |
| layout | noise32 | 72/72 | 0 | 0 |
| layout | near_match8 | 72/72 | 0 | 0 |
| exposures | 2 | 144/144 | 0 | 0 |
| exposures | 3 | 144/144 | 0 | 0 |
| kind | repeat | 144/144 | 0 | 0 |
| kind | intervention | 144/144 | 0 | 0 |
| negative | no_repeat | 0/18 | 0 | 18 |
| negative | one_source | 0/18 | 0 | 18 |
| negative | conflicting_sources | 0/18 | 0 | 18 |

Exact certificates apply only to their stated finite domain, relative to recorded references. A failure is retained
as a counterexample; no successful subset is certified after observing references. Strata are descriptive measurements,
not newly selected rules. Correlated variants share sequence groups. See `report.json` for raw counters and evidence.
