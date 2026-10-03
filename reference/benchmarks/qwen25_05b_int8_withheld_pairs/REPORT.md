# Fixed copy guard: generalization test

Oracle: `fieldrun`. **Empirical** measurements.
Circuits were copied unchanged from the prior experiment. Dataset and firing domains were frozen before all oracle queries.

| Circuit | Correct / all test | Wrong | Abstained | Negative answers | Exact firing-domain certificate |
|---|---:|---:|---:|---:|---|
| guarded | 166/168 | 2 | 0 | 0/21 | False |
| raw_copy | 166/168 | 2 | 0 | 0/21 | False |

## Guarded circuit by predeclared test condition

| Factor | Value | Correct / total | Wrong | Abstained |
|---|---|---:|---:|---:|
| seed | 5 | 54/56 | 2 | 0 |
| seed | 6 | 56/56 | 0 | 0 |
| seed | 7 | 56/56 | 0 | 0 |
| length | 12 | 166/168 | 2 | 0 |
| layout | plain | 83/84 | 1 | 0 |
| layout | noise8 | 83/84 | 1 | 0 |
| exposures | 2 | 82/84 | 2 | 0 |
| exposures | 3 | 84/84 | 0 | 0 |
| kind | repeat | 84/84 | 0 | 0 |
| kind | intervention | 82/84 | 2 | 0 |
| negative | no_repeat | 0/21 | 0 | 21 |

## Withheld pairs and the binding-blind baseline

k = novel (token, slot) pairs among the unconsumed slots, relative to the guard's selection data. Chance is the
expected gold-token hits of the strong binding-blind baseline (1 if the queried pair is known, else 1/k).

| k | Cases | Model = gold | Chance hits | Guard answered | Guard = model |
|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 12 | 12.0 | 12 | 12 |
| 1 | 36 | 34 | 36.0 | 36 | 34 |
| 2 | 24 | 24 | 12.0 | 24 | 24 |
| 3 | 24 | 24 | 8.0 | 24 | 24 |
| 4 | 24 | 24 | 6.0 | 24 | 24 |
| 5 | 24 | 24 | 4.8 | 24 | 24 |
| 6 | 24 | 24 | 4.0 | 24 | 24 |
| all | 168 | 166 | 82.8 | 168 | 166 |

Exact certificates apply only to their stated finite domain, relative to recorded references. A failure is retained
as a counterexample; no successful subset is certified after observing references. Strata are descriptive measurements,
not newly selected rules. Correlated variants share sequence groups. See `report.json` for raw counters and evidence.
