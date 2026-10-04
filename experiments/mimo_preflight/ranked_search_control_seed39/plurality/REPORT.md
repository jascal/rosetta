# Guarded copy: fresh holdout

Oracle: `copy-control`. **Empirical** measurements.
Selected guard `(id, suffix length, minimum agreeing sources)`: `[(1, 3, 2)]`.
An empty selection emits an abstaining circuit. Candidates and firing domains were frozen before test references.

| Artifact | Correct / all test | Wrong | Abstained | Frozen firing domain | Exact on firing domain? | Exact on full test? |
|---|---:|---:|---:|---:|---|---|
| baseline | 0/48 | 0 | 48 | 0 | False | False |
| raw_copy | 32/48 | 16 | 0 | 48 | False | False |
| strict_guard | 24/48 | 0 | 24 | 24 | True | False |
| selected | 48/48 | 0 | 0 | 48 | True | True |

A clean firing-domain certificate proves equality only on that explicitly frozen finite subset; it does not
prove equality on abstentions or arbitrary future inputs. Empty domains fail certification. The diagnostic strict
guard is always suffix length 9 / three sources, whether or not admission accepted it. Off-domain measurements
cover no-repeat and conflicting-source controls only. See `report.json` for every split and retained evidence.
