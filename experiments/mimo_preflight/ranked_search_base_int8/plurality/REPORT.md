# Guarded copy: fresh holdout

Oracle: `fieldrun`. **Empirical** measurements.
Selected guard `(id, suffix length, minimum agreeing sources)`: `[]`.
An empty selection emits an abstaining circuit. Candidates and firing domains were frozen before test references.

| Artifact | Correct / all test | Wrong | Abstained | Frozen firing domain | Exact on firing domain? | Exact on full test? |
|---|---:|---:|---:|---:|---|---|
| baseline | 0/48 | 0 | 48 | 0 | False | False |
| raw_copy | 15/48 | 33 | 0 | 48 | False | False |
| strict_guard | 11/48 | 13 | 24 | 24 | False | False |
| selected | 0/48 | 0 | 48 | 0 | False | False |

A clean firing-domain certificate proves equality only on that explicitly frozen finite subset; it does not
prove equality on abstentions or arbitrary future inputs. Empty domains fail certification. The diagnostic strict
guard is always suffix length 9 / three sources, whether or not admission accepted it. Off-domain measurements
cover no-repeat and conflicting-source controls only. See `report.json` for every split and retained evidence.
