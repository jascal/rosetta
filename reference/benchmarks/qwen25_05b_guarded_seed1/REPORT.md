# Guarded copy: fresh holdout

Oracle: `fieldrun`. **Empirical** measurements.
Selected guard `(id, suffix length, minimum agreeing sources)`: `[(1, 3, 2)]`.
An empty selection emits an abstaining circuit. Candidates and firing domains were frozen before test references.

| Artifact | Correct / all test | Wrong | Abstained | Frozen firing domain | Exact on firing domain? | Exact on full test? |
|---|---:|---:|---:|---:|---|---|
| baseline | 0/144 | 0 | 144 | 0 | False | False |
| raw_copy | 137/144 | 7 | 0 | 144 | False | False |
| strict_guard | 16/144 | 0 | 128 | 16 | True | False |
| selected | 96/144 | 0 | 48 | 96 | True | False |

A clean firing-domain certificate proves equality only on that explicitly frozen finite subset; it does not
prove equality on abstentions or arbitrary future inputs. Empty domains fail certification. The diagnostic strict
guard is always suffix length 9 / three sources, whether or not admission accepted it. Off-domain measurements
cover no-repeat and conflicting-source controls only. See `report.json` for every split and retained evidence.

**Baseline correction:** a later emitter fix restores shorter contexts in mixed-length batches. The [separate re-evaluation](baseline-correction/README.md) uses training-only rules and recorded references; test coverage is unchanged. Original frozen artifacts and guard certificates are retained.
