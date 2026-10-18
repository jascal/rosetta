# Run B handoff after pre-collection review

Branch: `research/native-trace-prereg2`. Run A is completed on upstream master;
its evidence is immutable. **No real run-B model observations have been collected.**
The prior handoff at `c1baa38` overstated readiness: direct function tests missed
broken CLI dispatch, incomplete generated hypotheses, discarded replay evidence,
and a missing registered diagnostic. Read
[native-trace-runb-repair.md](native-trace-runb-repair.md), the amended
[native-trace-prereg2.md](native-trace-prereg2.md), and `AGENTS.md` first.

Run B now uses a plain signed-margin sum, with negative values and zero preserved.
The raw-gate magnitude multiplier is removed before collection. Gate observations
are retained as descriptive evidence. The copy-versus-control feature labels are
a new task, not a proven explanation or fix for run A's null result. Raw margin
scales remain a limitation. Do not change the amended design after collection.

The study directory `reference/benchmarks/qwen25_native_trace_b/` contains sealed
inputs only. Dataset digest remains `80bce2e3…0634`; the hypotheses now include
the complete unchanged candidate runtime and therefore have a new digest. Keep
these committed inputs. Do not regenerate against changed historical datasets.
The unrelated untracked `docs/native-trace-handoff.md` is run A's old operational
note; it is not part of this work.

Validation after repair: **228 tests passed**, run A's 25 query verdicts replay
unchanged, and threx remains **CERTIFIED 25/25**. Tests used a declared synthetic
HTTP oracle; the pinned real model was only checked by hash.

## Commands for a subsequent real run

The same pinned executable and f32 bundle used by run A are in
`/tmp/rosetta-native-qwen-f32/`. The driver validates their hashes against the
retained `reference/benchmarks/native_trace_preflight/` by default. No new oracle
or repeated upstream preflight is required if those identities match.

Back up the three sealed input files before collection. In a fresh study directory,
copy those files rather than generating new inputs. From the rosetta root:

```bash
runb_study=reference/benchmarks/qwen25_native_trace_b
runb_binary=/tmp/rosetta-native-qwen-f32/fieldrun-0948cbe
runb_bundle=/tmp/rosetta-native-qwen-f32/model
.venv/bin/python py/run_native_trace2.py collect "$runb_study" --split select --binary "$runb_binary" --bundle "$runb_bundle"
.venv/bin/python py/run_native_trace2.py features "$runb_study"
.venv/bin/python py/run_native_trace2.py collect "$runb_study" --split sign --binary "$runb_binary" --bundle "$runb_bundle"
.venv/bin/python py/run_native_trace2.py select "$runb_study"
```

Inspect the Datalog non-degeneracy verdict. If degenerate, the driver seals a
terminal report and refuses further collection. Run the audit below and publish
that outcome; do not change thresholds or substitute rules.

Only if non-degenerate:

```bash
.venv/bin/python py/run_native_trace2.py diagnostic "$runb_study" --binary "$runb_binary" --bundle "$runb_bundle"
.venv/bin/python py/run_native_trace2.py freeze "$runb_study"
# Commit and publish the frozen boundary BEFORE the next command.
.venv/bin/python py/run_native_trace2.py final "$runb_study" --binary "$runb_binary" --bundle "$runb_bundle"
.venv/bin/python py/run_native_trace2.py evaluate "$runb_study"
```

For either terminal outcome:

```bash
.venv/bin/python py/audit_native_trace2.py "$runb_study" --write
```

Keep every failed certificate and counterexample. The final deliverable is an
honest, replayable outcome, whether positive, zero coverage, or a degenerate prior.
The driver never permits runtime model fallback. No sibling fieldrun changes,
no subagents, and no reuse or modification of run A's frozen evidence.
