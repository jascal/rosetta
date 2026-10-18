# Run B: pre-collection protocol and implementation repair

**Status: open experiment; no real run-B observations collected.** This repair
follows the implementation at `c1baa38`. It changes the run-B preregistration
before collection. Run A's inputs, sources, frozen manifest, and evidence remain
unchanged. The completed run-A result remains zero incremental certified coverage.

## Scientific amendment

Run B now uses the plain signed sum `S(I) = sum_f P_f * (V(I,f)-T_f)` and adds
`sum S(I)` over a candidate's firing rows to its behavioral score. Negative
margins keep their actual magnitudes; zero contributes zero. The implementation
previously kept positive sums but replaced non-positive sums with a fixed −4,
which did not implement the registered formula.

The proposed extra `|gate|/mean|gate|` weighting is removed. Raw negative gate
magnitude does not measure SwiGLU openness, and a routing weight times an expert
activation is not the same quantity. The actual gate channel is retained for
all features as descriptive evidence, including value features; it never changes
a margin's weight. Raw channel scales remain a declared limitation. No normalization
or gate-weight alternative will be selected from run-B outcomes.

The post-hoc account of run A is narrowed: the absence of discovery negative
controls is established, and population mismatch is a plausible contributor.
It is not a demonstrated cause of the uniform fresh signature. Run B changes
feature labels to copy-versus-control task class; classification of that class
need not predict model-rule agreement. This experiment tests the combined design,
not the isolated benefit of adding controls or changing aggregation.

## Implementation repairs

- `collect` requires `--split select` or `--split sign`; `final` alone collects
  test. Invalid/empty splits fail before server startup. The default preflight
  is the actual retained run-A preflight, not a nonexistent run-B directory.
- Generated hypotheses now include the fixed token-only runtime rules. The old
  generator emitted only declarations/facts and could not define `nt_candidate`.
  The same 24 bases and 1,344 candidate facts remain in the same order.
- Every feature, agreement, selection, domain, diagnostic, cost, and equivalence
  query retains input facts, program, outputs, verdict, hashes, and query timing.
- The registered sign-split interventions now execute: target zero, adjacent
  control zero, identity, and paired opposite-class swap. Datalog selects the
  first eight instances of each class and checks every planned result is present,
  with no extra results and exact identity consistency. Baseline/target logits,
  signed regimes, and local effects are retained.
- Freeze requires a non-degenerate selection and completed diagnostic. Its
  nonempty domain reader correctly unpacks single-column Souffle rows. Collection
  and input resealing refuse a frozen study.
- A degenerate selection instead writes a terminal report and seal, retains its
  evidence and observed costs, and refuses diagnostics/final collection. This is
  a separate terminal outcome, not a successful replacement certificate.
- The replay audit compares every output relation, checks referenced evidence
  exists, rejects an empty audit, and treats missing/failed standalone artifacts
  as errors. Runtime execution uses the actual frozen test contexts and compares
  both decisions and abstentions against the frozen predictions. It also handles
  the registered degenerate terminal outcome without inventing artifacts.

## Input and registration provenance

The dataset digest is unchanged:
`80bce2e349f8b67e3edc12b1fc89b94d5386b9f97d44fee78cbec116dead0634`.
All seeds, token exclusions, contexts, labels, paired substitutions, and group
boundaries are unchanged. The hypotheses digest changes because the missing
runtime rules are now included. `protocol.json` is resealed against the amended
preregistration and complete hypotheses. This happened before model collection.

## Validation

The authored tests are synthetic controls, not run-B evidence. The new CLI test
starts from the real input generator and runs against an explicitly synthetic
HTTP oracle: select collection → features → sign collection → selection →
diagnostic → freeze → final → evaluation → model-free replay. It reaches a
nonempty firing domain, injects a final counterexample, and requires the failed
certificate to survive replay. A separate CLI run forces a uniform signature,
checks the sealed terminal outcome, and proves final collection is refused.
Tests also check raw-gate independence, the exact negative/zero margin formula,
missing/extra diagnostic requests, frozen-run refusal, the inherited preflight
default, and failure when replay evidence is removed.

See `docs/native-trace-runb-handoff.md` for the executable collection commands.
All real collection remains a subsequent step; no real-run outcome is claimed.

Validated after repair: **228 tests passed** (`.venv/bin/pytest -q tests/`,
124.93 seconds), including all 25 retained run-A query verdicts and the new CLI
controls. `python3 py/verify_threx.py` remains **CERTIFIED 25/25**. The run-A
frozen-file/source audit and run-B input seal both pass; `git diff --check` is clean.
