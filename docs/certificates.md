# Certificate scope and replay

`proved` means a clean **Datalog verdict over an explicit, nonempty finite domain**. It does not assert behavior on
unseen contexts. Python stages inputs, runs Souffle, and reads its output relations. Candidate discovery is untrusted.

## Exact argmax

`dl/equiv.dl` takes `domain(inst)`, `tok(inst,pos,id)`, and `ref(inst,out)` facts. Every requested instance must have
exactly one reference, including instances with empty token contexts. Missing references are failed obligations;
they must not disappear from the domain. The candidate defines `cdecide(inst,out)`.

`certified()` requires a nonempty domain, complete and unambiguous references, no outside-domain data or predictions,
no mismatches, and no uncovered instances. Missing references, mismatches, and gaps have separate output relations.
`run_equiv` also rejects unequal context/reference list lengths before serializing them.

Python callers use `run_equiv` (or `certify`, which delegates to it) to stage the domain. `run_master`, `certify_T`,
and `check_argmax` also stage explicit domains. Candidate discovery in the idiom learner may use only available
references, but both final emission paths certify all requested windows. Direct Souffle callers must provide
`domain.facts`, one instance ID per line, independently of whether that instance has token or reference rows.
External callers are not covered by the repository caller audit.

For a distribution-producing candidate, `temperature.check_argmax` combines this checker with
`dl/argmax_adapter.dl`. Datalog derives all maximal-probability tokens; a tied conflicting output fails equivalence.
The unified-cover builder retains both finite-grid distributional and argmax evidence in `unified-certificate.json`.
Its circuit domain is selected using oracle agreement, so this is a finite selected-domain certificate, not an
untouched holdout evaluation. Empty circuit domains cannot pass.

```bash
python3 py/verify_threx.py --evidence-dir reference/threx/certificate-evidence
```

This checks the exhaustive 25 bearing pairs against `whole.dl` at build time. A missing oracle answer or failed
Souffle invocation causes failure. The CLI exits nonzero on an unsuccessful certificate.

## Distributional checks

`dl/equiv_dist.dl` takes the same explicit domain and token facts, plus `ref_logit(inst,token,score)`, one positive
`temp(t)`, and `epsilon(e)`. It computes reference softmax, candidate normalization, support-union total variation,
coverage, and `certified()` in Datalog. It rejects missing references, uncovered instances, ambiguous scores or
probabilities, invalid probability mass, invalid parameters, outside-domain data, and TV greater than or equal to epsilon.
Normalization uses a fixed absolute tolerance of 1e-9. These are checks under **Souffle floating-point semantics**,
not exact-real arithmetic proofs.

Cross-platform numerical reproducibility has not been established. Different Souffle versions, compiler builds,
or floating-point environments may affect results near the normalization tolerance or strict TV threshold.
Evidence records the Souffle version and exact outputs; replay on the deployment toolchain is the check, not a
promise of bit-identical floating-point output across platforms. No measured platform-variance bound is claimed.

The certificate compares the candidate to **softmax of exactly the supplied reference logits**. If those logits came
from `/topk`, renormalizing them does not bound the omitted model mass. A full-model distributional claim remains
`open` until the reference includes all logits or an independently checked tail bound. The build records the source
label, source-file hashes when available, and exact reference facts; those hashes identify evidence, not the truth of
an arbitrary oracle's assertions.

The emitter checks a finite grid: the lower endpoint, midpoint, and upper endpoint. These checks do not prove an
interval bound. Missing cached logits remain in the requested domain and prevent certification. There is no automatic
certificate inheritance for `circuits.symbols.dl`, which can omit idioms or change the token representation.

```bash
python3 py/temperature.py 300 8 reference/threx 1.0 0.02 0.7
```

This **rebuilds** `circuits.dl` and emits `certificate.json`, `CERTIFICATE.md`, and one retained check per temperature
under `certificate-evidence/`. The historical reports that predate this checker are marked `empirical`; their recorded
measurements have not been promoted to new Datalog certificates.

## Evidence and replay

Each retained check contains:

- `verify.dl` and `circuit.dl`: the exact verifier and candidate.
- `facts/`: the exact requested domain, tokens, and reference facts (and temperature/epsilon for distributions).
- `outputs/`: the actual Souffle verdict and diagnostic relations.
- `certificate.json`: SHA-256 hashes, reference provenance, tool version, command, and observed verdict.
- `stderr.txt`: verifier diagnostics, including failures.

The copy benchmark trees and threx certificate evidence are intentionally versioned together: 2,241 files totaling
15.4 MiB of file contents in the initial benchmark series, with the largest file about 374 KiB. All 1,121 retained
output CSVs are tracked, including empty verdicts. The `.gitignore` exceptions cover nested evidence outputs under
both `reference/` and `models/`. Failed runs and baseline corrections are retained to preserve the experimental
record. This series remains in Git for self-contained replay; a separate evidence store can be reconsidered if
future series make repository growth material, while preserving content hashes and access to complete bundles.

```bash
python3 py/certificate.py <evidence-directory>
```

Replay verifies the recorded hashes and runs the recorded checker with only its recorded inputs in a fresh temporary
directory. Hash mismatches, process failures, missing output relations, and unsuccessful verdicts fail. Neither the
model nor a Python implementation of equivalence is used. Hashes detect changed evidence; they are not signatures.

To replay without Python, check the hashes independently, create an empty output directory, then run from the retained
check directory:

```bash
souffle verify.dl -I . -F facts -D <empty-output-directory>
```

The raw verdict is the presence of the nullary tuple in `certified.csv`; successful process exit alone is not a proof.

## Runtime and remaining work

The emitted distributional `run.dl` uses only token and positive-temperature facts, producing `cdist` and an explicit
`abstain` relation for tokenized contexts where no rule fires. The crisp path produces `cdecide`. No runtime path may
load `whole.dl`, model weights, or call fieldrun. `complete_cover.py` is a historical build-time experiment, not a
standalone replacement.

Still `open`: interval bounds, top-K tail certificates, independent certification of the symbol representation,
and generalization to unseen inputs. Package cover builds now isolate held-out lines before extraction and fail
requested gates when evidence is missing (see [EXPERTS.md](../EXPERTS.md)); those empirical scorecards do not establish
model equivalence or evaluate the full retrieval cascade.
