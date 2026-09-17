# Unified expert packages — the best-practice quickstart

Every expert this workspace ships is ONE kind of artifact (see [`../PACKAGE.md`](../PACKAGE.md)):
a manifest of arbitrating rules where **every answer is cited to its rule, its origin, and its
evidence, and everything else is an abstention**. This page is the shortest path to a
best-practice package from each ingestion route, and the checklist any example here should meet.

## The checklist (what "best practice" means)

1. **Origin labeled** — manifest `origin` (+ `origin_model` for teacher builds); rules may
   override. Routes: `document` (converted content), `teacher` (wyly distillation),
   `feedback` (attributable patches).
2. **Grounding sidecar** — manifest `"grounding"` → the source text ships with the package, so
   runtimes can ATTEST answers (stratum 0) and quote spans.
3. **Strata** — `strata_tau` + per-rule `stratum`; stratum-2 pools must be qualified on the
   deployment distribution, never window statistics.
4. **Calibration** — every arbitration participant calibrated on deployment queries where a
   query set exists (rules, counts tier, stratum-2, and STRUCTURAL choices like canon reps and
   gate slots — mine them from deployment, not fit statistics).
5. **Canon where phrasings vary** — a mined template inventory (`canon` section) with
   serveability-scored reps: a canonical form is only canonical if the tables can SEE it
   (reach) and ANSWER it (measured). Parse-or-abstain; never raw-fragment fallback.
6. **A scorecard** — the build's claim, backed (EXPERTS.md): coverage/precision/abstention on
   held-out + off-domain sets.

## Route 1: document → package (classic conversion)

```bash
.venv/bin/python py/convert_classic.py ../sgiandubh/package_riscv /tmp/pkg_riscv_unified
# → 3014 document items → ~245k rules (origin: document, norm-id citations, grounding.txt)
```
Serve it with the unified runtime (python `py/serve_package.py` machinery, or
`sgiandubh --rosetta-package`). Note the document-origin semantics: **support 1 is valid**
(the document saying it once is the authority); determinism still gates ambiguity.

## Route 2: teacher → package (wyly distillation, in the pil repo)

The pil harness (`experiments/wyly_lm_v5.py`, WYLY_EMIT=1) emits packages that carry all of the
above; enrich with a canon inventory via `experiments/wyly_canon.py <corpus> <package>`.

## Route 3: feedback → patches

`pil experiments/wyly_feedback.py <rounds> --emit` writes `<package>_patched` with
`origin: feedback` rules, each citing the prompt that produced it.

## Serving contract (what consumers see)

Every decision carries: answer, confidence, rule id, `origin`, `stratum`
(0 attested / 1 certified / 2 supported), citation — plus `canonical` + bindings when the query
was canonicalized, and `attested`/`quote` when the grounding sidecar verifies the statement.
