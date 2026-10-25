# Layer-0 copy extraction: discovery and confirmation boundaries

Status: **open**. No new model observations or final-test references have been
collected for this study. Run D's existing observations are discovery evidence.

The goal is positive incremental certified coverage beyond run B's frozen
behavior-only composite, not merely another copy rule. The baseline is
`reference/benchmarks/qwen25_native_trace_b/artifacts/behavior/circuits.dl`.
Its bytes must be retained and hashed before any new comparison. Existing
experiments, including their test splits, are discovery data only.

## First question: what does the layer-0 intervention remove?

Removing all layer-0 MLP products at the answer position flipped 83/96 copy-row
answers. This shows a load-bearing contribution, not a copy algorithm. In
particular, a context-independent transformation of the final token could be
necessary for later attention to perform copying.

The first discovery probe exchanges the entire layer-0 product vector between
contexts with the **same final token but different model answers**. Donors are
selected by Datalog from run C's already observed contexts, choosing the lowest
instance ID among eligible donors. This holds the final token constant while
changing the desired continuation. Identity write-back and whole-vector zero
are required controls on each recipient. Exchange in both directions is
planned when each row has an eligible donor. No donor is selected by its
intervention outcome.

Readouts are recipient-answer preservation, donor-answer transfer, other-answer
flips, and changes in recipient/donor logits. Report copy rows and controls
separately. A failure to transfer does not prove that the vector lacks copy
information; a transfer does not prove sufficiency outside the intact recipient
network. Compare exchange effects to zeroing, and examine component differences
before proposing any symbolic computation. Same-answer donors and synthetic
instrument controls should be added explicitly before collection if used.

Model restoration is a separate preflight. Rebuild from cached upstream revision
`7ae557604adf67be50417f59c2c2f167def9a775`, record converter and runtime hashes,
compare the rebuilt bundle to the historical bundle, then check the new runtime
against recorded unpatched outputs and intervention controls. Never silently
inherit the old executable's validation for a different binary.

## Admission to a confirmation experiment

After discovery, specify the computation and its token-only structural guard.
Both must be Datalog. A behavioral approximation must not be called the
localized mechanism without intervention evidence connecting the two.

Before requesting any final references, freeze:

- the standalone candidate, unchanged baseline, and composite router;
- development observations, selection query and selection result;
- fresh input generator, seeds, complete inputs and historical exclusions;
- token-only firing domains for candidate, baseline, composite and incremental
  candidate coverage, including every disagreement obligation;
- the exact verifier sources and all artifact/input hashes.

Fresh development and final groups and vocabularies must be mutually disjoint
and exclude all discovery datasets used by this study. Final inputs must test
the computation, with counterfactual target changes, varied source distances
and distractors, and negative controls. Choose the matrix before observation;
do not shrink the firing domain after observing failures. Empty incremental
domains fail the goal even when other certificates pass.

`dl/equiv.dl` must certify candidate and composite on their predeclared nonempty
firing domains, with complete references and no mismatches or uncovered
obligations. Datalog must report incremental coverage over baseline abstentions
and all baseline/candidate conflicts. Also retain full-domain certificates,
including failures caused by abstention. A successful finite-domain check is
**proved only over that domain**, relative to the pinned oracle. Measurements
of generalization and intervention effects remain **empirical**.

Execute a relocated runtime using only token/domain facts and the emitted
Datalog; compare decisions and abstentions to the frozen outputs. Replay all
verdicts without the model. Runtime must not load weights, whole.dl, Python or
fieldrun. Preserve failures and counterexamples; any subsequent candidate
requires a new confirmation boundary and untouched inputs.
