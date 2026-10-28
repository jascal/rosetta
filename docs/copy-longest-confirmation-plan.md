# Conditional final confirmation for long-context source selection

Registered while seeds 601/602 development collection is still running. No
final model references exist. Proceed only if the frozen development selector
admits exactly one candidate under `copy-longest-development-plan.md`.

Generate the **same complete factor matrix**, three groups per source length,
with new seeds **701/702**, excluding all development and historical tokens.
This is 684 final contexts: both source orders, one-token padding, both gaps,
all three distractor lengths, paired target substitutions and independent
no-repeat/tied-source controls. Retain every row whether the candidate fires or
not. Final group IDs must be disjoint from development; the final splits are
labeled as test data and never fed to the development selector.

Before the first final reference request, retain and hash:

- the selected candidate and its standalone `run.dl`;
- the byte-identical behavior-only baseline;
- the composite with baseline priority and candidate only on baseline abstention;
- all final contexts, paired edits, exclusions and exact generator sources;
- Datalog-computed candidate, baseline, composite and incremental firing domains
  and their predictions, computed using token facts only;
- the selection query/evidence and complete development reference journal;
- the runtime, verifiers, plan, executable and actual bundle hashes.

Run every final context through the pinned build-time oracle after the freeze.
There is no second proposal, replacement rule or threshold adjustment using
final references. Preserve wrong predictions and missing references as failed
obligations. Do not exclude a final row because of its answer.

Required success gates:

1. Candidate, composite and incremental domains are nonempty and contain every
   instance where the respective frozen token-only query says they fire.
2. `dl/equiv.dl` passes on all three domains: zero mismatches, uncovered,
   missing-reference and invalid-input obligations. An empty baseline domain
   is reported as empty, never claimed as a passing certificate.
3. Incremental coverage is positive on each final seed. Report source order,
   padding, gaps, distractor lengths and paired source-target effects, including
   all abstentions and baseline errors/conflicts.
4. Retain full-final-domain candidate/composite certificates, including expected
   failures caused by deliberate abstention. These do not replace firing-domain
   obligations and do not establish complete model replacement.
5. Relocate and execute the candidate and composite with token/domain facts in
   an otherwise empty directory. Decisions and abstentions match the frozen
   reference-free query outputs. Only Souffle runs at runtime.
6. Hash-check and replay all retained queries without model, weights or fieldrun,
   preserving both successful and failed verdicts and exact relation outputs.

Passing these gates proves equality **only over the stated finite firing
domains relative to the pinned oracle**. Coverage and counterfactual behavior
are empirical. The anatomical explanation left open by layer-0 exchange is not
promoted into a mechanistic proof. If any required gate fails, retain and report
the failure, do not retune using the final answers, and leave the broader goal
unachieved.
