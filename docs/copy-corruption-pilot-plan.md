# Discovery pilot: does copy survive a corrupted query token?

Status: **open**, fixed before the pilot's first model request. This is not a
confirmation experiment or a claim that an anatomical mechanism was extracted.

Short-suffix literal copying failed despite predicting baseline abstentions.
Instead of excluding those errors by token identity, test a different
computation: approximate sequence alignment using longer redundant context.
If a model can continue an earlier sequence despite one corrupted query token,
an approximate-match circuit could cover inputs where exact suffix detectors
have no match. In particular, replacing the final query token with a fresh
token makes every run-B baseline route abstain by construction; the actual
baseline must nevertheless be executed and checked.

Use run C's existing positive plain-layout contexts, two exposures, sequence
lengths 12 and 24, both groups, and both original/target-substituted variants.
These are **historical discovery backgrounds**, not untouched groups. For each
of the eight backgrounds replace one query token at offsets 0, 1, or 3 from
the end. This is 24 new contexts. A replacement is a fresh token absent from
all historical copy/native-trace datasets and the short-prefix development
matrix. Use seed 501; each background group's replacement is shared across
its target-substitution pair and distinct from every other group's replacement.
No token is selected by model behavior or string category.

Keep the target, source copies and all other query tokens unchanged. Datalog
checks that exactly the declared query position changed, and that the source
target-substitution pairs remain as declared. Record the original uncorrupted
answers for comparison; obtain new model references for every corrupted case.

Four token-only approximate-copy hypotheses cross aligned suffix lengths
`{6,12}` and minimum agreeing sources `{2,3}`, allowing at most one unequal
aligned token and requiring at least one inequality. Sources end before the
query suffix starts. Abstain on conflicting successor tokens. The pilot is
descriptive: do not select or certify a production rule from these historical
backgrounds. Report predictions, abstentions, errors, baseline overlap and
counterexamples for every hypothesis and corruption offset.

If this pilot supports approximate alignment, use new, mutually disjoint
development and final groups, with varied lengths, gaps, corruption locations,
two-error and no-source controls, and paired source-target interventions.
Admission and confirmation require a new explicit protocol and the original
goal's zero-error, positive-incremental-coverage gates. Failure of this pilot
is retained; it does not justify changing the corruption matrix in place.
