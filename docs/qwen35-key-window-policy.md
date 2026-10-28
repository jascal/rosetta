# Frozen competing-source key-window test

This is the single user-requested test, following the repaired uniqueness and
nonvacuous reconstruction checks. It does not produce model coverage or a
standalone runtime circuit. There is no new localization or parameter search.

Use the same six Q/K contrasts and exactly one competing-successor position in
each. Preserve absolute positions by replaying the complete prefix through that
position, or replacing earlier prefix tokens with filler0 or1 while retaining
exactly the last4 or16 tokens. The five fixed feature modes are full_prefix,
window4_fill0, window4_fill1, window16_fill0 and window16_fill1. The source
successor remains the final token at its original index in each probe.

Run full-prefix captures first. Compare every captured competing-key dimension
to the frozen full-context key at `1e-5+1e-5*abs(expected)`. An incomplete expected
relation is a failure, even if a certified bit is present. If the complete
full-prefix key comparison fails, retain the negative outcome and stop without
requesting other probes or substitutions. No numerical tolerance is changed.

If that comparison passes, collect the other24 fixed feature probes. Before any
substitution, freeze all numerical payloads. Captured-key replay arms use
captured kvec everywhere; captured donor-short replay changes only the captured
competing key. They have distinct IDs from full-prefix proxy arms. Both
same-source and donor full-prefix proxy arms are reconstruction controls. The
donor probability and vector references both identify prior Q/K arm6*I+5,
never the earlier full-donor-route factor arm5*I+2. The self reference is the
captured target probability row and head vector.

Require complete expected probability and vector coverage with the same strict
`1e-5+1e-5abs` reconstruction band. Then run24 whole-vocabulary controls before
any local-window substitution: six captured-target, six captured-donor-short,
six same-source full-prefix and six donor full-prefix. Whole-vocabulary controls
also use `1e-5+1e-5abs` and exact argmax. Failure blocks interpretation of local
hybrids; no expected relation may be absent. Local same-source controls and
donor substitutions are then both run for every fixed local mode and contrast.

Report per mode: competing-key matches /6 at1e-5; same-source argmax matches /6
to the captured target; donor argmax matches /6 to the donor_short_keys
intervention. A transfer without a key match is negative for recovery. This is
empirical recovery of an activation-based proxy on six known contrasts, not a
token-computable kernel or a nonempty incremental dl/equiv.dl certificate.

Policy, repaired sources and input facts freeze before feature capture; numeric
payloads freeze before substitution. Retain all failures and model scores.
Stop after the outcome. No new window, filler, layer, head, localization family
or token-only extraction is authorized by this test.
