# Prior guard: finite development certificates, failed admission

**Proved over the stated finite development domains:** option 5 (minimum
match 32, two agreeing sources, margin 4, strict categorical-prior agreement)
passes `dl/equiv.dl` on all 54 of its frozen firing cases: zero mismatch,
uncovered, missing-reference or invalid obligations. Option 6 (three sources)
passes on 16 cases. These are agreement certificates against the pinned native
host's argmax, not arbitrary-input equivalence or truth guarantees.

**Empirical:** 612 fresh contexts, seeds 911/912, were sampled without
consulting the frozen lexical bands. Options 1–4 fail with respectively
38/159, 15/76, 16/110 and 5/48 mismatches. Option 5 covers 8 train and 46
validation cases; its train cases add no coverage over the old baseline and
exercise only one successful causal group. Option 6 has no train firing cases.
The registered admission query selects no option. No 913/914 final references
were collected.

Both passing artifacts were relocated and run using Souffle alone, with only
`circuits.dl`, `run.dl` and token/domain facts. Decisions and abstentions on all
612 contexts match the pre-reference domain query. The categorical lexical
table was extracted at build time, before these full contexts existed; its
alphabet is shared, so token-disjoint contexts are not an unseen-feature-alphabet
claim. Runtime contains discrete facts and logical rules, with no live model,
neural weights, raw-logit evaluation or Python dependency.

Evidence: [all development certificates](../reference/benchmarks/qwen25_copy_prior_guard/development-certificates.json),
[failed admission](../reference/benchmarks/qwen25_copy_prior_guard/selection.json),
[frozen inputs](../reference/benchmarks/qwen25_copy_prior_guard/protocol.json).
The unchanged option 5 remains **open** for independent confirmation under
the [separate fixed-hypothesis plan](copy-prior-guard-confirmation-plan.md).
