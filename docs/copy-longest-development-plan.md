# Long-context source selection: fresh development protocol

Status: **open**, before any new development model requests.

The historical pilot supports long-context selection over a recent conflicting
three-token match in one order, but rejects the unguarded rule. The source-order
effect suggests a boundary anchor, which must now be evaluated on new groups.
This is a computational hypothesis; no anatomical head mechanism is claimed.

Six candidates cross minimum longest-match length `{6,12,24}` with
`{unguarded, boundary-anchored}`. All require a conflicting successor supported
by at least three matching tokens, a winning length margin of at least three,
and a predicted successor appearing once. The boundary condition requires that
the winning longest matched source context start at position zero. All rules
and admission decisions are Datalog. This is a new family, not a mutation of
the rejected pilot artifacts.

Fresh development seeds are **601 and 602**, with three independent groups per
source length `{24,48,64}` and query-prefix lengths `{12,24,32}` respectively.
Each group crosses distractor suffix lengths `{3,6,9}`, both source orders,
zero/eight-token gaps, and paired source-target substitutions. Distractor and
gap tokens are distinct from source tokens except for the deliberately matched
suffix. Negative groups independently provide no-repeat contexts and tied
full-length conflicting matches. Include a counterbalanced one-token leading
padding condition for the long-first layout: it changes source boundary
alignment without changing source order or the matched tokens.

Vocabularies and groups are disjoint across splits and exclude all previous
copy/native-trace inputs, the short-prefix study and both recent pilots.
Record exact exclusions, generator, data, candidates and sources before the
first model request. The historical pilot is discovery only and contributes no
admission or final-test cases.

Admit a candidate only if it has zero mismatches on all firing cases in **each**
development split, positive incremental coverage over the actual frozen
behavior-only baseline in each split, and successful target-substitution pairs
in at least two independent groups per split. Select greatest combined
incremental coverage; ties prefer the longer minimum match, then unguarded over
anchored. If none qualifies, retain every rejection and collect no final test
for this family. Report all candidate/baseline conflicts, baseline errors and
source-order/padding strata, including candidate abstentions.

If admitted, freeze the selected standalone rule, unchanged baseline, composite
router, and fresh final inputs plus token-only firing/incremental domains before
requesting final references. Final groups and vocabularies must be disjoint from
development and all discovery. Include the same factors and negative controls;
do not drop padding, reverse-order or wrong rows after seeing answers. The
original goal's exact finite-domain certificates, positive incremental coverage,
isolated Souffle-only runtime and model-free replay remain mandatory. Passing
development or a selected historical subset does not establish completion.
