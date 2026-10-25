# Routing/value factorization: the demonstrated transfer follows source values

**Empirical**, the same twelve historical recipients and six fixed head sets
as the preceding exchange study. All 372 planned upstream interventions were
collected without selecting only the earlier successful transfers. This is
mechanism discovery, not fresh generalization or model-equivalence evidence.

The two joint donor-answer transfers survive when recipient attention routing
is held fixed and only donor values at the registered source-edit positions
are substituted. Donor routing with recipient values transfers no donor answer.

| head set | full exchange: reference / target transfers | all values: reference / target | source-site values: reference / target | routing only: reference / target |
|---|---:|---:|---:|---:|
| copy pair | 1 / 0 | 1 / 0 | 1 / 0 | 0 / 0 |
| semantic pair | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| joint treated | 2 / 1 | 2 / 1 | 2 / 1 | 0 / 0 |
| matched copy control | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| matched semantic control | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| matched joint control | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Reference-transfer eligibility is ten directions with different recipient and
donor model answers; all twelve intended targets differ. Every patch arm has
twelve requests. The two remaining equal-answer directions remain in all
counts and are not treated as evidence of donor-answer transfer.

The literal transfer is again **38←39**, ` presum` → ` delete`, under the joint
treated patch. The other reference transfer is **77←76**, ` mutual` → ` y`,
under either the copy pair or joint patch. The latter is not literal target
transfer: donor 76's intended successor is `\ty`. Neither effect appears with
routing-only patches or matched-control value patches.

These counts do not imply general sufficiency. Joint all-value and source-site
value patches each flip seven answers, but only two adopt the donor reference;
there are collateral outputs. Copy value patches flip five answers, with one
donor-reference transfer. Routing patches also produce isolated collateral
flips (joint treated, semantic treated, matched copy and matched joint), without
donor transfer. Joint interventions substitute separately captured products
at layers 16 and 21, rather than forming a natural donor trajectory.

## Instrument evidence

All 96 selected head products reconstruct as A×V, accounting for Qwen grouped
query heads. Maximum coordinate error is **9.5367431640625e-7**, below the frozen
0.0001 limit. All native baseline, identity and preceding full-exchange argmax
and logit comparisons pass their declared checks. Datalog reports no invalid,
parity, repeated-result or reconstruction failure.

Both Datalog queries replay unchanged without the model. Independently loading
the saved tensors with no checkpoint reproduces every recorded reconstruction
error exactly. Raw observations, float32 attention/value/product tensors,
checkpoint/build identity, all controls and frozen inputs are retained in
[`qwen25_copy_head_factorization`](../reference/benchmarks/qwen25_copy_head_factorization/).

```bash
.venv/bin/python py/audit_copy_head_factorization.py --tensors
```

## Extraction implication

The source-site value channel carries the demonstrated transferable answer
information; broader donor value changes and donor routing are unnecessary
for those two observed transfers. This supplies a more specific target for
extracting the source-value/output transformation.

**Open:** these deep values can depend on preceding context. The experiment
does not establish a context-free token lookup, an input-only attention router,
or a faithful Datalog implementation. There is no new incremental runtime
coverage or `dl/equiv.dl` certificate. The next extraction test must distinguish
token-local source values from context-dependent value computation; it must
not promote these internal products into a runtime model dependency.
