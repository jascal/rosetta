# Qwen3.5 copying inventory: completed collection and rejected baseline

The complete 300-row discovery inventory is frozen: 150 train and 150 validation
contexts with disjoint group vocabularies. Its input-only Datalog integrity
query passes, including all-source target edits and unchanged final queries.
The fixed behavior-baseline proposal/admission queries and reference extraction
were frozen before collection. Training-only proposals and their complete
reference witnesses were sealed before
validation collection. All 300 references have now been collected. The validation
seal passes. All 300 complete argmax witnesses and five baseline queries replay
unchanged; see the [replay audit](../reference/benchmarks/qwen35_copy_inventory/replay-audit.json).

Collection uses the serial FP32 official-graph reference, two threads and no
cache. A per-row Datalog query selects the global argmax from every vocabulary
score. Complete score facts, verifier and outputs are retained in lossless
SHA-256 archives for model-free replay.

## Fixed baseline outcome

**Empirical:** none of the 12 frozen proposals passes admission. Each fires on
96 training cases with 5 mismatches, and 96 validation cases with 13 mismatches.
Each has firing and successful paired-edit support in three groups per split;
the zero-error requirement is what rejects them. These are overlapping rules,
so their counts must not be added together. No replacement proposals were selected
after observing validation.

The emitted behavior baseline consequently abstains everywhere. Its development
`dl/equiv.dl` result is **not certified**: the firing domain is empty (`ndomain=0`),
even though mismatch and uncovered counts are zero. This is a retained negative
result, not a vacuous proof or successful conversion. See
[admission](../reference/benchmarks/qwen35_copy_inventory/admission.json) and
[certificate](../reference/benchmarks/qwen35_copy_inventory/baseline-certificate.json).

## Complete training diagnostic

**Empirical:** among 144 positive training contexts, the model returns the
annotated source successor in 108, a competing source successor in 26, and a
token absent from the input in 10. Some conditions have tied full-length sources;
these are observational categories, not an overall accuracy score. The examples
share three vocabulary groups and are correlated.

A separate raw ByteLevel byte check finds 108 literal token matches and 36
outputs that are not prefixes of the intended source continuation. None of those
36 departures is explained by splitting or merging the intended continuation's
token bytes. This check does not apply Unicode normalization and does not prove
which source the model read. Both diagnostic queries replay unchanged; neither
selects a circuit. See [training evidence](../reference/benchmarks/qwen35_copy_bytes/).

**Open:** identify the computation that routes between competing sources and
extract a guarded rule. A source-target counterfactual study can distinguish
tracking a competing source from a query-associated default. No extracted Qwen3.5
circuit has yet added fresh, certified coverage. The successful reference and
intervention preflights validate instruments, not the replacement objective.

## User-requested interim diagnostic

At a saved snapshot of the first 50 training observations, 48 are positive
paired-copy cases and two are negative controls. On those **48 positive cases**,
the unchanged source-classification query reports:

| model answer | cases |
|---|---:|
| designated source successor | 37 |
| competing suffix source's successor | 1 |
| another input token | 0 |
| token absent from the input | 10 |

This is **empirical interim source agreement**, not overall accuracy or Datalog
conversion fidelity. All 48 positive cases share one source vocabulary group
and source length32/query length16, so they are highly correlated. Some
conditions contain tied full-length competing sources. No comparison with
Qwen2.5 is justified from this partial, differently sampled cohort.

The snapshot was produced to answer the user's status question after collection
began. It does not alter the frozen inputs, ranking/admission rules or planned
sample size; no candidate is admitted from it. Both diagnostic queries replay
unchanged. Evidence:
[`snapshot-0050`](../reference/benchmarks/qwen35_copy_inventory_monitor/snapshot-0050/).
The classifications describe what the model does. Any replacement must agree
with the model's actual choices over its declared domain, including departures
from literal copying, or abstain under its frozen guard.

## Follow-up source intervention

The [completed decoy counterfactual](qwen35-decoy-counterfactual.md) changes only
the competing source successor in 36 selected training-derived contexts. Thirteen
answers change: five track the new decoy and eight switch to the unchanged long
target. This provides a causal source-value contrast for subsequent internal
localization; it does not add certified circuit coverage.

The [completed MLP exchanges](qwen35-routing-mlp.md) transfer the long-source
answer in all three old-to-new layer31 interventions, with zero reverse
transfers. Layer3 transfers one directed case; layers0 and15 transfer none.
All identity controls and full-site checks pass, and all 78 argmax witnesses plus
76 other queries replay unchanged. These are causal development observations,
not new certified circuit coverage.
