# Qwen3.5 copying inventory: frozen inputs, collection in progress

The complete 300-row discovery inventory is frozen: 150 train and 150 validation
contexts with disjoint group vocabularies. Its input-only Datalog integrity
query passes, including all-source target edits and unchanged final queries.
The fixed behavior-baseline proposal/admission queries and reference extraction
are frozen before collection. Validation answers remain unrequested until
train-only proposals are frozen and their reference witnesses sealed.

Collection uses the serial FP32 official-graph reference, two threads and no
cache. A per-row Datalog query selects the global argmax from every vocabulary
score. Complete score facts, verifier and outputs are retained in lossless
SHA-256 archives for model-free replay. Training collection is still running;
there are no completed baseline or final circuit results yet.

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
