# Fresh Qwen3.5 copy inventory and behavior baseline

Open before model requests. This is discovery/development data, never a final
test for an extracted circuit. Use the pinned FP32 reference backend validated
in the actual-checkpoint smoke and MLP instrument studies.

Freeze 300 raw-token contexts, seeds 2001/2002 as train/validation. Each split
has independent positive and negative groups at source lengths 32/48/64 and
query lengths 16/24/32. Positive groups cross long-source counts 1/2/3,
competing suffix lengths 12/16, both source orders, zero/eight-token gaps and
paired edits of every long-source successor. Negative groups retain no-repeat
and tied conflicting full-source contexts. Keep every factor and error.

Allocate query/target/decoy/background role pools uniformly without inspecting
token strings, activations, feature scores or model answers, from IDs
40000..139999 with seed2000. The generator draws disjoint group vocabularies
from these pools. Record used allocations, actual context tokens and prior
4B smoke exclusions. Future final contexts must exclude all these context
tokens and groups; the present pool alphabet is not an unseen-alphabet claim.

Build a new 4B behavior-only baseline using the existing fixed native-trace
grammar (consensus/nearest/plurality/longest, match horizon12, fixed guards),
with no trace term. On train only rank proposals by correct-4*wrong, then
candidate ID; retain the first12 without later backfill. Freeze these proposals
and their token-only validation firing domains before validation model requests.
Admit only proposals with zero errors on both splits and at least two firing
and successful paired-edit groups per split. Serialize exactly the Datalog
admissions; any baseline certificate is over the stated development domain,
not independent final generalization.

The Qwen2.5 baseline remains historical evidence; its model-specific admission
does not transfer to 4B. No extracted circuit is admitted from this inventory.
After discovery/localization, a separate candidate and fresh disjoint final
inputs must be frozen before final references. All original incremental
coverage, dl/equiv.dl, standalone Souffle and replay requirements remain.

All reference argmax choices are computed by Datalog from complete vocabulary
scores, with lowest-token-ID tie breaking. Retain each query's full score facts
and verifier in a lossless archive with SHA-256, allowing replay without a
checkpoint. Python only constructs inputs, invokes the FP32 oracle, stages
facts and serializes Datalog outputs. Use serial FP32 forwards, two threads,
no cache, and retain a resumable per-row journal. Collect train first; collecting
validation before freezing train-only proposals is forbidden.
