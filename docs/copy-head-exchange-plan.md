# Answer-position head exchange discovery

Status: open, before intervention requests. This study does not modify or
reopen the rejected prior-guard confirmation candidate.

The layer-0 MLP same-final-token exchange preserved recipient answers. Later
whole-sequence head deletion established a causal contribution on one correct
historical case, but did not isolate the answer-position write. We now test
whether answer-position head products carry transferable answer information.

Use the seven previously selected historical diagnostic cases and their paired
all-source target edits from the original longest-source development study.
Datalog selects both directions of every pair touched by this fixed selection.
No new case is chosen by intervention outcome. Input contexts, final queries
and lengths are unchanged within each pair; only registered source successors
differ. Retain pairs whose native model answers happen to be equal.

Freeze six head sets: copy (16/9,16/12), semantic (21/2,21/13), their union,
and the corresponding layer-matched controls from the previous seed-503 plan
(16/1,16/8; 21/5,21/3; union). For every recipient run an unpatched baseline,
then identity, zero and paired-donor exchange for each set (19 arms per row).
Patch only the last-position head slices entering `o_proj`. Donor products
come from an independently unpatched donor forward. The joint patch uses
unpatched products at both layers; it is an explicit intervention, not a
naturally occurring activation trajectory.

Use the locally pinned checkpoint, FP32/eager PyTorch, two threads, no sampling
or KV cache. This is an upstream intervention oracle, not a claim that a new
native intervention kernel has been validated. Native historical unpatched
argmax and both recorded logits must agree within 0.002 + 0.002*abs(reference).
Identity patches must preserve the upstream answer and readout logits within
the same tolerance. Capture products once per historical context, retaining
vector hashes and donor/recipient L1 differences. Record global argmax and
recipient-reference, donor-reference and gold logits on every intervention.

Datalog computes completeness, identity/native parity, flips, transfer to a
different donor answer, transfer to the donor's edited target, and per-set/arm
counts. Separate reference-answer eligibility from intended-target transfer;
an answer match is not evidence of literal copying unless the corresponding
targets differ. Compare treated and matched control sets without discarding
nulls or collateral effects. No success threshold admits a runtime circuit.

Freeze code, Datalog, plan, contexts, paired edits, checkpoint hashes and
reference provenance before model requests. Retain raw observations, failed
integrity gates and model-free replay. Tag results empirical, over this selected
historical finite sample. Even specific donor transfer establishes neither a
complete source router nor a standalone symbolic algorithm. The original goal
still requires a new computation, frozen before fresh disjoint final references,
with positive incremental `dl/equiv.dl` certification and Souffle-only runtime.
