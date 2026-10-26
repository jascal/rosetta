# Next model target: Qwen3.5-4B

User instruction: switch to Qwen3.5-4B when it makes sense. Prioritize a migration
preflight after closing the current frozen component-readout study. Do not wait
for indefinite Qwen2.5 parameter sweeps or require the old extraction goal to
succeed before assessing the new target. Retain Qwen2.5-0.5B as a regression
and historical-evidence fixture. All original certification/runtime requirements
remain unchanged; this instruction does not convert old failures into successes.

Target the post-trained **Qwen/Qwen3.5-4B**, rather than the Base checkpoint.
The [official configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
specifies 32 layers mixing three linear-attention blocks with one full-attention
block, gated full attention and Gated DeltaNet state. Old head indices,
attention equations, normalization formulas and localized findings must not be
assumed to transfer. The full-attention layers are 3,7,11,15,19,23,27,31 (zero
indexed); source routing in recurrent layers requires distinct instrumentation.

Read-only local inspection at fieldrun `f6c236b` finds qwen35 conversion and
forward dispatch through `src/qwen35moe.rs`, plus a tiny dense reference harness
at `scripts/qwen35_dense_ref.py`. This is architecture support, not measured
parity for the actual 4B checkpoint. That implementation does not override the
MLP probe or head/neuronal ablation hooks used in the present Rosetta studies.
Transformers support is installed locally. No Qwen3.5 checkpoint was found in
the local Hugging Face cache during this inspection.

Switch experiment collection after these concrete gates:

1. Pin model/tokenizer revision and file hashes. Record text-only scope,
   execution precision and any conversion; do not silently quantize the target.
2. Establish a feasible memory/throughput path. This host has about 14 GiB
   physical RAM, so a 4B FP32 weight copy alone is not a practical resident
   baseline. Measure an explicit lower-memory reference/native configuration
   without killing unrelated processes or borrowing unvalidated precision claims.
3. Run the dense hybrid tiny fixture, then actual-checkpoint forward parity
   on a frozen token-level panel. Include cold and incremental state agreement;
   the recurrent state must reset independently between contexts.
4. Validate the needed deletion/exchange instruments against the reference,
   including identity controls and Gated DeltaNet state effects. Native fieldrun
   changes belong in an isolated sibling branch/PR, not edits to this checkout.
5. Build a fresh model-specific behavior baseline and localization panel. Freeze
   candidate Datalog and firing domains before fresh disjoint final references.

Proceed to the preflight now; defer large model downloads/collections until the
memory, precision and instrument path is concrete. A runtime deliverable still
accepts token facts and emits decisions/abstentions in Souffle alone, with no
fieldrun, Python, checkpoint, neural weights or whole.dl fallback.

First measured check: the existing dense-hybrid synthetic fixture was rebuilt
with its fixed seeds and compared using the provided fieldrun binary. All
60/60 next-token argmax answers match the FP32 Transformers reference. The
`dl/qwen35_preflight.dl` completeness/parity query passes and replays unchanged.
This verifies this tiny fixture, not actual-checkpoint 4B parity, full-logit
equality, incremental state or intervention hooks. Evidence:
[`qwen35_4b_preflight/tiny-forward.json`](../reference/benchmarks/qwen35_4b_preflight/tiny-forward.json).

Subsequent progress: the checkpoint/tokenizer revision is now pinned and
downloaded. Two memory-bounded FP32 reference paths pass actual-checkpoint
full-vocabulary parity on two distinct smoke contexts plus an independent
repeat, with zero argmax differences and no reset-hash differences. Peak RSS
is about 821 MiB and short-context forwards take 5–7 seconds. This resolves the
initial resident-memory obstacle without weight quantization. Native 4B parity
and actual-checkpoint intervention validation remain open. See the
[reference preflight report](qwen35-reference-preflight.md) for exact scope and
retained evidence. The next work should validate probes and begin fresh 4B
baseline/localization collection, rather than resume the old model's sweeps.
