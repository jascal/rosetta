# Qwen3.5-4B: measured low-memory FP32 reference path

**Empirical instrument validation**, not a minimized-model certificate.
The target is post-trained `Qwen/Qwen3.5-4B`, pinned revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The downloaded checkpoint and
tokenizer files are hashed in the smoke protocol. Vision and MTP weights are
not used in these text-only forwards.

Two CPU paths avoid holding a full FP32 model in RAM:

- `qwen35_stream_reference.py` explicitly runs one decoder block at a time.
- `qwen35_lazy_reference.py` retains the official Transformers forward graph,
  materializing each block before its forward and releasing it afterward.

Both read only needed embedding rows and chunk the vocabulary projection.
Stored BF16/F32 checkpoint values are widened to FP32; weights are not
quantized. No cache is used. These are build-time reference implementations,
not runtime components of a proposed Datalog circuit.

## Validation

The unit tests compare both paths with an ordinary resident model on independent
contexts, tied/untied embeddings and a sharded BF16-stored fixture. The resident
reference uses the same widened checkpoint values. Full vocabulary scores pass
the fixed 2e-6 absolute/relative test tolerance.

The retained synthetic dense-hybrid check covers 60 contexts and 3,840 logits:
zero logit-tolerance or argmax failures. Datalog chooses the global argmax and
checks completeness; its query replays unchanged. See
[`qwen35_stream_preflight`](../reference/benchmarks/qwen35_stream_preflight/).

The **actual 4B checkpoint smoke** uses two distinct raw-text contexts, `The`
and `alpha beta gamma alpha beta`, then repeats `The` after the other context.
All 744,960 paired vocabulary scores pass the preregistered 0.002 absolute +
0.002 relative tolerance, with zero argmax differences. Each backend's repeated
context has an identical full-logit SHA-256, so this check finds no recurrent
state leakage. It does not validate incremental decoding or cached reuse.

The process peaked at **840,636 KiB RSS (about 821 MiB)**, including mapped
pages and fact staging. The six forwards took 5.28–7.18 seconds each with two
CPU threads. This is measured on short contexts; longer contexts and probes
still need their own throughput/memory measurements. The largest decoder block
has 451,695,360 FP32 parameter bytes by metadata analysis, excluding activations
and runtime overhead.

Evidence: [`qwen35_4b_smoke`](../reference/benchmarks/qwen35_4b_smoke/), including
frozen contexts, all checkpoint hashes, full score facts, Datalog parity and
reset verdicts, timings and model-free replay. Native fieldrun is not one of
these two real-checkpoint paths; native 4B parity remains unmeasured.

## Intervention readiness

`qwen35_mlp_reference.py` observes last-position gate, up-projection and product
values, and patches selected products before down-projection. It uses hooks
inside the official lazily loaded graph. A tiny hybrid test verifies identity,
multi-layer/source-position substitutions against a direct resident reference,
invalid-request rejection and cleanup before a following unpatched request.
This does not yet validate an actual-4B intervention panel or identify its copy
mechanism. The old model's layer/head findings are not inherited.

The next gate is a frozen actual-4B probe validation, then fresh model-specific
baseline/localization data. The original requirement for positive incremental
fresh `dl/equiv.dl` certification and Souffle-only runtime remains unmet.
