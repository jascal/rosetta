# Post-hoc observations about run A's degenerate prior

Status: **empirical observations; causal explanation open.** Run A remains
closed, its frozen artifacts unchanged, and its zero incremental certified
coverage stands. This note is an interpretation of retained evidence, not a
new experiment or a revision of the original preregistration.

The discovery population contained 288 positive-copy rows and no negative
controls. Datalog fitted its features to agreement versus disagreement with the
strict suffix-3/two-source rule. The seven disagreements were instances
2, 4, 6, 10, 12, 230, and 232. Each fresh split contained 96 positive-copy inputs
and 18 negative controls. The resulting majority signature was good on all
114 training rows, including all 18 negative controls.

With a uniform Boolean signature, run A's bonus becomes candidate firing count.
Trace and shuffled scores were consequently identical across all 1,090 scored
candidates. This algebraic consequence is established by the retained relations.
It does not imply that all raw activations were constant or uninformative.

**Open:** the mismatch between discovery and development populations may have
contributed to poor transfer. These observations do not isolate its causal
contribution, prove that adding controls fixes the prior, or explain why the
96 new positive contexts also all received the same label. The small error class,
feature selection, thresholds, and majority aggregation are other possible
contributors. A discriminating prior was possible under run A's design; the
observed prior was simply not discriminating on its training set.

Run B changes labels to positive-copy versus negative-control task class and
uses fresh select/sign/test splits. That is a new prediction target, not merely
better calibration of model-rule agreement. Good task classification need not
predict rule correctness or improve certified coverage. Its shuffled and
behavior-only controls remain necessary. See the amended
[run-B preregistration](native-trace-prereg2.md).

Evidence: `reference/benchmarks/qwen25_native_trace/discovery.json`,
`feature-7-gate.json`, `dataset.json`, and `selection.json`. Those files and
all original failed certificates remain untouched. The label-degeneracy check
can stop a run after sign collection and before ranking; it cannot know the
model-derived signatures before collecting them.
