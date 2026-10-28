# Qwen3.5 shared-gate intervention

**Open:** are final-MLP neurons162 and381 individually or jointly sufficient for
the observed source-choice transfers? A post-hoc overlap query found both in
all three successful forward top32 sets. Overlap alone is not causal evidence.

This study retains all six directed contrasts. Five arms per contrast substitute
donor-gate/target-up products at neuron162 only, neuron381 only, both jointly,
the context's previous top32 excluding both, and a matched random two-neuron
control. The control takes the first two nonshared neurons from the previously
frozen seed2200 random order. Datalog chooses every site; no new answer is used
to select cases, neurons, or modes. Thirty forwards and all numeric payloads are
frozen before any model invocation.

The remaining-top32 arm can have different size in reverse contrasts because the
two shared neurons were discovered in the forward sets. Every selected site and
count is retained. Other products remain the target model's original values.
The same FP32 narrowing tolerance and prior full-width reconstruction evidence
apply. The summary checks every requested/applied patch and obtains every argmax
from the complete vocabulary in Datalog.

An individual or joint transfer would establish sufficiency only in these
selected development contexts, not necessity, uniqueness, or a token-computable
algorithm. Remaining-neuron transfers may expose alternative sufficient paths.
No outcome counts as standalone conversion or fresh incremental certification.

[Protocol](../reference/benchmarks/qwen35_shared_gate/protocol.json),
[payloads](../reference/benchmarks/qwen35_shared_gate/payload.json).
Commands:

```bash
.venv/bin/python py/run_qwen35_shared_gate.py run
.venv/bin/python py/run_qwen35_shared_gate.py evaluate
python3 py/audit_qwen35_shared_gate.py
```

## Completed outcome

**Empirical negative result:** all 30 forwards complete and the patch/argmax
summary passes. Neuron162 alone, neuron381 alone, both jointly, and the random
two-neuron control each transfer **zero of six** directed answers. Removing both
neurons from the previous top32 leaves 30 neurons per context and still
transfers all three forward answers, with zero reverse transfers.

This rejects individual/joint sufficiency of the two shared neurons for these
contrasts. Their common membership was not an identified causal core. It does
not show that they have no contribution under other conditions; it shows that
the tested joint substitution is insufficient and that the other 30 changes
are sufficient here. All failed substitutions remain retained.

**Open:** reduce those remaining sufficient sets using a preregistered bisection
search, testing both halves before choosing a successful branch, and stopping
when neither half succeeds. That can identify smaller sufficient subsets or
expose an interaction at the tested split, without treating neuron overlap as a
causal selection rule. It still cannot supply a token-only guard or fresh
incremental equivalence certificate by itself.

[Summary](../reference/benchmarks/qwen35_shared_gate/summary.json),
[replay report](../reference/benchmarks/qwen35_shared_gate_replay.json).

The [completed bisection](qwen35-gate-bisection-outcome.md) finds singleton
sufficient interventions at neurons603 and1765 in two contexts and an interaction
stop at seven neurons in the third. These are context-specific candidates,
not a recovered common rule.
