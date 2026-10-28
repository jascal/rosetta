# Qwen3.5 MLP routing exchange

**Open:** does a last-position MLP state carry the source-choice change caused by
a competing successor edit? The completed 36-context source-edit study provides
causal input contrasts, not internal localization. This development study uses
those contrasts without accessing a final test set.

`dl/qwen35_routing_mlp_plan.dl` chooses the lowest-ID changed-to-long contrast in
each training vocabulary group. Each selected pair has the same final query and
long-source target; only the competing source successor changes. There are three
pairs, six contexts. Both directions are tested. The full plan and context bytes
are hashed before any new forward passes.

For each context, a baseline forward observes every one of the 9,216 MLP products
at layers0,3,15,31 at the last position. At each layer separately, three further
forwards substitute the entire product vector immediately before down projection:
identity from the same baseline, zero, and the opposite context's baseline.
Other layers and positions receive no substitutions. Six baseline and 72 patched
forwards give 78 forwards. Layer0 comes from the historical Qwen2.5 hypothesis;
its localization does not transfer by assumption. Layers3,15,31 sample early,
middle, and late full-attention blocks in the hybrid model.

Datalog builds all substitution payloads. Python serializes observations, invokes
the official reference graph, and stages the applied values. Each full-vocabulary
argmax is computed in Datalog with lossless reference evidence. The summary
requires baseline agreement with the prior context references, byte-identical
identity score hashes, and exact agreement between requested and applied values.
Identity failure invalidates interpretation and is retained.

The main observational outcome is a donor-answer transfer under an exchange,
reported separately by layer and direction. Zeroing-induced answer changes alone
are disruption, not a recovered computation. Identity controls check substitution
fidelity, while reverse exchanges test directional asymmetry. All failures remain
in the evidence. None of these interventions is a standalone replacement, a
finite-domain equivalence proof, or positive incremental certified coverage.

Commands:

```bash
.venv/bin/python py/run_qwen35_routing_mlp.py run
.venv/bin/python py/run_qwen35_routing_mlp.py evaluate
```

[Protocol](../reference/benchmarks/qwen35_routing_mlp/protocol.json),
[plan](../reference/benchmarks/qwen35_routing_mlp/plan.json).
The runner checks frozen hashes before resuming. Completed forwards are retained
atomically and reused; no outcome-dependent case or layer substitution is allowed.

Before interpreting the outcome, a supplemental completeness audit checks every
baseline layer/neuron observation and every full-width patch site. It reconstructs
the expected identity/zero/donor values in Datalog, checks exact requested/applied
values, and checks each pre-patch value against the target baseline. This audit
adds validation without changing the frozen cases, sites, interventions, or
outcome definitions. Run it after collection and before closing the evaluation:

```bash
.venv/bin/python py/audit_qwen35_routing_integrity.py
.venv/bin/python py/run_qwen35_routing_mlp.py evaluate
```

The supplemental baseline check (`.venv/bin/python
py/check_qwen35_routing_baselines.py`) compares all six newly observed argmaxes
to their original references, retaining and replaying all six score archives.
After closure, `python3 py/audit_qwen35_routing_replay.py` replays all argmax
archives and Datalog payload/plan/summary/integrity/baseline queries without
loading Torch or a model. Its report lives beside the closed study directory,
so replay does not change the closed artifact.

## Completed outcome

**Empirical:** all 78 forwards completed. All six baselines reproduce their
previous argmaxes. All 24 identity interventions preserve full score hashes.
The complete-width audit verifies every observed/requested/applied site,
including pre-substitution values. The intervention summary passes. All 78 argmax witnesses and 76 other Datalog
queries replay unchanged without a model; see the
[replay report](../reference/benchmarks/qwen35_routing_mlp_replay.json).

| layer | donor transfers / 6 directions | zero-induced answer changes / 6 |
|---|---:|---:|
| 0 | 0 | 1 |
| 3 | 1 | 1 |
| 15 | 0 | 1 |
| 31 | 3 | 3 |

All three layer31 transfers are old-to-new: the original context receives the
long-answer choice from its decoy-edited counterpart. None of the three reverse
exchanges transfers the old answer. Layer3 transfers that answer in one group,
also old-to-new. These six contexts are deliberately selected development cases,
not independent draws or a general success-rate estimate.

This supports a causal role for the final MLP state in these routing changes;
it does not show that the MLP alone implements source selection. The unchanged
target residual and downstream normalization/readout also participate. The
layer0 hypothesis does not obtain donor-transfer support at the tested final
position, despite a zeroing effect in one case. Source-position effects remain
untested here.

**Open:** factor the layer31 donor substitution into its gate and up-value
components, with reconstruction controls, to distinguish a transferable value
from an inseparable state change. An eventual token-only rule must still be
frozen and independently certified on fresh inputs. No circuit coverage has
been added by these interventions.

[Summary](../reference/benchmarks/qwen35_routing_mlp/summary.json),
[site integrity](../reference/benchmarks/qwen35_routing_mlp/site-integrity.json),
[baseline reproduction](../reference/benchmarks/qwen35_routing_mlp/baseline-reproduction.json).

The [completed gate/up factorization](qwen35-mlp-factor.md) reproduces all three
forward transfers with donor gates and target up values, versus one with donor
up values and target gates. All reconstruction controls pass. This narrows the
next sparse causal test; it still adds no standalone certified coverage.
