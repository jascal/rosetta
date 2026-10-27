# Qwen3.5 source-conditioned two-unit memory kernel

**Open:** can source-local responses explain source competition better than
isolated-query responses? The previous isolated four-token predictor reduced
source disagreement but retained17 guarded errors. It did not include the
activation induced by seeing a source successor.

This diagnostic fixes nine minimal association probes: for each four-token
query and every source successor in its training vocabulary group, run
`query4 + successor + query4`. Observe the two fixed final-MLP products603/1765
and retain full logits. Probe selection uses token structure only. All full
training-context answers are already known; this is development, not fresh
confirmation.

Three fixed mixtures combine source-local products: equal occurrence weights,
maximum matched-suffix-length weights, and matched length divided by distance
from the query. Each source position contributes once using its maximum match,
not once for every matching suffix length. Normalize the weights, then apply
the fixed two-unit output coefficients to compare actual source successors.
These are partial readout scores, not sums of unnormalized unit effects with
normalized full-model logits. No scale or mixture weight is fit to answers.

The unchanged diagnostic guard requires the unique preferred source to agree
with the unique longest successor, match>=24, a competing match>=12, and margin
>=4. All three rules and all144 training contexts are retained, without choosing
a winning mode for final confirmation. Model outputs absent from the candidate
sources still count as errors. Unknown source features invalidate the query;
ties abstain.

During collection, the summary unit test caught a compile-time number/float
mismatch. The original frozen module remains unchanged. A separate typed
module inserts explicit `to_float` casts for the already declared length and
distance weights. The amendment was recorded before any result evaluation;
no probe, formula, coefficient, threshold, case or interpretation changed.
The typed summary test passes, including rejection of a missing feature.

This is a test of associative composition, not proof that the two-unit mixture
is the model's memory algorithm. Even a clean known-training result would need
a separately frozen standalone rule and fresh disjoint confirmation. Weight
facts and build-time activations here are not an allowed final runtime input.

[Protocol](../reference/benchmarks/qwen35_memory_kernel/protocol.json),
[type amendment](../reference/benchmarks/qwen35_memory_kernel/evaluation-amendment.json).
Commands:

```bash
.venv/bin/python py/run_qwen35_memory_kernel.py collect
.venv/bin/python py/evaluate_qwen35_memory_kernel_typed.py
python3 py/audit_qwen35_memory_kernel.py
```

## Completed diagnostic: rejected

All nine association forwards complete, and all three fixed mixtures fail the
zero-error requirement on the known training contexts:

| source mixture | source disagreements /144 | guarded firings | guard mismatches |
|---|---:|---:|---:|
| equal occurrences | 94 | 48 | 12 |
| matched suffix length | 92 | 50 | 12 |
| matched length / distance | 81 | 61 | 12 |

The recency mixture changes preferences and enlarges the guarded domain, but
removes none of the12 mismatches. All mixtures have worse source agreement than
the earlier isolated four-token two-unit predictor (44/144 disagreements), which
itself failed certification. No mixture is selected for final confirmation.
All preferences, source weights, predicted products, scores and counterexamples
remain retained. These results reject the fixed two-unit source-mixture model;
they do not show that the actual recurrent/attention memory is an additive
mixture or that all possible source-conditioned kernels fail.

The remaining extraction gap is upstream source retrieval and its interaction
with the rest of the model state. Last-layer gate sensitivity alone has not
provided a token-computable predictor. No standalone artifact or new certified
coverage results from this diagnostic.

[Summary](../reference/benchmarks/qwen35_memory_kernel/summary.json),
[model-free replay](../reference/benchmarks/qwen35_memory_kernel_replay.json).
