# Qwen3.5 query-history prior diagnostic

**Open:** does isolated query history help explain source-value preference? The
old Qwen2.5 one-token prior guard failed fresh confirmation (13 mismatches among
240 firings). The Qwen3.5 causal work found that successor edits can change
source choice through final-MLP gates, but no common neuron identity explains
all three groups. This motivates a computation-level feature test, not another
claim that a neuron ID is the algorithm.

The fixed diagnostic uses all 144 positive training contexts from the completed
Qwen3.5 inventory. All their answers were known before this study. Validation
labels are not used, and no admission or final certificate follows from this
sample. For each of three training vocabulary groups, obtain logits from the
last query token alone, last four query tokens, and the full repeated query
(16/24/32 tokens). Nine forwards are frozen before requests. No target or decoy
successor appears in these query-only inputs.

Datalog extracts every source successor following a suffix match of at least12
and stages its prior score key, without inspecting answer agreement. Full
vocabulary scores are retained for each query; the source-token scores are an
I/O projection of those outputs. Two previously discovered gate units are also
observed as build-time data, with no substitutions or runtime use.

For each history length, report the unique prior-preferred source's agreement
with the model. Separately report firings/errors of one fixed diagnostic guard:
its prior winner must also be the unique longest-source successor, match at
least24 query tokens, beat a competing source of at least12 by four or more
matching tokens. There is no threshold search, source-support tuning, or model
answer-based domain selection. This guard is a diagnostic, not an admitted
circuit. Missing priors invalidate the query; ties abstain.

A clean training subset would only motivate a new preregistered candidate and
fresh disjoint confirmation. It would not rescue the old failed confirmation,
prove whole-model equivalence, or demonstrate positive new certified coverage.
A failure would show that this isolated-history decomposition is insufficient
at the fixed guard, with counterexamples retained.

[Protocol](../reference/benchmarks/qwen35_query_prior/protocol.json).

```bash
.venv/bin/python py/run_qwen35_query_prior.py collect
.venv/bin/python py/run_qwen35_query_prior.py evaluate
```

## Completed diagnostic: insufficient

**Empirical:** all nine query-only forwards complete. The following are corrected
per-context counts over the same 144 positive training contexts:

| isolated query history | prior source disagreements /144 | fixed guard firings | guard mismatches |
|---|---:|---:|---:|
| last1 token | 58 | 56 | 12 |
| last4 tokens | 58 | 56 | 12 |
| full repeated query | 94 | 48 | 12 |

Longer isolated history does not solve this source-choice gap. At the fixed
rule it removes eight firings without removing any of the 12 mismatches. The
query-only prior can favor the long-source successor while the full model
chooses its competitor. For example, in training context50 the rule predicts
54058 while the model returns107366; changing the long successor in context51
changes the prediction to131673 but leaves the model at107366. These are
retained development counterexamples, not a final-test subset to retune.

The original frozen summary's aggregates used anonymous variables in a join,
counting query groups rather than contexts. The raw per-instance preferences,
guards and errors are intact. A separate Datalog query with explicit instance
variables produces the table above, and a two-context test verifies its counting
scope. Both original and corrected reports are retained; the original aggregate
columns must not be read as per-context counts.

All nine full-score argmax witnesses and three original queries replay unchanged
without a model. The separate corrected-count query also replays unchanged.
No rule is admitted from this post-hoc sample, no validation answers were used,
and no new equivalence certificate or standalone runtime artifact is claimed.
The result rejects the tested isolated-history prior decomposition at this guard;
it does not rule out other token-computable source-choice algorithms.

[Raw diagnostic relations](../reference/benchmarks/qwen35_query_prior/summary.json),
[corrected per-context counts](../reference/benchmarks/qwen35_query_prior_counts/counts.json),
[model-free replay](../reference/benchmarks/qwen35_query_prior_replay.json).
