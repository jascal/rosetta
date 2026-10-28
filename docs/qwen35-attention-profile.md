# Qwen3.5 full-attention source profile

**Open:** which full-attention heads change source-directed attention when a
decoy-successor edit changes the model's answer? This observational profile uses
the same six known development contexts (three old/new contrasts). No final
labels or new contexts enter head selection.

The preceding actual4B instrument gate passes: seven forwards, exact identity
and cleanup, independent source/last-position zero substitutions, five complete
vocabulary comparisons (1,241,600 scores), zero tolerance/argmax failures, valid
attention row sums, and all256 intended nonzero zero-substitution doses. All
four instrument queries replay unchanged without a model.

Datalog enumerates all128 heads at the eight full-attention blocks. The probe
captures each final-query attention row and post-gate head vector. Each context's
full score hash must match its earlier unpatched reference. Datalog validates
all dimensions, positions and row masses before computing a profile.

A source position is the **successor** after a matching query suffix (length>=12),
not the preceding query-anchor position. The long role is the unique longest
source's successor; short is a different matching successor. For each head,
measure the old-to-new change in `(long-source attention - short-source
attention)` and sum across the three pairs. Report four leading heads with a
fixed layer/head tie rule. The procedure is frozen before collection.

This is observational ranking, not source retrieval sufficiency or a circuit
certificate. A high shift can be a consequence of earlier processing rather
than the operative mechanism. Gated DeltaNet blocks remain in the forward but
are not measured by this full-attention probe. Causal donor/zero interventions
are required before treating a ranked head as mechanistic evidence.

[Protocol](../reference/benchmarks/qwen35_attention_profile/protocol.json).

```bash
.venv/bin/python py/run_qwen35_attention_profile.py run
python3 py/audit_qwen35_attention_profile.py
```

## Completed observational profile

All six captures reproduce the earlier complete score hashes exactly. Every
attention row and vector is complete and passes the site/row-mass checks. All
nine retained Datalog queries replay unchanged without a model.

| rank | layer/head | total long-minus-short shift | positive contrasts /3 |
|---|---|---:|---:|
| 0 | 27/12 | 1.10279046 | 3 |
| 1 | 31/2 | 0.92502304 | 2 |
| 2 | 27/1 | 0.85537254 | 2 |
| 3 | 27/10 | 0.75197717 | 3 |

These are source-successor attention shifts, not output attribution or measured
causal sufficiency. In particular, 27/12 and27/10 are consistent-direction
candidates among the reported leaders whose head-vector substitutions were
subsequently tested in the linked frozen exchange study. The ranking itself
remains observational and supplies no copy rule or new coverage certificate.

[Profile relations](../reference/benchmarks/qwen35_attention_profile/profile.json),
[exact score reproduction](../reference/benchmarks/qwen35_attention_profile/reproduction.json),
[model-free replay](../reference/benchmarks/qwen35_attention_profile_replay.json).

The [completed donor exchange](qwen35-attention-exchange.md) finds two forward
transfers at27/10, one at27/12, and no extra transfer from their joint. Matched
random heads transfer none with nonzero doses; all identities pass. This is
causal head-output evidence. The linked factorization and Q/K studies subsequently
separated routing, values and output gating and tested competing-successor keys;
a token-computable key and standalone copying rule remain open.
