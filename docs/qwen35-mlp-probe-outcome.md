# Actual 4B MLP probe: finite instrument gate passes

**Empirical instrument validation**, pinned Qwen3.5-4B FP32 reference,
one previously frozen five-token smoke context. All seven planned forwards
were collected. This is not a copy-localization or generalization result.

The Datalog plan requests last-position units 0/1 at layers 0,3,15,31.
The identity patch copies all eight baseline products using a Datalog query.
Answer-position zeroing changes eight products; source-position zeroing
changes layer0 units0/1 at position0. Independently installed substitutions
in the explicit streamed forward reproduce each zero arm's vocabulary scores
within the frozen tolerance and have identical argmax answers.

All requested observations and applied-patch records are complete, and every
applied value equals its declared value. Both zero doses are nonempty. Identity
and cleanup have exactly the baseline full-logit SHA-256, all 248,320 scores
unchanged, and matching gate/up/product observations. The baseline also replays
the previous smoke observation. No invalid, logit, argmax, observation or hash
failure is reported.

Zeroing the answer-position units changes all 248,320 score values; the source
patch changes 248,273. Neither changes the greedy answer (token 20956).
Logit movement is therefore not presented as necessity, sufficiency or
localization of copying. The two comparison paths share the installed
Transformers operators; they differ in forward staging and patch installation.
No actual-4B native fieldrun intervention is validated by this study.

All three queries (plan, identity payload and summary) replay unchanged without
checkpoint access. Evidence:
[`qwen35_4b_probe_validation`](../reference/benchmarks/qwen35_4b_probe_validation/),
including the frozen sources and checkpoint hashes, all raw score tensors,
applied values, observations, complete-vocabulary summary and resources.

```bash
python3 py/audit_qwen35_preflight.py
```

This validates the reference-backend unit-substitution operations on the stated
finite context. Full-layer ablations, head/state intervention, donor transfer,
new-model behavior baselines and a token-only copy circuit remain open. Fresh
4B baseline/localization collection can now use this backend; all certification
and standalone-runtime requirements still apply. No new `dl/equiv.dl` success
or incremental certified coverage is claimed.
