# Copy/induction holdout benchmark

Oracle: `copy-control`. **empirical** corpus measurements.
Selection used train and validation only. Test references were queried after artifact hashes were frozen.
Selected copy length: `[1]` (empty means rejected; selected artifact uses the baseline).

| Artifact | Test correct / total | Coverage | Precision | Wrong / total | Off-domain answers | Exact test certificate |
|---|---|---|---|---|---|---|
| baseline | 0/24 | 0.0% | undefined | 0/24 | 0/4 | not certified |
| raw_copy | 24/24 | 100.0% | 100.0% | 0/24 | 0/4 | proved on stated test domain |
| selected | 24/24 | 100.0% | 100.0% | 0/24 | 0/4 | proved on stated test domain |

Baseline: 24 suffix facts. Runtime sizes (including Datalog scaffolding): baseline=3185 bytes, raw_copy=972 bytes, selected=972 bytes

Synthetic controls certify only their authored task. Real-model certificates are relative to the recorded oracle.
No generalization or faithfulness outside the recorded finite domain is proved. Off-domain here means no-repeat
nonce contexts, not a broad natural-language safety evaluation. See `report.json` for all split metrics and evidence.

**Baseline correction:** a later emitter fix restores shorter contexts in mixed-length batches. The [separate re-evaluation](baseline-correction/README.md) uses training-only rules and recorded references; test coverage is unchanged. Original frozen artifacts and guard certificates are retained.
