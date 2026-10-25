# Head-deletion discovery: a specific causal contribution, no extracted rule yet

**Empirical**, seven selected historical contexts, whole-sequence query-head
output deletion, pinned f32 Qwen. The 19-arm / 133-request plan was frozen before
upstream or native intervention requests. No new final-test references or
standalone model-equivalent circuit were produced.

## Instrument validation

All **133/133** native argmax answers agree with FP32/eager PyTorch masking the
same `o_proj` input head slices at every position. Both recorded logits meet
`0.002 + 0.002*abs(upstream)` on every request. Empty-head controls reproduce all
seven historical answers and pass the same logit tolerance. Datalog reports
empty `parity_bad`, `empty_bad` and `invalid` relations.

The isolated fieldrun branch adds a validated `/probe/heads` wrapper to the
existing ablation kernel. It leaves normal inference and the original sibling
checkout unchanged. The source patch, baseline commit, executable hash, bundle
hashes, checkpoint hashes and upstream versions are retained in the study.
Rust tests check strict requests, bounds, duplicate rejection and agreement
with the direct ablation call. This finite validation is not universal parity
for all models or interventions.

## Causal results

| Treatment | Correct answers lost (of 2) | Previously wrong cases rescued (of 5) |
|---|---:|---:|
| layer16/head9 | 1 | 0 |
| layer16/head12 | 1 | 0 |
| layer16/heads9+12 | 1 | 0 |
| layer16/heads7+9+12 | 1 | 0 |
| matched layer16 pair control | 0 | 0 |
| matched layer16 triplet control | 0 | 1 |
| layer21/heads2+13 | 0 | 2 |
| matched layer21 pair control | 0 | 1 |
| copy triplet + semantic pair | 1 | 0 |
| matched five-head joint control | 0 | 1 |

Every loss in the table's copy arms is **case 0** (` Ralph` → ` Kend`). The
other correct case, 38, survives every registered intervention. Thus the copy
heads are load-bearing on one demonstrated computation, not a universal or
unique copy mechanism. Their matched layer-16 controls preserve both correct
answers. Whole-sequence deletion may affect both source construction and the
answer; it does not isolate an answer-position write.

Deleting the semantic pair rescues **case 13**, from normalized competing
` STOP` to intended ` tendency`, while matched controls leave that case wrong.
The same semantic pair also rescues case 14, but several controls rescue case
14 too. That second rescue is not specific evidence for the treated pair.
Deleting semantic head 2 alone loses case 0, while deleting heads 2+13 together
preserves it: effects are not monotone in the deletion set.

The output-transformation example, case 76 (`\ty` → actual ` y`), is not repaired
by deleting the copy pair/triplet. Those interventions change the answer to
competing ` cond`. This makes a literal token-normalization repair unsupported:
removing the hypothesized copy path does not restore the literal source token.

All single-head, control and joint observations remain in the report. The
selected sample is too small and biased for population accuracy or necessity
claims. Attribution plus deletion is not sufficiency, and neither establishes
a standalone symbolic implementation.

## Evidence and remaining work

Study directory: `reference/benchmarks/qwen25_copy_heads/`. It contains the
frozen plan, deterministic matched controls, raw upstream/native journals,
native source patch and Datalog instrument verdicts. `certified()` means the
empirical instrument gate, not `dl/equiv.dl` model equivalence.

The original goal still requires a token-only rule that adds coverage over the
frozen behavior baseline, then a fresh frozen firing-domain certificate. These
head effects do not satisfy that requirement. They establish a concrete causal
copy contribution and a competing contribution to investigate, without adding
certified coverage.
