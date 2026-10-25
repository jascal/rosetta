# Source-value locality: bigram preservation is not faithful donor transfer

**Empirical**, twelve fixed historical recipients, six treated/control head
sets, all 444 planned full-context interventions plus twelve token/bigram
feature forwards. No cases, feature lengths or thresholds were selected after
these observations. The failed final prior-guard circuit remains unchanged.

Replacing source-site contextual values with values computed on the same
`[preceding token, source token]` preserves all **72/72** self-substitution
answers. A one-token feature changes 13/72 answers. These are interventions
on twelve contexts, not 72 independent natural-language examples.

| head set | token self: preserved / 12 | bigram self: preserved / 12 | contextual donor: reference / target transfers | bigram donor: reference / target transfers |
|---|---:|---:|---:|---:|
| copy pair | 7 | 12 | 1 / 0 | 1 / 0 |
| semantic pair | 12 | 12 | 0 / 0 | 0 / 0 |
| joint treated | 6 | 12 | 2 / 1 | 1 / 0 |
| matched copy control | 11 | 12 | 0 / 0 | 0 / 0 |
| matched semantic control | 12 | 12 | 0 / 0 | 0 / 0 |
| matched joint control | 11 | 12 | 0 / 0 | 0 / 0 |

One-token donor substitutions transfer no donor references or intended targets
under any head set. Donor-reference transfer eligibility is ten paired
directions; all twelve intended targets differ. The two equal-reference
directions remain in all intervention counts. Self substitutions are fidelity
tests, not donor-transfer successes.

The distinguishing counterexample is **38←39**. Contextual donor source values
under the joint patch change ` presum` to the donor's literal ` delete`.
Both token and bigram donor values instead leave ` presum` unchanged. Thus
the successful bigram self checks do not establish a faithful token-pair
implementation of the source transformation.

The bigram donor feature does retain the **77←76** reference transfer, under
either the copy pair or joint patch: ` mutual` → ` y`. This is the donor's
model answer, not its intended literal `\ty`. Bigram joint donor patches flip
eight answers, but only one adopts the donor reference and none adopts the
edited literal target; collateral effects are retained.

All baseline, contextual identity and contextual donor argmax/logit replay
checks against the prior factorization study pass. There are no missing,
invalid, parity or repeated-result failures. Both Datalog queries replay
unchanged without loading the checkpoint. The feature tensors, frozen keys,
raw observations and provenance are retained in
[`qwen25_copy_source_value_locality`](../reference/benchmarks/qwen25_copy_source_value_locality/).

```bash
python3 py/audit_copy_source_value_locality.py
```

**Open:** the bigram feature approximates the value channel well enough to
preserve these unmodified answers, but not to reproduce the one demonstrated
literal donor-target transfer. Longer source context or another value-code
component remains to identify. Self preservation alone may reflect redundancy
or margin and must not be promoted into an exact internal-value claim.

There is still no extracted token-only router/output rule, new incremental
runtime coverage or fresh `dl/equiv.dl` certificate. The saved contextual and
lexical tensors are build-time evidence; they are not a proposed runtime
dependency. The goal remains unchanged and active.
