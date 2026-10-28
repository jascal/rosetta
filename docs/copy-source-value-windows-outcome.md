# Source-window values: eight tokens recover transfer, with remaining errors

**Empirical**, the same twelve historical recipients and six fixed head sets.
All 588 planned full-context interventions and 36 source-window feature
forwards are retained. Windows 4/8/16 were frozen before new model requests;
no size or case was selected afterward.

The eight-token source window recovers the demonstrated literal joint target
transfer **38←39**, ` presum` → ` delete`, which token, bigram and four-token
features do not reproduce. It also retains **77←76**, ` mutual` → ` y`, under
the copy pair and joint patch. The latter remains a donor-reference transfer,
not literal transfer of donor 76's intended `\ty`.

| window | self answers preserved / 72 | donor outcomes matching contextual source-value patch / 72 | joint donor-reference transfers / 10 eligible | joint literal target transfers / 12 |
|---|---:|---:|---:|---:|
| 4 tokens | 70 | 66 | 1 | 0 |
| 8 tokens | 72 | 70 | 2 | 1 |
| 16-token cap | 72 | 72 | 2 | 1 |

The donor-outcome column compares all patched argmax answers, including
collateral outputs, with the contextual source-value control. The eight-token
window disagrees on two joint-patch cases, **1 and 76**. Matching the two
demonstrated transfers therefore does not establish faithful replacement of
the whole value computation. Matched control donor patches transfer no donor
references or intended targets at any tested size.

All source sites are position 12. Four- and eight-token windows genuinely
discard preceding context; the 16-token cap includes the complete 13-token
causal prefix at every source. Its exact intervention replay is a control,
not compression evidence. The observed shorter-window result is limited to
this selected historical panel and is not a universal memory bound.

All baseline, identity and contextual-donor argmax/logit replay checks pass.
Datalog reports no invalid, parity, locality or repeated-result failure. An
additional Datalog query compares every window outcome with its contextual
control; it retains all differences without selecting a passing subset.
All three queries replay unchanged without loading the checkpoint.

Evidence: [`qwen25_copy_source_value_windows`](../reference/benchmarks/qwen25_copy_source_value_windows/),
including the frozen plan, actual window tokens, feature tensors and raw
observations. The inherited free-text scope label mentions token/bigram
values; the authoritative frozen Datalog plan and request relations explicitly
specify 4/8/16-token windows, as executed.

```bash
python3 py/audit_copy_source_value_windows.py
```

**Open:** this narrows the context needed for the demonstrated transfer, but
eight-token values are still not a faithful substitute across the complete
intervention panel. Source routing and final output selection remain to
extract. No contextual tensor, model or weights are proposed as runtime
dependencies. There is no new standalone circuit, incremental certified
coverage or fresh `dl/equiv.dl` success; the original goal remains active.
