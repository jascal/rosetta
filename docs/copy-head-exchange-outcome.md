# Answer-position head exchange: limited transfer, no extracted circuit

**Empirical**, selected historical development cases, pinned Qwen checkpoint,
FP32/eager PyTorch intervention oracle. The fixed Datalog plan selected six
source-edit pairs, both directions: 12 recipients and 228 interventions.
This is not fresh generalization evidence or a new native intervention kernel.
The rejected prior-guard candidate remains unchanged and rejected.

The independent paired-context Datalog audit passes: donor and recipient differ
only at registered source-successor edits, and final queries are identical.
All 12 unpatched argmax answers and reference/gold logits reproduce the native
historical observations within the declared tolerance. All 72 identity patches
preserve answers and logits. Every registered request is present, and every
donor/recipient head-set difference is nonzero. No parity or invalid row fires.

| head set | zero-patch flips / 12 | donor-patch flips / 12 | donor-reference transfers / 10 eligible | donor-target transfers / 12 |
|---|---:|---:|---:|---:|
| copy: 16/9,16/12 | 5 | 5 | 1 | 0 |
| semantic: 21/2,21/13 | 3 | 1 | 0 | 0 |
| joint treated | 7 | 6 | 2 | 1 |
| matched copy control | 1 | 0 | 0 | 0 |
| matched semantic control | 1 | 0 | 0 | 0 |
| matched joint control | 1 | 1 | 0 | 0 |

Ten directions have different recipient and donor model answers. The other
two, 62↔63, remain in the study; matching their shared answer cannot establish
answer transfer. All 12 paired intended targets differ. A flip alone does not
establish either type of transfer.

The sole literal donor-target transfer is recipient **38**, originally
` presum`: the joint patch from donor 39 changes it to ` delete`, the donor's
actual and intended successor. Neither treated pair alone transfers this
answer. The reverse joint exchange changes ` delete` to ` search`, rather than
the donor's ` presum`; the effect is not a symmetric exchange of an isolated
answer variable.

Recipient **77** changes from ` mutual` to donor 76's actual ` y` under either
the copy pair or joint patch. Donor 76's intended literal successor is `\ty`,
so this is transfer of a model answer, not transfer of the edited literal
target. This distinction is consistent with the earlier observational
output-transformation diagnosis. It does not yield a token-normalization rule.

Unlike the earlier layer-0 MLP exchange, these later answer-position head
products can transfer answer-specific information on demonstrated cases.
Most treated flips still fail to adopt the donor answer, and treated patches
also produce collateral outputs. The sample was selected for diagnosis;
there is no population estimate, unique-mechanism claim or general sufficiency
proof. Joint patching uses unpatched donor products from two different layers,
an explicitly artificial intervention trajectory.

**Open:** source routing and output transformation still need a standalone
token-only implementation. This result localizes an answer-specific contribution
more precisely; it supplies no new `dl/equiv.dl` certificate or incremental
runtime coverage. A later candidate must be frozen before fresh final references.

Evidence directory:
[`qwen25_copy_head_exchange`](../reference/benchmarks/qwen25_copy_head_exchange/).
It retains the frozen plan, raw observations, activation hashes and L1
differences, checkpoint/build metadata, parity/transfer query, paired-context
audit and preflight amendment. The initial compiler error was corrected before
any model request; both that failure and the subsequent missing-observation
rejection are retained. All five queries replay unchanged, including these
preflight failures. The passing `certified()` here is instrument consistency,
not model equivalence.

```bash
python3 py/audit_copy_head_exchange.py
```
