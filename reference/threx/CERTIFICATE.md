# threx — faithfulness certificate (the Rosetta Stone)

The **threx** is a purpose-built tiny rope LM (d=32, 2 layers, vocab 31) with three deliberately-distinct decision types
— *retrieved* (copy), *selected* (gated lookup), *composed* (arithmetic). It is rosetta's seed reference: small enough
to work end-to-end, rich enough that each circuit class appears. The whole-model program (`whole.dl`, ~22k lines) is
`fieldrun export --logic-whole` and is faithful to the model by construction (parity-exact with the model's argmax).

## Proved (Datalog certificate)

| circuit | rule | domain | `dl/equiv.dl` verdict |
|---------|------|--------|-----------------------|
| **composed** | `THING = sumtable[ strength(Bi) + strength(Bj) ]`, fired at `⟨ ∿ Bi Bj · · gɪ` | all 25 bearing pairs | `ncover=25, nmiss=0, nuncov=0` → **CERTIFIED** |

Reproduce: `python3 py/verify_threx.py` (verdict computed in `equiv.dl`, not Python). The negative control
(`tests/test_reference.py::test_equiv_catches_a_wrong_circuit`) confirms the certificate is not vacuous — a corrupted
frame yields `nuncov=1`.

The composed circuit's `strength` map and additive structure were **rediscovered from behavior alone** (no grammar):
causal localization finds the two operand positions, corpus-entropy separates them from the fixed frame, and a labeling
search recovers that the output depends only on the *sum* — then `equiv.dl` certifies the result against the model. See
`py/discover.py` (distilled from `fieldrun/experiments/whole-datalog/composed_discover.py`).

## Empirical (measured, from the seeding experiment)

- **retrieved / selected / structural** circuits are recoverable by grammar-blind corpus suffix-mining; a circuits-only
  program (no forward) reproduced the model on **31/48** sampled contexts with **0 wrong** — faithful wherever it fires.
- the residual (free-choice positions — which call after `⟨`, which bearings after `∿`) is a marginal **prior**, not a
  rule: incompressible by suffix-mining, closed only by a lookup. This is the honest boundary between *rule* and *recall*.
- circuits-only program: **454 lines vs 22,222** (98% smaller) at that coverage.

## Open

- close the residual: certify the retrieved/selected/structural circuits through `equiv.dl` (currently only composed is
  certified here) and add a certified free-choice default → a fully-certified circuits-only program that **drops** the
  forward.
- multi-instance `whole.dl` emission (a `fieldrun` change) would let `equiv.dl` read `ref` without a per-instance Python
  loop — making the whole certification a single Datalog run.

## Provenance

Developed in `fieldrun/experiments/whole-datalog` (branch `feat/whole-datalog-poc`) — the lab notebook. The `.dl` files
here (`whole.dl`, `circuit.dl`, `circuits.dl`) are carried over verbatim as the reference translation.
