# threx — faithfulness certificate (the Rosetta Stone)

The **threx** is a purpose-built tiny rope LM (d=32, 2 layers, vocab 31) with three deliberately-distinct decision types
— *retrieved* (copy), *selected* (gated lookup), *composed* (arithmetic). It is rosetta's seed reference: small enough
to work end-to-end, rich enough that each circuit class appears. The whole-model program (`whole.dl`, ~22k lines) is
`fieldrun export --logic-whole` and is faithful to the model by construction (parity-exact with the model's argmax).

## Proved (Datalog certificate)

| what | domain | `dl/equiv.dl` verdict |
|------|--------|-----------------------|
| **composed circuit** `THING = sumtable[strength(Bi)+strength(Bj)]` | all 25 bearing pairs | `ncover=25, nmiss=0, nuncov=0` → **CERTIFIED** |
| **full circuits program** (`circuits.dl`: composed + minimal-suffix cover) | 300 deduped corpus decision windows (W=8) | `ncover=300, nmiss=0, nuncov=0` → **CERTIFIED** |

Reproduce: `python3 py/verify_threx.py` (composed, 25 cells) and `python3 py/minimize.py 300 8` (full program). Both
verdicts are computed in `equiv.dl`, not Python. The negative control
(`tests/test_reference.py::test_equiv_catches_a_wrong_circuit`) confirms the certificate is not vacuous — a corrupted
frame yields `nuncov=1`.

**The minimized program** (`circuits.dl`): **1 composed rule + 121 suffix rules = 122 rules for 300 decisions**
(59% fewer than memorizing; a few hundred lines vs `whole.dl`'s 22,222). The honest breakdown by rule length —
`{1:10, 2:38, 3:18, 4:32, 5:19, 7:4}` — separates *rule* from *recall*:
- **len-1 (10)** — marginal **priors** (free-choice positions: which call after `⟨`, which bearing after `∿`); these are
  the model's learned prior, an irreducible lookup, not a computation.
- **len 2–5 (107)** — **selected / structural** transitions (e.g. the place gate, frame skeletons).
- **len-7 (4)** — long-context tail, effectively memorized (the W=8 window's worst case).
- **composed (1)** — the arithmetic, the one tier that *computes*: 1 rule for the whole 5×5 table.

The minimal-suffix cover guarantees `nmiss=0 ∧ nuncov=0` over the instance set by construction (each rule is the model's
own answer for the shortest model-deterministic suffix); the *value* is how few rules it takes, and that the expensive
tier (composed) collapses to one. Oracle is the compiled `whole.dl` binary (~140× faster than the interpreter).

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

- ~~close the residual: certify retrieved/selected/structural + a free-choice default~~ — **done**: the full
  `circuits.dl` certifies 300/300. Remaining: widen the proven domain (more instances / exhaustive over the structural
  decision space, not just W=8 windows) and shrink the len-7 tail (a longer window or smarter cover would absorb it).
- the len-1 priors are an honest lookup, not a computation — worth labelling `prior` vs `rule` in the emitted program.
- multi-instance `whole.dl` emission (a `fieldrun` change) would let `equiv.dl` read `ref` in one Datalog run instead of
  the per-instance oracle loop (already ~140× faster via the compiled binary, but still N runs).

## Provenance

Developed in `fieldrun/experiments/whole-datalog` (branch `feat/whole-datalog-poc`) — the lab notebook. The `.dl` files
here (`whole.dl`, `circuit.dl`, `circuits.dl`) are carried over verbatim as the reference translation.
