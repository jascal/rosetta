# rosetta

**Minimize a whole LLM into Datalog, provably faithfully.** rosetta takes the faithful Datalog transcription of a
model's forward pass (the *whole-model program*, emitted by [`fieldrun`](../fieldrun) `export --logic-whole`), and
reduces it to a small set of named, human-readable **circuits** — each one *certified equivalent to the model* by a
Datalog query, not by trust. The implementation is **Datalog first**: the rewrite rules, the equivalence proofs, and the
causal probes are `.dl` programs; Python only stages inputs and drives `souffle`.

The name is literal: a model and its logic are two languages for the same computation, and rosetta is the stone that
carries the same text in both — with a proof that the translation is exact.

## The thesis

> A trained model *is* an **algorithm**, and an algorithm is **substrate-free**. rosetta extracts that algorithm into a
> certified, legible Datalog form you can **re-run, rebuild, and re-learn in any substrate — and prove it's the same**.
> The weights ("thingies that do stuff") are one realizer; the algorithm is the invariant. We already demonstrate one
> substrate transfer: `circuits.dl` runs the model's behavior in **souffle alone — pure logic, no weights, no GPU** —
> with a Datalog certificate that it computes the same function.

The corollary that drives the research: capturing the algorithm needs the *computation*, not just the recall. An n-gram
rule transfers a lookup table; an **idiom** (the composed `i+j`, an induction/copy circuit) transfers a *generalizing*
circuit — the way the model itself generalizes. So **holdout generalization is the "how much of the algorithm did we
actually capture" metric**, and better idiom detection is what drives it toward a faithful, substrate-portable whole.
Two models with different weights that extract to the *same* `circuits.dl` are the same algorithm — an algorithmic
identity test, regardless of mechanism.

This is the minimization arm of the PIC **certified-compression loop** (`i-orca` verifies · `fieldrun` analyzes ·
`pil` learns · **rosetta minimizes**). See [`AGENTS.md`](./AGENTS.md) for how it wires in, and the tag discipline
(`proved` / `empirical` / `open`) every claim here carries.

## The pipeline (each stage Datalog-checked)

```
whole.dl  ──mine──▶  deterministic n-gram circuits      (retrieved / selected / structural)   grammar-blind
   │      ──localize▶ causal operand discovery           (ablate whole.dl, find what moves)     dl/ablate.dl
   │      ──discover▶ structure search over operands     (additive / copy / max … then certify)
   │      ──certify─▶ EXHAUSTIVE equivalence, in Datalog (dl/equiv.dl: nmiss=0 ∧ nuncov=0)      the certificate
   ▼
circuits.dl  +  CERTIFICATE.md         the minimized model, and the proof it equals the original
```

Two principles learned the hard way (in the fieldrun threx experiment that seeded this repo):

1. **The oracle is the faithful program, never the binary.** Equivalence is checked against `whole.dl` (faithful by
   construction), so verification is Datalog-vs-Datalog and needs no GPU or model runtime.
2. **Exhaustive beats sampled, and wildcards must be typed.** A sampled certificate is optimistic; `equiv.dl` checks
   *every* instance in the supplied domain. A free wildcard that looked sound under sampling failed at 23/31 — the
   honest rule was a *typed* wildcard. The certificate is the result of a Datalog query, so it cannot lie by omission.

## What's here

| path | role |
|------|------|
| `dl/equiv.dl` | **the keystone** — multi-instance equivalence verifier; `certified()` iff `nmiss=0 ∧ nuncov=0` over the domain |
| `dl/` | the Datalog implementation: equivalence, causal ablation, circuit routing |
| `py/oracle.py` | thin `souffle` driver — runs `whole.dl`, stages the `ref`/`tok` facts, returns the Datalog verdict |
| `py/minimize.py` | model-general: build a certified circuits-only program (composed plugin + minimal-suffix cover) |
| `py/split_facts.py` | split inline-fact `whole.dl` → tiny `forward.dl` + `weights/*.facts` data modules (~100× faster) |
| `reference/threx/` | **the Rosetta Stone** — a tiny model fully worked: `whole.dl`, a certified `circuit.dl`, corpus, certificate |
| `models/` | where real models get minimized (one dir each: `whole.dl` + corpus + discovered circuits + certificate) |
| `tests/` | the certificate must stay clean on the reference model(s) |

## Quickstart

```bash
python3 py/verify_threx.py     # certify the threx composed circuit (25 cells) — verdict from dl/equiv.dl
python3 py/minimize.py 300 8   # minimize + FULLY certify threx: composed + minimal-suffix cover, all 300 windows
```

Expected: composed `25 · 0 · 0 · CERTIFIED`; full program `ncover=300 nmiss=0 nuncov=0 · FULLY CERTIFIED`.
The oracle compiles `whole.dl` to a native binary on first use (~140× faster than the souffle interpreter; needs `g++`).

## Scaling: whole.dl in parts (the dense-Gram wall)

A whole.dl is ~99.96% **weights-as-facts** and ~0.04% rules (stories260K: 261,092 fact lines vs 116 rules). Two
consequences, and the fixes:

- **facts-as-data, not facts-as-code** (`py/split_facts.py`): inline facts make souffle re-parse the weights every call
  and make `souffle -c` inline them into a giant `.cpp` (a 261k-fact model → a 106 MB `.cpp` that g++ chokes on). The
  splitter rewrites whole.dl into a tiny `forward.dl` (the rules + an `.input` per weight relation) plus
  `weights/<relation>.facts` data modules. souffle then bulk-loads weights as data and compiles only the rules.
  Measured: **~0.4 s/call vs ~30–60 s** for the 261k-line inline form. The oracle does this automatically.
- **the dense-Gram wall** — embed/unembed are `vocab × d` facts; `emit_whole` refuses above ~4M. Decompositions that
  *preserve* faithfulness (each carries a Datalog-checkable bound): **corpus-restricted embed** (emit only rows for
  tokens the corpus uses — exact for a fixed corpus, the biggest win for minimization); **tiled unembed with per-block
  rank-1 certificates** (a block whose best-possible logit can't beat the leader is provably elided — `--shortlist` is
  the 1-block case); **low-rank `U≈A·B` with a certified residual bound**; **hierarchical coarse-to-fine argmax**. These
  are emitter changes (a `fieldrun` branch + PR) and are what unlock full-vocab real models.

## License

[Apache License 2.0](./LICENSE).

## Status — the T=0 ladder so far

Four models minimized and **certified `nmiss=0 ∧ nuncov=0` in-domain** (T=0 / greedy), via the fieldrun-refs path for
the real models. Same generation budget (250×80, temp 0.8) so the comparison is fair; threx is the capped toy.

| model | params | corpus windows | rules | compression | params/rule | **holdout loss** |
|-------|------:|------:|------:|----:|----:|----:|
| threx (Threxian) | 21,632 | 360 | 151 | 58% | 143 | **12%** |
| stories260K | 260K | 13,911 | 7,921 | 43% | 33 | **47%** |
| stories15M | 15.2M | 15,264 | 10,890 | 29% | 1,395 | **63%** |
| stories110M | 110M | 18,124 | 12,959 | 28% | 8,452 | **62%** |

Three curves, all pointing the same way: as capacity grows, **compression drops** (less idiomatic), **params/rule
explodes** (more capacity beyond the recall skeleton), and **holdout generalization loss rises** (a pure n-gram cover
generalizes *worse* on bigger, more diverse models). The standout is **threx vs stories110M**: threx, which has one real
*computed* idiom (`THINGS[i+j]`), generalizes at **88%** (12% loss); the 110M model, captured as n-grams only,
generalizes at **38%** (62% loss). **The idiom is what generalizes** — direct evidence that closing holdout loss = better
idiom detection = a more substrate-transferable algorithm. That gap (the ~60% the n-gram cover can't generalize) is the
idiom research program, now with a score.

Next: the non-n-gram detector library (induction is in; agreement / delimiter / coreference next) to attack that
holdout gap. `proved`/`empirical`/`open` tags gate every claim.

## Frontiers (revisit later)

- **Temperature-parameterized completeness.** Today's `circuits.dl` is the **T=0 (greedy)** corner: each context → its
  **argmax** (top-1), exact. "Complete for this corpus at temperatures {T₁…T_N}" reduces to **complete up to `T_max`**
  (the softmax tail only grows with T, so the hottest point dominates): make each rule a **top-K distribution** (token +
  logit, from fieldrun's `logit` scoreboard) with **K chosen so the elided tail's softmax mass is provably < ε at
  `T_max`** — the rank-1 shortlist certificate generalized from argmax-equality to a *distributional* bound (TV/KL). The
  `run.dl` harness then softmax-samples the kept logits at any supported T, making `circuits.dl` a faithful **sampler**,
  not just a greedy predictor. Spectrum: `T=0` → argmax (exact, most compressible); `T≤T_max` → top-K + logits
  (distributional certificate, larger K, less compressible); `T→∞` → ≈ the full unembed (minimization buys nothing). So
  `T_max` is the knob trading fidelity-range against compression. *Not built yet — slot after the detector library.*
- **Non-n-gram circuit detectors** beyond `ngram.dl`/`induction.dl` — agreement, delimiter/bracket-matching, coreference
  — to capture the long-order tail that recall can't. params/rule grows with model size precisely because that tail does.
- **Runtime input ergonomics**: a JSON / quoted-CSV input adapter so `circuits.symbols.dl` runs on contexts containing
  control-char tokens (tab/newline); `<0xNN>` rendering for byte-fallback tokens in the lexicon.
- **The learning-curriculum (time axis).** Run rosetta over a model's **training checkpoints**. The conjecture — *learn
  n-grams first, then progressively more abstract/tight circuits on top* — makes concrete predictions: early steps →
  low effective order, behavior ≈ the n-gram cover, **high** holdout loss (memorizing); later → idiom structure appears,
  holdout loss **drops**, the induction detector starts firing — with the induction *phase transition* showing as a
  sudden holdout-loss drop at a specific step. rosetta + the detectors are the instrument to *watch* the curriculum, with
  certificates. (Size axis = this Status ladder; temperature axis = above; this is the time axis.)
- **Substrate transfer / re-learning.** Because the extracted algorithm is certified and substrate-free, it can be
  re-instantiated *or re-trained* in another substrate (silicon, a rule engine, a smaller/native model) and **re-certified
  against the same spec** (`equiv.dl`) — verifiable distillation against the *algorithm*, not black-box input→output
  matching. Precondition: low holdout loss (the real algorithm captured, not just a lookup table). This is the link to
  the rest of the PIC program (`fieldrun` circuit-identity, `sae-forge` re-forging, `pil` learning).
