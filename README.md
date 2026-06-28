# rosetta

**Minimize a whole LLM into Datalog, provably faithfully.** rosetta takes the faithful Datalog transcription of a
model's forward pass (the *whole-model program*, emitted by [`fieldrun`](../fieldrun) `export --logic-whole`), and
reduces it to a small set of named, human-readable **circuits** — each one *certified equivalent to the model* by a
Datalog query, not by trust. The implementation is **Datalog first**: the rewrite rules, the equivalence proofs, and the
causal probes are `.dl` programs; Python only stages inputs and drives `souffle`.

The name is literal: a model and its logic are two languages for the same computation, and rosetta is the stone that
carries the same text in both — with a proof that the translation is exact.

## The thesis

> A trained model *is* a Datalog program. If we can identify every circuit it computes — each as a rewrite rule
> certified against the faithful whole-model program — then the weights are redundant and can be dropped. The minimized
> Datalog program is the model, made legible.

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

## Status

The threx Rosetta Stone is **fully validated**: the complete `circuits.dl` (1 composed rule + 121 minimal-suffix rules)
is **certified `nmiss=0 ∧ nuncov=0` over 300 decision windows** in Datalog — 122 rules for 300 decisions, a few hundred
lines vs `whole.dl`'s 22,222. The breakdown is honest (`reference/threx/CERTIFICATE.md`): the composed *computation*
collapses to one rule, free-choice *priors* are len-1 lookups, the rest are selected/structural transitions.

Next is **breadth**: minimize *tens* of small models to confirm the rewrite-rule / idiom library is complete and general
— the Rosetta Stone proves the method on one; confidence comes from many. `proved`/`empirical`/`open` tags gate every
claim.

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
