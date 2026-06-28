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
| `dl/equiv.dl` | **the keystone (live)** — multi-instance equivalence verifier; `certified()` iff `nmiss=0 ∧ nuncov=0` over the domain. Every emitted `circuits.dl` is proved against the model through this. |
| `dl/{ngram,induction,master}.dl` | **reference / legacy** hand-coded Datalog detectors — *subsumed* by the learner, which discovers these circuits unsupervised and inlines them into the emitted `circuits.dl`. Not on the live path. |
| `dl/primitives.dl` | **design reference** — the ILP primitive vocabulary (`prev_occ`, `at_offset`, `sum_at`, …) the learner composes over (implemented in Python, inlined when emitted). |
| `py/idiom_learn.py` | **the main tool** — unsupervised idiom learning (select / compose / copy-induction, causally confirmed) → `--emit` a runtime-independent `circuits.dl` + `run.dl` → `--certify` via `equiv.dl`. Model-general (CLI flags). |
| `py/probe_induction.py` | measure a model's copy/induction circuit, isolated from the n-gram confound (novel-repeat + causal perturbation). |
| `py/oracle.py` | model-oracle + `souffle` driver. Build-time refs come from `whole.dl` (small/pure), a `fieldrun` bundle, or a **resident `fieldrun --serve` server** (`serve_decide`, big models); runtime stays souffle-only. |
| `py/minimize.py` | the n-gram minimal-suffix cover + emit + certify (the memoization backstop the idioms sit on top of). |
| `py/make_corpus.py` | tokenize real text → `corpus.json` with the model's bundle tokenizer (needs the rosetta `.venv`). |
| `reference/threx/` | **the Rosetta Stone** — a tiny model fully worked: `whole.dl`, a certified `circuits.dl` (now from *learned* idioms), corpus, certificate |
| `models/` | real models (one dir each: `bundle.fieldrun.*` + `corpus.json` + emitted `circuits.dl` + certificate) |
| `tests/` | the certificate must stay clean on the reference model(s) |

## Quickstart

```bash
# learn idioms, emit a souffle-only circuits.dl, and prove it == the model — one tool, CLI-configured:
python3 py/idiom_learn.py 1400 8 reference/threx --emit --certify
#   → 1 compose + 2 select-gate idioms (LEARNED, nothing hand-coded) + n-gram backfill, CERTIFIED nmiss=0 nuncov=0

# a real model (resident server oracle; bundle loads once):
fieldrun --bundle models/<m>/bundle --serve 8177 &        # build-time refs server
FIELDRUN_SERVE=8177 python3 py/idiom_learn.py 1000 8 models/<m> --emit --certify
```

The oracle compiles `whole.dl` to a native binary on first use (~140× faster than the souffle interpreter; needs `g++`);
for bundles it uses the resident `fieldrun --serve` server (`FIELDRUN_SERVE=<port>`).

## Scaling to real models: the fieldrun-refs path + resident server

For real models the build-time oracle is the **`fieldrun` binary**, not `whole.dl` (which is faithful but slow, and its
logic-export is rope-only). A model dir ships a `bundle.fieldrun.*` (`fieldrun convert --model <hf-id> --arch <rope|neox|
gemma4|…> --dtype int8`); the learner reads its argmax via `oracle`. This sidesteps both the dense-Gram wall and the
rope-only export, so the ladder is open to **any architecture fieldrun runs** (Llama/Qwen RoPE, Pythia NeoX, Gemma…).

- **Resident server, not subprocess-per-call.** A naive `fieldrun --bundle … --ids …` reloads the whole bundle every
  call — fatal for big models (a 1B int8 bundle is 1.2 GB). Instead run one **`fieldrun --bundle <stem> --serve <port>`**
  and point the oracle at it with `FIELDRUN_SERVE=<port>` (`oracle.serve_decide` → `POST /predict`). The bundle loads
  **once**; measured **0.11 s vs 1.78 s/call (16×)** on Llama-3.2-1B. The server is single-threaded but each forward uses
  all cores, so one resident server + internally-parallel forward is the right shape (no RAM blow-up, no oversubscription).
- **Runtime independence is preserved.** fieldrun is *build-time only* (computing the refs the circuits are certified
  against). The emitted `circuits.dl` runs in **souffle alone** — no fieldrun, no weights — via `run.dl`.

### whole.dl in parts (the purity path, the dense-Gram wall)

The `whole.dl` route — the model's forward pass *as Datalog* — remains the showcase/purity path for small models. A
whole.dl is ~99.96% **weights-as-facts** and ~0.04% rules (stories260K: 261,092 fact lines vs 116 rules). Two
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

Next: more idiom **families** to attack that holdout gap. Detection is now **learned, not hand-coded** — `py/idiom_learn.py`
discovers idioms unsupervised and causally confirms them (select gate, compose, copy/induction so far); the open families
are agreement / delimiter-bracket / coreference. `proved`/`empirical`/`open` tags gate every claim.

## Frontiers (revisit later)

- **Temperature: one rule set, T parameterized at query time.** Today's `circuits.dl` is the **T=0 (greedy)** corner —
  each context → its **argmax**, exact, rules with no weights. The goal is *not* a separate program per temperature but
  **one set of rules carrying the logits as incidence values**, with **T applied at query time** as `softmax(logits/T)`.
  This works because logits are **T-invariant** (T only scales them in the softmax), so a single export serves every
  temperature; `run.dl` softmax-samples the kept logits at any T, making `circuits.dl` a faithful **sampler**, not just a
  greedy predictor. It is a **semiring lift**: T=0 is the boolean/tropical (argmax) collapse; T>0 is the probability
  (sum-product) semiring where the incidence weights reappear. Each idiom keeps its *structure* but emits a distribution,
  and its incidence value **measures circuit sharpness** (induction strength = the copy probability). The same lift turns
  the causal test binary→graded (mass shifts, not argmax flips) and the certificate exact→distributional (a TV/KL bound,
  the rank-1 shortlist certificate generalized). **Built + certified on threx** (`py/temperature.py`, `oracle.logits`):
  each rule carries top-K (token, logit); the runtime computes `softmax(logits/T)` in souffle at a queried `.input temp`;
  `circuits.t.dl` reproduces the model's full distribution within ε across a temperature *range*. threx T∈[0.5,1.0] ε=0.02:
  173 rules (top-K mean 3.8 — only 6 more than the 167-rule T=0 cover), CERTIFIED at T=0.5/0.75/1.0 (max TV ≤ 0.016).
  - **A T-cover spans a [T_min, T_max] range, not a point — two opposing error sources.** Top-K *truncation* is worst at
    the **hot** end (the tail fattens with T → size K at `T_max`); but *group consistency* (one representative per suffix)
    is worst at the **cold** end (low T amplifies within-group logit gaps → check grouping at `T_min`). `T_max` still trades
    fidelity-range against compression (`T=0` most compressible; `T→∞` ≈ the full unembed). *Done for the n-gram cover via
    whole.dl; remaining: idioms carrying distributions too, and top-K logits from fieldrun for big models (serve `/predict`
    returns argmax only — needs a top-K endpoint).*
  - **The routing *is* the T=0 shadow of this.** The cover-ordering priority in `circuits.dl` (compose > select-gate >
    longest n-gram > copy/induction fallback), encoded as hard negation guards, is exactly the **argmax-override collapse**
    of an incidence-weighted mixture: at T>0 the rules don't strictly override — each contributes logit-weighted mass and
    the model's distribution is their (semiring) sum; the priority order is just *which contribution has the max logit*.
    Bonus: induction's hard "OOD fallback" gating dissolves into a graded copy-mass contribution at T>0 (the incidence
    weight handles it, no special-casing). *The n-gram T-cover is built; folding the idioms into the weighted mixture is next.*
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
