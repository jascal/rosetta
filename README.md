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
| `dl/equiv.dl` | **the keystone (live)** — multi-instance equivalence verifier; `certified()` iff `nmiss=0 ∧ nuncov=0` over the domain. Explicit nonempty domain; missing references fail certification. |
| `dl/equiv_dist.dl` | **distributional verifier** — finite-temperature checks against supplied reference logits; computes softmax, TV, coverage, and the verdict in Datalog. |
| `py/certificate.py` | Isolated verifier I/O, retained evidence with hashes, and model-free replay. |
| `py/benchmark_induction.py` | Grouped train/validation/test benchmark; Datalog copy admission, frozen artifacts, train-only n-gram baseline, and exact finite-domain certificates. [Results and replay](docs/induction-benchmark.md). |
| `py/benchmark_guarded_copy.py` | Consensus copy guards with Datalog selection; freezes the firing domain before fresh test references. [Protocol and results](docs/guarded-copy-benchmark.md). |
| `py/benchmark_copy_generalization.py` | Keeps the guard unchanged across fresh seeds, lengths, and distractors; Datalog reports strata and counterexamples. [Protocol and results](docs/copy-generalization.md). |
| `py/diagnose_copy_targets.py` | Controlled target substitutions cross prefix length, sequence length, and gaps to investigate copy failures. [Diagnostic protocol](docs/copy-target-diagnosis.md). |
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
#   → learned idioms + n-gram backfill; finite-grid Datalog checks and replayable evidence
# Add --crisp for exact argmax certification through equiv.dl.

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

**Certificate integrity update:** the historical measurements below are `empirical`. Exact certification now requires
an explicit nonempty domain with complete references. The canonical distributional path uses `dl/equiv_dist.dl` and
retains replayable, hashed evidence. Its proof is restricted to the supplied reference logits at the listed finite
temperature grid; it establishes neither an interval bound nor a bound on omitted model mass. Historical temperature
reports have been relabeled accordingly. See [certificate scope and replay](docs/certificates.md).

The original four-model experiment reported **`nmiss=0 ∧ nuncov=0` in-domain** (T=0 / greedy), via the fieldrun-refs path for
the real models. Same generation budget (250×80, temp 0.8) so the comparison is fair; threx is the capped toy.

These historical loss figures are exploratory **empirical** measurements, not untouched final-test evidence:
the legacy selector uses its evaluation split to choose families. For an isolated final test, see the
[copy/induction benchmark](docs/induction-benchmark.md): the synthetic copy control certifies 24/24;
the Qwen2.5-0.5B-Instruct copy hypothesis matches 22/24 and fails certification.

| model | params | corpus windows | rules | compression | params/rule | **historical split loss** |
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

- **Temperature: one rule set, T parameterized at query time.** The canonical emitter carries top-K logits per rule
  and computes `softmax(logits/T)` in Souffle for positive T. It emits a probability relation; it does not itself sample.
  The crisp emitter handles exact argmax. Learned compose/select idioms can carry distributions; copy/induction remains
  a point-mass fallback. Routing still uses hard priority guards.
  - **proved only over stated inputs:** `dl/equiv_dist.dl` checks TV against the supplied reference logits at the lower
    endpoint, midpoint, and upper endpoint. The certificate retains these exact temperatures and input facts.
  - **empirical:** historical threx experiments reported max TV ≤ 0.016 at T=0.5/0.75/1.0. These three measurements
    establish neither a continuous interval bound nor the correctness of an uncertified symbol rendering.
  - **open:** certified interval bounds, omitted-mass bounds for fieldrun `/topk`, symbol-form certification, and a
    justified weighted mixture of idioms. The current top-K endpoint is available but carries no tail certificate.
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
