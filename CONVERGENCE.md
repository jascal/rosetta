# rosetta ⇄ sgiandubh convergence — rosetta builds, sgiandubh serves

*Design doc (Phase 0). Forward-looking claims are tagged `open` until the artifact backs them; this is a plan, not a
measurement. Companion to [`PACKAGE.md`](./PACKAGE.md) (the package schema) — this doc is the **build-ownership**
migration that makes rosetta the sole builder.*

## The one principle

> **sgiandubh builds nothing.** It loads a finished package and serves it — HTTP (OpenAI/Anthropic) + REPL. **rosetta
> builds the package.** The package is the *only* interface between the two repos (the workspace philosophy: coupled by
> a published artifact, never by code or a local path).

Today this is violated: `sgiandubh/tools/` is a complete builder (distill → ground → gram → package → model-free
adapters) living *inside* the runtime repo. That is the "complex hybrid" this migration dissolves. The convergence
already named the destination (rosetta = builder, sgiandubh = thin runtime); this doc makes it **total** — *all*
expert-build work moves to rosetta, and sgiandubh becomes pure server + REPL.

## Scope discipline (the principled part)

rosetta's identity is **provably-faithful model → Datalog minimization** (the cover; `dl/equiv.dl`). Some build steps an
expert needs are emphatically *not* minimization — corpus grounding embeddings, the model-free structured-source
adapter, curated-answer assembly. Moving those into rosetta's *core* would blur the one thing rosetta is. So the move is
into a **delineated layer**, not the core:

- **`rosetta/py/` (core — scope unchanged):** model → certified cover + certificate. The science. `idiom_learn`,
  `temperature`, `minimize`, `abstain_emit`, `oracle`, `dl/*.dl`.
- **`rosetta/py/pack/` (new — depends on core, never the reverse):** assemble a *deployable bounded expert* — cover +
  curated answers + grounding + citations + scope → one package. The *factory*, built **on** the minimization core and
  explicitly separate from it.

**Invariant:** the minimization core must never `import` from `pack/`. `pack/` orchestrates the core plus the
deployment-only concerns. A grep gate (`core never imports pack`) keeps the boundary honest. This is how rosetta absorbs
the whole builder without the minimization science absorbing retrieval/packaging concerns.

## What moves, stays, retires

| piece (today in sgiandubh) | destiny | rationale |
|---|---|---|
| `src/server.cpp` (HTTP + REPL + routing) | **stays** | this *is* the thin server + REPL — sgiandubh's whole remaining job |
| `src/rosetta_package.h` (consumer + `decode_facts`) | **stays** | serving + the C++ semiring decode |
| `tok_ffi/` | **stays** | runtime BPE tokenization |
| serving-side `ground()` / `retrieve_answer()` / `gram.h` | **stays (shrinks)** | runtime *use* of the index; the *build* of it leaves |
| `src/engine.dl` (decode spec) | **→ rosetta core** | it is the semiring-decode *certificate* — a PIC/rosetta artifact, not a runtime file |
| `tools/build_expert(s).sh` (orchestration) | **→ `pack/`** | the expert-build entry point |
| `tools/dl2package.py` (extraction → answers + facts) | **→ `pack/`** | model-extraction packaging (minimization-adjacent) |
| `tools/build_gram.py` (n-gram KB) | **retire** `open` | subsumed by rosetta's gated-n-gram cover — *pending coverage-parity check* |
| `tools/build_grounding.py` (GloVe/PPMI retrieval index) | **→ `pack/`** | retrieval, not minimization → the delineated layer |
| `tools/normrules2package.py` + `riscv_questions.py` | **→ `pack/`** | model-free structured-source adapter; packaging, not minimization |

## The unified package (rosetta emits → sgiandubh serves)

One directory, extends the [`PACKAGE.md`](./PACKAGE.md) schema. sgiandubh serves all of it, builds none of it:

- `manifest.json` — the cover: causal idioms + gated n-grams + provenance (core). **Required.**
- `index.json` + `facts_*/` — curated Q&A answers + per-decision facts for the faithful tier (pack). *Optional* (a
  model-free expert has none).
- `wordvec.txt`, `knowledge.tsv` — grounding embeddings + cited passages for the retrieval tier (pack). *Optional*.
- `bundle.tokenizer.json` — the model's BPE tokenizer (so the runtime tokenizes into the cover's id-space).
- scope / citation metadata.

All gitignored on both sides (reproducible from the build recipe; may carry licensed source content).

## Runtime shape after migration (sgiandubh)

Routing shrinks to a single ordered cascade, all over a loaded package:

```
scope-gate → curated (faithful, if index.json present) → cover (causal idioms → gated n-grams) → retrieval (grounding/cite) → abstain
```

The separate **`gram` tier retires** — the cover already carries gated n-grams with provenance + confidence, strictly
richer than `build_gram`'s bare n-grams. `open`: verify coverage parity (cover n-grams ⊇ build_gram n-grams over a
corpus sample) before deleting `gram.h`.

## Phased plan

- **Phase 0 — this doc.** Anchor the principle + the boundary before any code moves. (PR for review.)
- **Phase 1 — move the builders into `rosetta/py/pack/`** + a `rosetta build-expert` entry that emits the unified
  package from a corpus (+ optional model bundle). Validate by rebuilding *both* example experts via rosetta:
  - logic (model-distilled) — **this is where the truncation fix lands**: rebuilt at an adequate `--steps` so answers
    reach EOS instead of the cap (the current demo packages were built short). `empirical` once rebuilt.
  - riscv (model-free) — via the structured-source adapter.
- **Phase 2 — sgiandubh consumes the unified package**; **delete `sgiandubh/tools/`**, shrink `server.cpp` routing to
  the cascade above, drop the `gram` build dependence. This is "online the rosetta code" made total.
- **Phase 3 — validate both experts end-to-end** against a held-out set per expert (coverage / precision / abstain;
  the [`ABSTAIN.md`](./ABSTAIN.md) frontier), and confirm no confident-wrong regressions. sgiandubh is now pure
  server + REPL.

The earlier standalone asks — *fix the truncated logic answers* and *online rosetta* — stop being sgiandubh tasks; they
become **outcomes** of rosetta becoming the builder (Phase 1 and Phase 2 respectively).

## Open questions (tagged)

- `open` **gram subsumption** — does the cover's gated-n-gram tier fully cover `build_gram`'s n-grams? Measure before
  retiring `gram.h`; if not, fold the gap into the cover emit rather than keeping a second n-gram path.
- `open` **grounding necessity** — retrieval/grounding earns its place by supplying *citation* (the bounded/cited
  property), but its answer quality was mediocre (mean-pooled embeddings have a similarity floor). Measure its marginal
  contribution; it stays a package component regardless, built in `pack/`.
- `open` **curated answers vs cover** — the faithful curated-Q&A tier (full answers) and the cover (next-token) are
  different artifacts; the package carries both, the runtime prefers curated on an exact match. Confirm this ordering is
  what we want per expert.
- `design` **`pack/` boundary** — enforce "core never imports `pack`" with a test/grep gate.

## Non-goals

- Not changing rosetta's minimization science (the core stays exactly as scoped).
- Not retaining *any* build capability in sgiandubh.
- Not introducing a runtime dependency between the repos — coupling stays package-only.
