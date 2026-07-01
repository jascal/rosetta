# Status review & plan (2026-07-01) — handoff for Opus

A full-repo review at `master` (d37ad62, post PR #24). Code health: `python3 py/verify_threx.py` → CERTIFIED 25/25
(nmiss=0, nuncov=0); `.venv/bin/pytest tests/` → 34/34 pass; no open PRs; master synced with origin.

This document has two parts: **(A)** an evaluation of where the domain-specific model extraction actually stands on
the models already tried, and **(B)** a prioritized plan. Every claim below carries the program's tag discipline —
nothing here promotes a tag; where a number is corpus-measured it is `empirical`.

---

## A. Evaluation

rosetta has two arms, and the honest one-line verdict differs sharply between them:

1. **Minimization arm** (`whole.dl` → certified `circuits.dl`): the *certification machinery* is real and healthy;
   the *discovered content* on real models is n-gram recall plus a hand-authored circuit catalog — genuine, causally
   confirmed, but narrow-domain and not (yet) discovered from the model unsupervised.
2. **Expert-package arm** (`build_expert.py` → sgiandubh packages): the pipeline that works and measures well is
   **document-derived**; the model-extraction tier is currently **not load-bearing in any shipped expert**.

### A.1 Minimization arm — per-model results

Current canonical artifact (PRs #23/#24): one unified `circuits.dl` per model with a **distributional leg**
(T-parameterized n-gram cover, certified by total-variation ≤ ε=0.02 across T∈[0.7,1.0] over 300 natural W=8 windows)
plus an **argmax leg** (frame-gated structural circuits from the exercise-then-confirm catalog, certified at argmax).

| model | T-leg (max TV) | argmax leg | families with certified instances |
|---|---|---|---|
| threx (reference) | 0.0100 CERTIFIED | 161 rules incl. **1 compose + 2 select LEARNED idioms** | the only model where unsupervised discovery → emit → certify → holdout all close (holdout loss 12%) |
| stories260K/15M/110M | ≤0.0119 CERTIFIED | **none** (n-gram only; never re-run through exercise_confirm) | 0 causal idioms at every scale; holdout loss 47/63/62% |
| pythia70m | 0.0100 CERTIFIED | 111/111 | 6 of 8 (no IOI, no syllogism; succession=1, MP=2) |
| pythia160m | 0.0100 CERTIFIED | 398/398 | all 8 (induction 135 … syllogism 6) |
| llama32_1b | 0.0100 CERTIFIED | 744/744 | all 8 (transitivity 0→100 after #24 frame-gating) |
| qwen25coder15b | ≤0.0100 CERTIFIED | 761/761 | all 8; CROSS_ARCH 12/15 families |

What genuinely works:
- `dl/equiv.dl` / `dl/master.dl` certificates, souffle-only runtime (`run.dl`), facts splitter, resident-serve oracle.
- **Causal confirmation as the universal discriminator** — pruned correlational gates, spurious additive fits, and
  n-gram-masquerading-as-induction, with zero false positives on the stories/threx negative controls.
- Exercise-then-confirm (#23) + frame-gating (#24): converted "0 idioms on real models" from a silent failure into a
  measured **masking phenomenon** — e.g. pythia160m induction causal 2% on its own corpus vs 82–86% on novel-repeat.
- The silent-oracle-failure mode is fixed (`assert_oracle_live`, py/idiom_learn.py) — a real class of wrong-conclusion
  bug eliminated.

The honest limits (all admitted in-repo, confirmed by this review):
- **Unsupervised idiom discovery on real models remains zero.** Every structural circuit in the real-model covers
  comes from the hand-authored 15-template catalog in `py/exercise_confirm.py`, not from `idiom_learn`'s
  select/compose/skeleton miners. Exercise-then-confirm is the right doctrine, but the stimuli are hand-written.
- **Frames are template skeletons.** Emitted circuit rules hard-code the exact template token skeleton
  (e.g. `transitivity_pm` in `models/pythia160m/circuits.dl`); only entity slots generalize. The argmax "N/N" is over
  a domain pre-filtered to instances where the cover already matched — it proves faithful *emission*, not capability.
- **The certified natural domain is tiny** on the 4 real models: `corpus.json` ≈ 1.5k tokens (300 windows) vs 20–24k
  for stories, 84k for threx.
- **Pure n-gram covers anti-scale**: holdout loss 12% (threx) → 47/63/62% (stories ladder); compression 58%→28%.
  ~60% of stories110M behavior is not captured generalizably. This is the research program's central open problem.
- Never-closed items: skeleton family mined but never emitted (`learn_skeleton` feeds only MDL scoring); `additive()`
  still hard-codes threx-shaped arity 3–7; coreference never causally confirmed on any model; IOI-in-NeoX residual
  (~57–60%) undisambiguated between architecture and probe artifact.

### A.2 Expert-package arm — is model extraction useful?

**Measured answer so far: no shipped expert derives value from the model.** `empirical`, and the decisive experiment
has not been run (see B.1):

- **riscv** — distilled with llama-3.2-1b; `idiom_learn` found **0 causal gate/compose idioms** → switched to
  model-free (examples/SOURCES.md). The shipped package is `knowledge.tsv` (4,049 passages) + `strategy.tsv` +
  wordvec; `index.json` empty, **no manifest.json** — retrieval + strategy + abstain only.
- **logic** — the only model-distilled spec (`examples/logic/expert.toml`, gemma-4-e4b-it): 7,045 fieldrun `.dl`
  distill exports sit in `package/_export/`, but **no built index.json, no manifest, no scorecard**. The model-distill
  path is plumbed but has never produced a measured expert.
- **nh-family-law** — template only, not buildable (adapter absent), deliberately model-free.
- **librarian / pedagogy / math (aata+fcla)** — model-free by design; verified by spot-checks only.

Meanwhile the **document-derived pipeline measures well** and is the arm delivering value:
- riscv scorecard: **recall 100% / content-precision 100% / leak 0%** at calibrated `cos 0.70 / margin 0.30`
  (29-row testset — small, directional); off-domain queries abstain; count/list/define routed via `strategy.tsv`
  materialized at build from ergo Datalog.
- Scale: 121-doc arXiv build (16,327 passages) in 13.7s, 284 q/s at 16 clients.
- Known weaknesses the docs admit: holdout leakage (auto holdout = tail of the training corpus), the `[gate]` in
  expert.toml is **silently skipped for model-free builds** (`_score_if_gated`), tiny testsets, abstain erosion at
  121 diverse docs ("capital of France" leaked via a KG paper), GloVe weak for legal language.

### A.3 Verdict

The instrument is honest and the certificates are real, but on the models tried so far **domain-specific model
extraction contributes**: (a) in the minimization arm — a certified n-gram distributional cover plus a hand-templated,
causally-confirmed circuit catalog whose domain is the exercised stimuli; (b) in the expert arm — nothing yet: every
working expert is document-derived. The one model behavior that clearly earns its keep is **pythia160m induction**
(82–86% causal on novel-repeat, +81pp over the n-gram cover) — and it has never been wired into a shipped expert
package. The gap between "we can certify what we extract" and "what we extract is the model's algorithm" is the
whole game; the plan below is ordered around closing it.

---

## B. Plan for Opus (prioritized)

Work through P1→P3 in order; P4/P5 are parallelizable. Keep the repo's rules: every verdict is a souffle query
result; state domains; never promote a tag without the artifact; emitter changes go in a fieldrun branch + PR.

### P1 — Run the decisive "does the model tier earn its place?" experiment (expert arm)

The question the whole expert arm hinges on, and the data is already on disk.

1. **Build the logic expert end-to-end** from `examples/logic/package/_export/` (7,045 distill programs): assemble
   `index.json` + `facts_*/` via `py/pack/answers.py`, author a testset (≥50 rows: answerable / abstain / off-domain),
   run `py/pack/score_retrieval.py`.
2. **Ablate**: score the same testset with (i) curated-answer tier only, (ii) document tier only (Open Logic Project
   prose), (iii) both. The delta of (iii)−(ii) is the measured value of model distillation on a non-lookup domain.
3. **Wire one real cover into one package**: pythia160m induction is the only causally-confirmed real-model idiom —
   emit its manifest (`py/idiom_learn.py --package`) and measure trusted-cover % vs abstain on an
   induction-exercising corpus (threx baseline: 27% trusted / 33% gated / 40% abstain).
4. If both come back ≈0: write the tagged `empirical` verdict in EXPERTS.md that the model tier adds nothing on the
   domains tried (do **not** say "can't" — recipe plateaus, achievability open) and officially re-scope the expert
   arm to document-first with the model tier as a research slot.

### P2 — Widen the certified domain (minimization arm)

The current certificates are correct but narrow; make the domain claim strong enough to mean something.

1. **De-templatize the frames**: generate paraphrase/format-varied stimuli per family (vary carrier phrases, entity
   sets, punctuation, positions), re-run exercise-then-confirm, and measure which frame-gated rules survive. Either
   the frames widen, or the certificate's stated domain shrinks to what is real — both are wins for honesty.
2. **Grow the natural-corpus leg** on the 4 real models from ~300 windows (~1.5k tokens) to ≥10k windows; re-certify
   the T-leg and publish the coverage-vs-abstain curve (ABSTAIN.md machinery exists; idiom-tier gating is not yet in
   the canonical emit — wire it).
3. **Backfill or close stories\***: either run exercise_confirm on stories models (needs tokenizer.json in the dirs)
   or record in their CERTIFICATE.md why the argmax leg is absent (0 causal idioms at every scale is itself a result).

### P3 — Attack the unsupervised-discovery gap (the research crux)

"0 idioms on natural corpora" is now understood as masking, not absence. The next rung is removing the hand-authored
stimulus bottleneck.

1. **Automate stimulus/frame proposal**: mine candidate frames from the corpus (the `learn_skeleton` miner exists but
   is dead code end-to-end — either wire it into `emit_unified_cover` behind causal confirmation, or delete it and
   say so), and/or propose families from capability probes, then let exercise-then-confirm accept/reject.
2. **Generalize the additive miner** past its threx shape (arity hard-coded 3–7, brute-force permutations).
3. **Close the two honest residuals**: coreference (never confirmed anywhere — build a better probe or record it as a
   consistent cross-model failure) and IOI-in-NeoX (~57–60% at 160m *and* 1.4b — disambiguate architecture vs probe).

### P4 — Infra unblocking (fieldrun-branch PRs, per convention)

1. **top-K `/predict` serve endpoint** — removes the `logit_cache.json` build dependency.
2. **Multi-instance whole.dl emit** (`ctx(inst,pos,id) → ref`) — the last piece that makes `dl/master.dl` a true
   weights-in → certified-circuits-out single program.
3. Dense-Gram wall for full-vocab `whole.dl` on real models.

### P5 — Hygiene (small, high-leverage)

1. Enforce (or loudly report) the `[gate]` for model-free builds — today `max_leak` in expert.toml is dead config.
2. Grow the riscv testset well past 29 rows; add labeled testsets for librarian/pedagogy/math (spot-checks → scores).
3. Fix holdout leakage: proper train/holdout isolation in `build.py:_eval_sets` (the EXPERTS.md open item).
4. `md_render.h` is duplicated between sgiandubh and claymore — keep in sync when touching tutor output.

### What NOT to do

- Don't resume the SAE feature-idiom route: it failed the ≥80% causal gate on both substrates after a fair test
  (SAE_BRIDGE.md verdict "not justified on current evidence") — revisit only with a new idea, not a re-run.
- Don't treat argmax-leg N/N counts as capability claims; the domain is the exercised stimuli (IDIOM_LEARNER.md
  domain caveat).
- Don't modify fieldrun from this repo; emitter work is a fieldrun branch + PR.
