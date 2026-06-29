# SAE / feature bridge — phase 1: is entity reasoning carried by features tokens can't express? (`empirical`)

The token vocabulary can't compress the residual on real models (llama: 0 idioms, pure n-gram) and can't capture
entity-level families (coreference never confirmed token-level in `CROSS_ARCH.md`). Hypothesis: those live in a
**sparse-feature** basis. Before building the Datalog bridge, test the hypothesis causally.

**Substrate.** pythia-160m + `EleutherAI/sae-pythia-160m-32k` (TopK SAE, k=32, 32768 latents, one residual SAE per
layer). The SAE at `layers.L` hooks `gpt_neox.layers[L]` output = `hidden_states[L+1]`; reconstruction **cos 0.96–0.997 /
FVU ≤ 0.02** confirms a matched model + hookpoint. Driver: `py/sae_bridge.py` (deps in `requirements-sae.txt`).

**Test.** IOI minimal pairs differing only in the **subject** name — clean `ABB` "…A and B… **B** gave a drink to" → A
vs corrupt `ABA` "…A and B… **A** gave a drink to" → B. The **final token is `to` in both**, so the surface token cannot
distinguish the answer; the entity binding (who is the indirect object) is in the residual. Activation-patch corrupt→clean
at each layer/position; metric = logit-diff recovery, `LD = logit(IO) − logit(S)` (n=24). Residual patch = the layer's
causal contribution; **SAE-feature patch** (swap only the SAE-reconstructed part, keep the residual error) = how much the
**sparse basis** carries.

| layer | recon cos | END (`to`, token IDENTICAL) resid / feat | subject pos (token differs) resid / feat |
|---|---|---|---|
| 6  | 0.965 | +35% / +18% | +77% / +69% |
| 7  | 0.974 | +43% / +34% | +54% / +49% |
| 8  | 0.964 | **+95% / +63%** | +19% / +14% |
| 9  | 0.961 | **+95% / +68%** |  +3% /  +2% |
| 10 | 0.964 | **+97% / +73%** |  +0% /  −0% |
| 11 | 0.997 | **+75% / +81%** |  +0% /  +0% |

## Findings

1. **Entity reasoning is feature-carried where the token is useless.** The END token is `to` in clean and corrupt alike,
   so it carries **zero** information about the answer — yet at layers 8–11 the **SAE features at END causally recover
   63–81%** of the entity flip. Tokens can't; features can. The hypothesis holds. (The weak 18% at layer 6 was *purely
   the layer* — before the resolution — not a ceiling.)
2. **The name-mover is visible.** The causal locus migrates from the **subject position** early (77% at L6) to the **END
   position** late (95–97% at L8–10, subject → 0%) — the entity computed at the name, then transported to the readout
   position across layers. This confirms the patches measure real causal flow, not artifact.
3. **Forge tax ≈ 20–35%.** Features recover less than the raw residual (e.g. 63% vs 95% at L8) — the sparse basis loses
   some causal content (near-zero gap at L11, cos 0.997). Consistent with lm-sae's "composition doesn't fully factor
   through features." Not fatal: it sets how much entity reasoning becomes *legible feature rules* vs stays in the backstop.

## Why this justifies the bridge (and how it fits the architecture)

Feature idioms compress the entity reasoning that's legible; the forge-tax residual routes to the **semiring backstop
(whole.dl, exact)** — the same compress-the-compressible / backstop-the-rest pattern as the token cover, with a vocabulary
that reaches entity-level structure. A feature plays the same semiring role as an MLP hidden unit (activation ⊗
decoder-direction ⊕-summed into the residual), so it drops into `whole.dl`'s sum-product graph as a fact.

## Build plan (reviewed & refined — design of record)

Four phases, risk-managed (Phase 1 is non-destructive). The two architectural calls below were settled by external
review: a **hybrid runtime tiering** (keep the token-level artifact pure-souffle; features are a parallel, optional tier)
and a **compositional certificate** (state faithfulness at the SAE-layer interface).

- **Phase 1 — feature export (non-destructive).** `fieldrun --export-residual --layer L` dumps the layer-L residual
  (read-only slice); rosetta applies the SAE encode in numpy (`TopK(W_enc·(x−b_dec)+b_enc)`) → `feat(inst,pos,feature,act)`
  facts. This is a *derived* relation — it never appears in a forward-pass rule, so logits are byte-identical and the
  existing certificate is untouched. Ships first: a lossless feature-labeled view, zero risk to current certificates.
- **Phase 2 — feature ablation (the causal gate).** `fieldrun --inject-residual` runs with `resid' = resid − act·W_dec[f]`
  and resumes → the causal test. A feature-keyed rule is admitted only if ablation moves the output as the rule predicts.
- **Phase 3 — feature-keyed idioms in the cover.** Learn rules keyed on active features (generalize across surface
  tokens), emit + certify (TV<ε), cover-ordering **feature-idiom > token-idiom > n-gram > backstop**.
- **Phase 4 (stretch) — feature→feature transition circuits.** SAEs at multiple layers; `feature(L)→feature(L+k)` rules
  replace the blocks between — the only path to real compute compression (see Decision 4).

### Settled decisions (post-review)

1. **Runtime independence — HYBRID TIERING.** The primary runtime artifact stays the token-level `circuits.dl` (token in →
   distribution out, pure souffle, no model binary). Features are a **parallel, optional `feature_circuits.dl`** that
   assumes `feat(...)` facts are present, produced by the documented companion path (`--export-residual` + numpy SAE
   encode, or a precomputed fact file for a fixed corpus). A pruned Datalog forward-prefix (embed + blocks 0..L + SAE) is
   pursued only for early-layer/tiny-model experiments; for the entity layers (8–11/12) feature idioms are a legibility +
   analysis tier in the near term. This ships certified feature idioms without compromising the token runtime.
2. **Forge tax — accept WITH measurement.** Track an auxiliary metric: *% of predictions on entity-heavy windows resolved
   by a feature idiom vs. falling through to n-gram/backstop*, plus end-to-end TV stays controlled. Justified: a few
   high-quality generalizing circuits move the needle (cf. threx: one idiom → 88% holdout vs 38% n-gram-only at scale).
3. **SAE — keep PLUGGABLE.** Narrow interface: a feature dictionary (id → optional decoder vector / metadata) + encode/decode
   callables (the TopK sparse code already used). Start on the validated EleutherAI pythia-70m/160m SAEs; add Gemma Scope
   and our own lm-sae / sae-forge loaders later. The detect+causal+certify gate is the filter — a bad SAE simply fails to
   produce passing rules; we measure the model *via* its features, not any one SAE.
4. **Late layer ⇒ little compute saved — legibility is a FIRST-CLASS deliverable.** Rosetta's value is certified,
   human-auditable, substrate-transferable transcription, not inference speedup. A late single SAE still buys entity-level
   vocabulary + dropping the unembed (~20% of params) for those rules. Real parameter reduction / early-exit needs Phase 4.
5. **Certificate — COMPOSITIONAL (stated precisely, which strengthens it).** *"`feature_circuits.dl` reproduces the model's
   output within max TV < ε **given the model's own layer-L features**. The prefix (embedding + blocks 0..L + SAE encode) is
   exact by construction — it is the model plus the SAE it was analyzed with. When features are obtained from the model at
   inference/analysis time, the end-to-end system is ε-faithful (exact at T=0 where the rules support it)."* This is
   compositional verification at the natural interface, and it future-proofs Phase 4 (transitions certified at their own boundaries).

### Schema sketch (Datalog)

```prolog
// feature facts — companion path: `fieldrun --export-residual --layer L` | numpy TopK SAE encode
.decl feat(inst:number, pos:number, feature:number, act:float)
.input feat

// feature-GATE idiom (analogue of the token gate): a feature active at the decision position → output distribution
.decl fgate0_tab(f:number, token:number, sc:float)            // feature → top-K logits (incidence), carries the distribution
fgate0_ctxlogit(I,Tk,SC) :- mp(I,P), feat(I,P,F,A), A>thresh, fgate0_tab(F,Tk,SC).
fgate0_any(I) :- fgate0_ctxlogit(I,_,_).

// feature-relative COPY (the entity case token-induction can't express): copy the token at the position where
// the entity-feature F first fired — content-relative pointer in FEATURE space, not token space.
// cover routing (priority via negation guards): feature-idiom > token-idiom > longest n-gram > backstop
ctxlogit(I,Tk,S) :- fgate0_ctxlogit(I,Tk,S).                  // highest tier
// ... token idioms guarded by !fgate*_any(I); n-gram guarded by all idiom _any; then softmax(logits/T) as today.
```

### Success / failure criteria

- **Success (shippable):** ≥1 family of feature-keyed idioms (coreference / entity-copy / name-mover) passes
  detect + causal + certify (TV<ε) on a corpus that exercises it; the idioms generalize across surface tokens (one rule,
  many entity surface forms); a measurable win vs the pure token cover (entity-window coverage ↑ or backstop reliance ↓);
  a clear compositional certificate + feature-frontend docs.
- **Kill signal:** no feature-keyed rule survives the full pipeline on entity-stressing text despite Phase-0 patching
  showing causal signal → the cover's discipline is too strict for current feature quality (a useful negative result).

### Immediate next steps

1. Phase 1 (non-destructive): `fieldrun --export-residual --layer L` (its own branch+PR) + numpy TopK encode → `feat`
   facts; verify byte-identical logits. Land first.
2. Curate an evaluation corpus stressing controlled entity tracking (IOI variants + longer coreference chains where the
   surface tokens are deliberately uninformative).
3. Prototype the Phase-2 causal gate on the high-activation feature sets from the Phase-0 layers (highest leverage).
4. Sketch `dl/` schema + `py/idiom_learn.py` feature-gate / feature-compose primitives (analogues of the token ones).

### Caveats

One SAE per layer (pick the resolution layer); SAE is lossy (ε-certified, never exact → backstop keeps "no floor"
literal); SAE quality gates the result. pythia-160m is small + NeoX-IOI-weak — a stronger model with per-layer SAEs
(Gemma Scope) would sharpen, but the effect is already clear here.

### Ecosystem synergy

Feature idioms discovered here are candidates for encoding as verified quantum circuits in `polygram` (MPS rung-1/2) or as
feature-transition blocks in `n-orca`; `sae-forge` is a natural SAE source for Decision 3. Rosetta is the symbolic,
certified *target*; the sibling tools provide implementation and scaling paths.

*(Plan refined via external review; the five tensions above are the load-bearing design decisions.)*

## Phase 1.5 result — idiom EXTRACTION on pythia-160m: substrate-limited (honest negative)

Before building the fieldrun feature-export plumbing, tested the kill signal directly (`py/feat_idiom.py`): can a
feature-keyed *idiom* (a generalizing rule) be extracted from SAE features and clear the cover's detect + causal gate?

- **IOI (feature-relative copy pointer).** The simplest feature idiom — "copy the name an indirect-object feature marks"
  — does **not** clear the gate: best ~65% vs truth / ~54% vs model / 7% causal (cover needs ≥80%). Phase-0 showed the
  features *carry* the entity info (63–81% patchable), but it is **not cleanly extractable as a generalizing rule** — the
  forge tax / "doesn't factor through features," concretized. And IOI is **token-structural** anyway (the answer is the
  non-duplicated name = a duplicate-token primitive, already token-relative), so it doesn't *need* features.
- **Semantic gender-binding (the genuinely feature-needing case** — answer selected by an in-context gender assignment,
  both names symmetric in position so position/duplication can't pick it). **pythia-160m gets it 21%** (below the 50%
  two-name chance; it mostly predicts punctuation/articles, not names). No capability → nothing to extract.

**Conclusion: the bottleneck is the model, not the bridge or the SAE.** On pythia-160m there is no task that is *both*
genuinely-semantic *and* within the model's ability — it does the token-structural entity tasks (IOI 77%, but those are
token-doable and features don't cleanly extract them) and fails the semantic ones (gender-binding 21%; cf. `CROSS_ARCH.md`:
capital 0%, antonym 8%, coreference 15% at 160m). This is the *earned* trigger for the substrate pivot.

**Next: stand up Gemma Scope** (per-layer SAEs on Gemma-2-2b/9b — a model that actually does semantic entity binding *and*
has SAEs at every layer). Re-run this exact harness there; the kill signal is only a real kill if a feature idiom fails on
a substrate where the phenomenon is present. pythia was necessary to validate that features carry the signal (phase 0) and
to find that simple extraction + small models don't suffice (phase 1.5); Gemma Scope is where the bridge gets a fair test.
