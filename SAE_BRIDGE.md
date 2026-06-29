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

## Next phase (the bridge proper — not yet built)

- fieldrun exports a **late-layer residual** (layer 8–11) as facts; rosetta applies the SAE encode (numpy: `TopK(W_enc·(x−b_dec)+b_enc)`) → `feat(inst, pos, feature, act)`.
- the cover admits **feature-keyed idioms** under the existing discipline: detect + **causal = ablate the feature → output follows** + certify (TV<ε); cover-ordering feature-idiom > token-idiom > n-gram > backstop.
- caveats: one SAE per layer (pick the resolution layer); SAE is lossy (ε-certified, never exact → backstop keeps "no floor" literal); SAE quality gates the result. pythia-160m is small + NeoX-IOI-weak — a stronger model with per-layer SAEs (Gemma Scope) would sharpen, but the effect is already clear here.
