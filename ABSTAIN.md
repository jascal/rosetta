# Abstaining rules — the bounded-expert cover (`empirical`)

## The problem

The semiring backstop catches **abstentions** (no rule fires) exactly, but not **mispredictions** (a rule fired but is
wrong on an unseen context) — n-grams can't self-limit, they fire on any matching suffix. So a naive cover mispredicts on
holdout and the backstop can't save it (see `py/complete_cover.py`). Fix: rules **abstain where unreliable** and defer to
the backstop → the complete artifact's loss → the residual confident-but-wrong rate (OOD-near-exact).

## Design (informed by sgiandubh, the partial-backstop runtime)

The look at `sgiandubh` (a shipped bounded expert that abstains by construction) sharpened the design:

1. **Carry per-rule confidence, set at build, gated at runtime** (sgiandubh ships a per-item `margin`). Here each rule
   carries **support** (train occurrences) + **determinism** (majority fraction); the carried T>0 distribution also gives
   a logit **margin** for free. A rule fires iff `support ≥ k AND determinism ≥ d` (flags — the expose-thresholds convention).
2. **Gate every tier, including the most-trusted one.** sgiandubh's bug is its *faithful* tier is ungated — "matched" ≠
   "this decision is reliable." So abstention isn't an n-gram-tail afterthought; the idiom tier needs a gate too.
3. **The reliable unit is the causal idiom (≈ sgiandubh's curated item), not the raw suffix.** Idioms are causally
   confirmed → they generalize → the TRUSTED, ~ungated tier; raw n-grams are the gated tail. Tiering: **idiom > gated n-gram > abstain**.
4. **The reject option** = `softmax over { top-K, ABSTAIN }` (ABSTAIN = the semiring ⊥). In the emitted Datalog this is
   simply: a confident rule emits `cdecide`; no confident rule → **no `cdecide` = abstain** → backstop (or refuse).

## Result

`py/abstain_emit.py` emits an abstaining `circuits.dl` (confident rules only) and verifies it in souffle. On stories110M
(corpus next-token = real suffix ambiguity, 17k train / 7.3k holdout, W=8), the **bounded-expert scorecard** —
a reject-option classifier, *coverage / precision / abstain* — as the two confidence flags sweep:

| support | determinism | coverage | precision | abstain | complete-loss (if exact backstop) |
|---|---|---|---|---|---|
| ≥1 | ≥0.0 *(naive)* | 97% | 49% | 3% | 49.8% |
| ≥1 | ≥1.0 | 43% | 62% | 57% | 16.4% |
| ≥3 | ≥0.8 | 32% | 89% | 68% | 3.5% |
| ≥3 | ≥1.0 | 16% | 93% | 84% | 1.2% |
| ≥5 | ≥1.0 | 12% | 96% | 88% | 0.5% |
| ≥10 | ≥1.0 | 9% | 98% | 91% | 0.2% |

The emitted `circuits.abstain.dl` (1315 confident rules at supp≥3/det≥1.0) runs in souffle and matches the Python cover
**exactly** (0 mismatch / 400 sample; 336 abstained in souffle — it really refuses). Earlier `py/abstain_cover.py` showed
val-calibration reaches ~3% complete-loss at ~75% abstain; carried support+determinism (no val split, build-time only)
reaches ~1% at ~84% abstain — and emits to a runnable artifact.

## Two regimes, one artifact

- **Exact backstop (rosetta, `whole.dl`):** abstain → exact recompute. The cover *compresses* the confident fraction;
  the backstop handles the rest exactly. "complete-loss" → the confident-but-wrong residual (~1%). Abstention is a
  compute/accuracy knob.
- **Partial / absent backstop (sgiandubh bounded expert):** abstain → honest **refusal**. The same artifact is now a
  high-precision expert that answers where reliable and refuses elsewhere. Abstention is the **safety boundary**, not a
  knob — a miscalibrated rule *hallucinates* (no model to catch it). The precision/coverage frontier *is* the deployment
  dial. Rosetta's `circuits.t.dl` (contrib + softmax-at-T) ≈ sgiandubh's package (facts + contrib + logprobs); they're
  converging on the same artifact + the same reject option.

## Honest notes / next

- On diverse text only ~15–30% of contexts have a reliably-generalizing short rule; the rest genuinely need the
  model/backstop. The frontier makes that *explicit and exact*, instead of guessed-wrong (naive 49% precision).
- This demo is **n-gram only** (stories has no causal idioms); on a model with idioms (threx/llama) the idiom tier is the
  trusted, ungated unit and lifts the confident-coverage. Wiring the gate into the canonical `temperature.py`/`idiom_learn`
  emit (and an explicit `ABSTAIN` softmax element / per-decision margin gate) is the remaining integration.
