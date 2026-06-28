# Idiom learner — idiom detection as an ML/search problem (design)

## Why (the reframe)

Hand-coding idiom detectors (`dl/ngram.dl`, `dl/induction.dl`, the threx composed rule, then *guessing* agreement /
delimiter / coreference …) is **a priori enumeration of a space we cannot enumerate.** The models have circuits we'd
never think to name — they're exactly the ~60% holdout-generalization gap on the real-model ladder. To *capture it all*,
idiom detection must become **learned discovery**: find compact, *generalizing* rules from the model's behavior, not
match hand-written templates.

This is **program synthesis / inductive logic programming (ILP) / symbolic regression**, and rosetta is already the
harness for it because the two hard pieces exist:

- **Objective** = `holdout generalization loss + rule count (MDL)`. Minimize description length subject to faithfulness,
  scored by generalization. (Both already computed by `py/minimize.py`.)
- **Verifier** = `dl/equiv.dl` (faithfulness) + the holdout split (generalization). So the **generator can be untrusted /
  learned** — propose freely; the certificate makes it sound. *You don't need a correct idiom-finder, you need a
  productive one + a sound checker.* We have the checker.

The loop: **propose → certify (`equiv.dl`) → score (holdout-loss drop + compression) → admit → iterate** on the residual
until the loss stops falling. The hand-coded detectors become *seeds/priors*, not the library.

## Generator order (decided)

1. **ILP / anti-unification** *(build first)* — least-general-generalization over the memorized residual using the
   relational-primitive library below. Deterministic, reproducible, no LLM. Limited to the primitives it's given.
2. **Enumerative synthesis over a grammar** *(then)* — guided search over {primitives × operations}, certify each.
   Generalizes the threx composed-discovery (`py/discover.py`: localize operands → search operation → certify).
3. **LLM/agent proposes** *(last resort)* — only if 1–2 stall; neuro-symbolic, certificate-gated.

## Hypothesis space — from ergo (the a priori catalog)

The a priori guesses live in the (uncommitted) **`ergo`** module. `ergo/PROBES.md` is the family catalog; `ergo/core.dl`
is a ready library of domain-general inference rules. They map to the learner as the **primitive vocabulary the ILP
search composes** (NOT as hand-coded detectors):

| ergo family (PROBES.md) | relational primitive (core.dl / new) | idiom it would let ILP learn |
|---|---|---|
| structural recursion / is-a | `subsumes`/`extends` transitivity, inheritance | inheritance / type-propagation rules |
| conditional / MP / MT | `implies` (+ transitivity), `neg` modus tollens | implication chains |
| workflow / state-tracking | `num` + arithmetic (Soufflé functors) | state-update (the threx composed `i+j` is this) |
| temporal / spatial | `greater` transitivity, relational composition | ordering / composition rules |
| analogy / structure-mapping | `eq` congruence, structure transport | relation-transfer rules |
| set / quantifier | set-relation predicates (new) | membership / region rules |
| (rosetta-native) | suffix-match (`ngram.dl`), previous-occurrence (`induction.dl`) | n-gram recall, induction/copy |

So: **`core.dl` rules + the two rosetta detectors = the seed primitive set**; ILP learns which *combinations* fit each
model's residual behavior, certified.

## The build

`py/idiom_learn.py` + `dl/primitives.dl`:
1. **residual** = contexts the n-gram cover only *memorizes* (long-suffix / holdout-failing) + the model's answers.
2. **anti-unify** matching residual cases over the primitive library → a variabilized candidate rule (generalizing, not
   constant).
3. **certify** with `equiv.dl`; **score** by holdout-loss drop + compression; **admit** if it generalizes *and* shrinks
   the program.
4. **iterate** on the remaining residual until holdout-loss stops dropping (loop-until-dry).

## Validation target — threx

The learner must **rediscover threx's idioms from behavior alone**, using the primitives, with *nothing hand-coded*: the
composed `THINGS[i+j]` (via the arithmetic/state primitive — `py/discover.py` already does this hand-run), the place
gate (via a keyed-lookup primitive), and copy/echo (via previous-occurrence). If ILP recovers those + certifies them +
drops threx's holdout loss toward 0, the approach is proven; then scale to the stories ladder, where the unknown circuits
live.

## Status — unsupervised gate learning SOLVED on threx (`py/idiom_learn.py`)

The 'select' family (frame-conditioned single-slot gates) is learned **fully unsupervised** — nothing about the grammar
given — and the place gate is rediscovered exactly: `GATE@4, frame {wø@1, ·@3, ⟨@5, ⟩@6}, table {hï→fï, fa→bo, dø→sto}`,
with `who@2` correctly identified as **ignored**, zero faithfulness violations, **causal 100%**. The three ideas that
made it work (each was a failed simpler attempt first):

1. **Anchored harvesting, not global greedy.** Global purity-greedy chases the largest pure region and never grows a rare
   idiom's frame. Anchoring on each instance and growing with *that instance's own values* surfaces every local idiom.
2. **Separate table-discovery from frame-invention.** Greedy overfits the frame to a co-occurring content word
   (`@7=lum`). So keep only the *table*, EXPAND it to its full support, and DERIVE the frame as the offsets CONSTANT
   across that support — the true structural frame; varying offsets are the ignored slots. (This is the data-derived
   frame validated earlier, but the support class is now discovered, not seeded.)
3. **Observational mine, causal confirm.** Frame-conditioned gates are an observational pattern — mine them with **zero**
   perturbations, then spend `decide()` only to causally confirm the top candidates (perturb the slot; output must follow
   the table). This both bounds cost and is the discriminator that prunes correlational/leaky gates.

`select` (one causal operand → lookup) and `compose` (≥2 operands → computation, e.g. `THINGS[i+j]`) are distinct
families: the gate learner correctly captures the former and leaves the latter to the arith template (`py/discover.py`).
**Next:** the arith/compose learner as a second unsupervised template, then run both up the stories ladder (idioms unknown).

## Payoff

A **learned, certified, per-model idiom library** — and across models, the idioms that *recur* are the universal circuits
(the substrate-free algorithm); the ones that don't are model-specific. Holdout loss → 0 means the real algorithm is
captured (substrate-transferable), not just a lookup table. `proved`/`empirical`/`open` tags gate every learned rule.
