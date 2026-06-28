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

## Status — BOTH idiom families learned unsupervised on threx (`py/idiom_learn.py`)

All three threx tiers are now rediscovered from behavior alone, nothing hand-coded, in one tool:
- **retrieved / select** — frame-conditioned single-slot GATEs. The place gate exactly: `GATE@4, frame {wø@1, ·@3, ⟨@5,
  ⟩@6}, table {hï→fï, fa→bo, dø→sto}`, `who@2` correctly **ignored**, zero faithfulness violations, **causal 100%**.
- **compose** — 2-operand arithmetic. `THINGS[i+j]` exactly: `COMPOSE @4+@5, frame {gɪ,·,·,∿,⟨,⟩}`, recovered strengths
  `↑↗→↘↓ = 0..4` (the hidden labeling), `sum→thing` table, **causal 100%** over the full operand grid.

The ideas that made the GATE (select) learner work — each fixed a failed simpler attempt:
1. **Anchored harvesting, not global greedy.** Global purity-greedy chases the largest pure region and never grows a rare
   idiom's frame. Anchoring on each instance and growing with *that instance's own values* surfaces every local idiom.
2. **Separate table-discovery from frame-invention.** Greedy overfits the frame to a co-occurring content word
   (`@7=lum`). So keep only the *table*, EXPAND it to its full support, and DERIVE the frame as the offsets CONSTANT
   across that support — the true structural frame; varying offsets are the ignored slots.
3. **Observational mine, causal confirm.** Frame-conditioned gates are an observational pattern — mine with **zero**
   perturbations; spend `decide()` only to causally confirm the top candidates (perturb the slot; output must follow the
   table). Bounds cost AND is the discriminator that prunes correlational/leaky gates.

The COMPOSE learner is the same shape lifted to **pairs** of slots, plus one access-pattern fix:
4. **Frame-first, not anchored, for compose.** Compose frames are rare (~2% of decisions), so uniform anchor-sampling
   misses them. Instead iterate single-guard frame *regions* directly (cheap, exhaustive), and within each search operand
   PAIRS for a clean additive 2D table where *neither slot alone* determines the output (that's what makes it compose,
   not two gates). Additive structure is `discover.py`'s permutation-labeling search; over-generation (sparse 2D tables
   admit some labeling) is killed by causal confirmation over the full operand grid (threx: 1 real of 14 mined).

`select` = one causal operand (lookup); `compose` = two operands (computation) — told apart by how many slots are
causally load-bearing. **Next:** run both up the stories ladder (idioms unknown there); add the additive solver beyond
brute-force permutations for large operand alphabets; wire confirmed idioms into the cover (`minimize.py` + `equiv.dl`)
to measure the holdout-loss drop.

## Status — third family: COPY/INDUCTION, learned unsupervised on pythia-160m

Moving up size + changing domain is what surfaces NEW families (the whole point). TinyStories (s260K/15M/110M) turned out
to be **pure n-gram recall at every scale** — 0 causally-confirmed select/compose, and copy/induction is causally **absent**
(the decisive test: perturb the earlier occurrence of the output token → output follows only **2%**; the high observational
"output appears earlier" rate, up to 87% at large W, is coincidental recurrent vocabulary). The learner correctly finds
nothing (no false positives — validated against threx). Two lessons banked: (a) the **window must = full context** (W=8 was
a threx holdover that forces n-gram behavior); (b) a near-constant gate table is a false-positive mode (s110M's "Once upon
a time" boilerplate) — fixed with a **discrimination filter** (the slot must genuinely select: majority output < 70%).

So we went to a model KNOWN to have the circuit: **pythia-160m** (induction heads). The COPY/INDUCTION family is
content-relative (not offset-relative): `output = ctx[ prev_occ(last-L suffix) + L ]` (the `prev_occ` primitive). Learned
unsupervised + causally confirmed: **L=1/2/3 causal 84-90%** (`py/probe_induction.py`, `learn_relational`). The decisive
methodological result — **observational induction is confounded by n-gram determinism**, and only the causal test
disentangles them:

| corpus | observational | **causal** | verdict |
|---|---|---|---|
| pythia greedy natural text | 59-91% | **11%** | n-gram determinism (a recurring suffix → same next for n-gram reasons) |
| pythia repeated NOVEL tokens | 79-84% | **85-90%** | **true induction** (no n-gram to lean on; only copy can predict) |
| threx (negative control) | 12-38% | **1-2%** | no copy circuit (correctly) |

The **causal test is the universal discriminator** across all three families — it pruned correlational gates, spurious
additive fits, AND n-gram-masquerading-as-induction. Probe novel-repeat to isolate the circuit; perturb to prove it.

**select** (one operand → lookup) and **compose** (two operands → computation) are *offset-relative*; **copy/induction**
is *content-relative* — the two axes of idiom. Next families likely live on these axes too (coreference, delimiter/bracket
matching = content-relative; agreement = offset- or content-relative constraint).

## n-grams are memoized rules (cover-ordering principle, user)

For the whole-`circuits.dl` goal: **learn the generalizing idioms FIRST (unsupervised), then backfill n-gram determinism
LAST** — never the reverse. An n-gram is the *compiled cache* of a rule the model actually learned; if you seed the cover
with n-grams before exhausting idiom learning you can't tell which n-grams are just memoized instances of a rule you should
have captured generally. So: idioms = the algorithm (generalizing, drops holdout loss); the n-gram suffix-cover = the
memoization backstop for the residual + runtime efficiency, added only after the unsupervised learners have said what's
genuinely irreducible. (The minimal-suffix cover in `minimize.py` is already that backstop; integration = idioms-then-ngrams.)

## Payoff

A **learned, certified, per-model idiom library** — and across models, the idioms that *recur* are the universal circuits
(the substrate-free algorithm); the ones that don't are model-specific. Holdout loss → 0 means the real algorithm is
captured (substrate-transferable), not just a lookup table. `proved`/`empirical`/`open` tags gate every learned rule.
