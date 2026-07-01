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

## Exercise-then-Confirm — validating idioms on REAL models (`py/exercise_confirm.py`)

The learner only ever produced a *surviving* idiom on synthetic threx. Two threats to validity explained the real-model
zeros: (1) a silently-dead causal oracle scores every idiom `causal=0` — a FALSE "no idioms" indistinguishable from a
genuinely n-gram model (now blocked by `assert_oracle_live`, which aborts unless the live oracle reproduces known refs);
(2) a model's **natural corpus MASKS circuits it demonstrably has** — pythia-160m induction is causal **2%** on its own
corpus but **84–90%** on novel-repeat stimuli (same model, same oracle). So we validate on stimuli that *exercise* the
circuit, with two bars:

- **RECOVERY** (`empirical`) — detect (argmax = the circuit's answer) **and** causal (perturb the operand → the output
  follows), both ≥ τ = 0.8. (Induction additionally reports the exercise-vs-natural gap = the mask.)
- **ADMISSION** (`empirical`) — on a held-out region of NOVEL content (novel to the n-gram *cover*, not necessarily to the
  model's pretraining), the circuit RULE matches the model on strictly more instances than a minimal-suffix n-gram cover
  built on the train split: **`admits ⇔ circuit_match > ngram_match`**, both scored against the model's argmax. No tuned
  threshold — a strict count. Robust because on novel content the n-gram cover scores ~0 (for IOI it forms *zero* rules —
  the answer isn't a function of any suffix), so deltas are large (+50…+100pp). Admission is a coarse "beats memorization"
  screen; the emitted cover's real guarantee is the `equiv.dl` certificate below.

### The unifying finding — reasoning-as-binding (`empirical`; certified over stated domains, **behavioral not mechanistic**)

The 8 admittable families reduce to **three token-level mechanisms**: copy (induction), ordinal (succession), and
**once-appearing / name-mover binding** (output = the entity appearing exactly once among the entity set). Six "reasoning"
families — IOI, transitivity, modus ponens, temporal, spatial, syllogism — are all served by the ONE `once_app` rule.
This is established **behaviorally**: a single Datalog rule reproduces the model's argmax on all six, PROVEN by `equiv.dl`
(`nmiss=0` over the stated domain) — *not* by attention-head patching/ablation, and it is **not** a claim of mechanistic
identity (rosetta certifies computed behavior, per the behavior≡algorithm thesis).

Scope + counterexamples (why it is *not* "all binding"): it holds only where the model's answer IS the structurally-unique
token. Binding tasks whose answer is **not** the once-appearing token do NOT reduce to it and stay recovery-only —
**coreference** (both names appear once; bound by semantic gender) and **set-membership** (both names once; bound by a
co-occurring item). Semantic recall (antonym, capital, analogy) has no structural rule at all. And within the six,
instances where the model diverges from once-appearing (llama MP detect 80%, spatial 96%) are excluded from the
certificate, not certified against.

### Which circuits are admitted (emitted) vs recovery-only

| class | families |
|---|---|
| **STRUCTURAL — admitted + emitted** (a token rule → cover rule) | induction · succession · IOI · transitivity · modus ponens · temporal · spatial · syllogism |
| **SEMANTIC / RECALL — recovery-only** (no structural rule → not emittable) | coreference · antonym · capital · analogy · set · defeasible · causal |

Capability (RECOVERY / ADMISSION) on the capable model **llama-3.2-1B** (detect/causal; Δ = admission vs n-gram):

| family | detect | causal | admits | | family | detect | causal | admits |
|---|---|---|---|---|---|---|---|---|
| succession | 100% | 100% | +100pp | | capital | 100% | 100% | recall (N/A) |
| IOI | 99% | 98% | +98pp | | analogy | 100% | 100% | recall (N/A) |
| transitivity | 100% | 100% | +100pp | | antonym | 71% | 93% | N/A |
| temporal | 100% | 100% | +98pp | | coreference | 73% | 22% | N/A (never confirmed) |
| syllogism | 100% | 98% | +100pp | | defeasible | 60% | 80% | N/A |
| spatial | 96% | 98% | +98pp | | set | 40% | 36% | N/A (hardest) |
| modus ponens | 80% | 88% | +60pp | | causal do/see | ctrl 0% | interv 100% | N/A |

(Induction measured on pythia-160m: causal 82/84/86% at L=1/2/3, +81pp admission; llama induction not run — the 1B novel-repeat sweep exceeds the serve-run budget.)

### The emitted cover + its certificate (`proved` over a stated domain)

`emit_full_cover` writes `circuits.full.dl` = the natural-corpus n-gram cover + the three mechanisms as OOD fallbacks
(routing: **longest n-gram > once-appearing > succession > induction > abstain** — succession is above induction because a
comma-separated run ends in a repeated punctuation token, on which the copy head fires spuriously) + a
`circuits.full.symbols.dl` legible twin, then certifies via `equiv.dl` over natural ∪ circuit-behavior stimuli:

| model | arch · scale | domain (inst) | verdict |
|---|---|---|---|
| pythia160m | NeoX · 160M | 693 | `nmiss=0 ∧ nuncov=0` — CERTIFIED |
| llama32_1b | RoPE · 1B | 944 | CERTIFIED |
| qwen25coder15b | Qwen · 1.5B | 1061 | CERTIFIED |

Per-circuit **certified instances** in the emitted cover (`circuits.full.CERT.json`):

| circuit | mech | pythia160m | llama32_1b | qwen25coder15b |
|---|---|---|---|---|
| induction | copy | 135 | 136 | 133 |
| succession | ordinal | 13 | 23 | 23 |
| IOI | once-app | 69 | 119 | 119 |
| transitivity | once-app | 34 | **0 †** | 100 |
| modus ponens | once-app | **0 †** | 70 | 100 |
| temporal | once-app | 69 | 99 | 86 |
| spatial | once-app | 67 | 97 | 100 |
| syllogism | once-app | 6 | 100 | 100 |

**DOMAIN CAVEAT (`open`).** The certificate is over the *stated domain* only. Because routing is n-gram-**first**, a
stimulus whose template suffix collides with a natural n-gram rule is EXCLUDED — the cover would return the n-gram's
(possibly wrong) answer there, and that instance is outside the certificate, not certified against. So `certified_instances`
per circuit is **domain-dependent, not a capability measure**: the **†** cells are 0 not for lack of capability (both
recover + admit in the measurement) but because the template suffix collides with the natural cover (transitivity ends in
the ultra-common `' a'`; modus ponens' tail collides on pythia). A cover that routes these to the circuit rather than the
pre-empting n-gram is achievable-**open** (precedence tension: routing circuits above n-grams risks spurious firing on
natural text). The clean per-circuit *capability* signal is the RECOVERY/ADMISSION table above, not the certified count.

## Induction wired into an EXPERT PACKAGE (`py/induction_package.py`) · `empirical`

The work above emits induction into the souffle `circuits.dl` (the minimization arm). The bounded-**expert** builder
(`emit_expert_package`, the rosetta→sgiandubh serving path) was a different story: it counted only gate/compose idioms
as the trusted tier and dropped induction into the souffle twin as an **uncounted OOD limb** — so the served
`manifest.json` never carried it. That was the "induction never emitted to the package" gap. Now `emit_expert_package`
counts induction coverage and emits a first-class `induction` manifest rule, and `serve_package` serves it host-side
(routed OOD, after n-grams: `[… A B … A] → B`, copy the successor of the current suffix's previous occurrence).

Measured on **pythia-160m** (resident `fieldrun --serve` oracle; 30 train + 30 **disjoint** held-out novel-repeat
sequences S+S, seqlen 20):

| set | gate/compose | n-grams | induction (causal) | package coverage | precision | abstain |
|---|---|---|---|---|---|---|
| train | 0 | 0 (novel tokens) | L=2 (80%) + L=3 (91%) admitted | — | — | — |
| **held-out · WITH induction** | 0 | 0 | 2 rules | **94% (510/540)** | **84%** | 6% |
| **held-out · n-gram only** | 0 | 0 | — | **0%** | — | **100%** |

So on held-out novel tokens — where the n-gram cache has **no support** — the induction rule is the *entire*
load-bearing tier and it **generalizes** (94% vs 0%). Precision 84% tracks the model's own induction consistency
(obs ~82%), not a rule defect: the served rule is as faithful as the head. Manifest: `trusted_idioms=0,
induction_ood=2` — 0 gate/compose idioms (the natural-corpus finding restated) but induction now carried and served.

This is the **circuit-tier mirror** of the logic-expert answer-tier ablation ([`EXPERTS.md`](./EXPERTS.md)): on a
lookup domain the model's *circuit* tier is empty (0 idioms) and the distilled *answer* tier carries the expert; on
induction stimuli the *circuit* tier is everything (94% vs 0%) and there is no answer tier. Reproduce (needs the
bundle): `.venv/bin/python py/induction_package.py models/pythia160m/bundle 30 20`.
