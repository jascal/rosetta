# Cross-architecture toolkit validation (`empirical`)

Does the symbolic-family toolkit (`py/probe_families.py`) hold across architectures, or is it Llama-specific?
The PIC thesis predicts **behavior ≡ algorithm** — a real circuit (detect + causal) should recur regardless of
substrate. We probed four model-points spanning two architectures and two scales, with templated/nonce stimuli +
a foil + a causal perturbation. A family is **in the toolkit** iff detect ≥ 80% **and** causal ≥ 80%.

| family | llama-3.2-1B (RoPE) | qwen2.5-coder-1.5B (Qwen) | pythia-160m (NeoX) | pythia-1.4b (NeoX) |
|---|---|---|---|---|
| copy/name-mover (IOI) | ✓ 100/98 | ✓ 95/98 | ~ 60/70 | ~ 57/64 |
| succession | ✓ | ✓ (months; commas) | ~ 78/90 | ✓ 100/100 |
| capital-of *(knowledge)* | ✓ 100/100 | ✓ 100/100 | ✗ 0/0 | ✓ 100/100 |
| antonym *(knowledge)* | ✓ 100/100 | ✓ 100/100 | ✗ 8/2 | ✓ 100/100 |
| is-a transitivity | ✓ | ✓ | ✓ 100/100 | ✓ |
| modus ponens | ✓ | ✓ | ~ 35/52 | ✓ 100/100 |
| temporal ordering | ✓ | ✓ | ✓ | ✓ |
| syllogism (instantiation) | ✓ | ✓ | ✓ 98/100 | ✓ |
| spatial (left-of) | ✓ | ✓ | ✓ | ✓ |
| set membership (∩) | ✓ 88/90 | ✗ 0/0 | ✗ 0/0 | ✗ 7/7 |
| defeasible (exception) | ✓ | ✓ | ✓ | ✓ |
| analogy (a:b::c:?) | ✓ | ✓ 85/92 | ~ 22/60 | ✓ 86/100 |
| causal do-vs-see | ✓ | ✓ | ✓ | ✓ |
| coreference (gender) | ✗ 100/57 | ✗ 40/88 | ✗ 15/90 | ✗ 71/36 |
| **confirmed** | **14/15** | **12/15** | **6/15** | **11/15** |

(detect/causal shown where not a clean ✓; `~` = present but below threshold.)

## Verdict: the toolkit is largely architecture-independent

The apparent NeoX "failures" at 160m **decompose into scale and probe-format, not architecture**:

- **Universal — all four points, including tiny NeoX:** is-a transitivity, temporal, syllogism, spatial,
  defeasible, causal do-vs-see. These are pure rule-application over *given* premises (nonce fillers), so they are
  capacity-light and substrate-blind — the strongest PIC-thesis support.
- **Architecture-independent but scale-gated:** capital-of, antonym, modus ponens, analogy. They *fail at 160m and
  recover to 100% at 1.4b NeoX* (and hold at llama/qwen ~1B). The 160m "arch failure" was capacity, not NeoX.
- **Architecture-independent but probe-format-sensitive:** succession. A code model reads bare-space `Mon Tue Wed`
  as a token list and predicts a number; with commas it predicts the successor (logit 18.5). **A 0% can be the probe,
  not the model** — `family_succession` is now format-robust (tries both joins, takes the best).

Two genuine residuals (honest exceptions, not explained away):

- **IOI / copy is weaker in NeoX** — strong in RoPE (llama 100, qwen 95) but ~57–60 in pythia at *both* 160m and
  1.4b (scale doesn't fix it). Either a real arch/training difference (the full S-inhibition + name-mover chain) or a
  NeoX name-tokenization probe artifact — flagged, not yet disambiguated.
- **set-membership is model-dependent** — llama-1B does it (88/90); qwen and both pythias answer the *recency* name,
  not the set-logic one (verified by direct query). The hardest reasoning family; not universal.
- **coreference (gender) is never robustly confirmed** anywhere (detect or causal weak) — consistent with the prior
  llama-only finding (recency vs gender-binding).

## Method notes / caveats

- pythia-160m and pythia-1.4b share the GPT-NeoX tokenizer, so 160m→1.4b isolates **scale** cleanly; llama-1B vs
  pythia-1.4b isolates **architecture** at matched ~1B scale.
- pythia-1.4b is f32 (no int8 build) → run at n=14 (vs n=40 elsewhere); verdicts are robust but the percentages are
  coarser-grained.
- This validates the toolkit as a **capability catalog** (what a model *can* do). It is a separate question whether
  those capabilities earn rules in a model's *cover* (`circuits.dl`) on a generic corpus — they mostly do not, because
  generic windows rarely exercise them (see `py/cover_structured.py`); the cover stays n-gram-dominated until the
  corpus exercises the family.
