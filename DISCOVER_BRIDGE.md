# DISCOVER bridge — can a fitted role scheme propose rules the miner cannot find? (`open`, pre-registered, NOT built)

McCoy, Soulos, Linzen and Smolensky (2026, *The Emergent Symbolic Structure of Artificial Neural Networks*,
arXiv:2608.29530) fit hidden states with a linearly-transformed tensor product representation (TPR),
`E ≈ W(Σᵢ fᵢ ⊗ rᵢ) + b`. Fillers `fᵢ` are tokens and roles `rᵢ` are hypothesized structural positions; the fit is
done under a role scheme chosen in advance. For GPT-OSS on six symbolic tasks, substituting the fit for every input
representation keeps task accuracy within 2.4 points. Their 31 kinds of constituent edit average 0.903 accuracy.

A **role scheme is naturally a Datalog relation** over token positions, `role(inst,pos,r)`:
- left-to-right or right-to-left index;
- bidirectional `(i, n−1−i)`;
- predecessor token;
- Wickelroles (previous and next token);
- a task parse path.

rosetta's stated bottleneck is candidate generation (`IDIOM_LEARNER.md`). A role scheme that fits the model well could
therefore be read as a proposal: *try rules keyed on this relation*.

**Verdict up front: do NOT build the proposer.** It is held behind the pre-registered gate below. The gate is written
before any fit is run, and no plumbing in the certified path may be added until it passes. If built, DISCOVER would
be an **untrusted proposer** only. `dl/equiv.dl` remains the sole arbiter of `proved`, and the proposer can never
promote a tag.

## Why a skeptical prior

This is the third proposer of this shape. Both earlier ones were measured and not built:

- **SAE_BRIDGE (phase 1.5/1.6):** entity information was present in features and causally patchable (63–81%), but it
  could not be extracted as a generalizing certified rule on Pythia-160m or Gemma Scope.
- **JLENS_BRIDGE (2026-07-10):** the derived per-context prior did not discriminate between rule families on any local
  model.

The paper's evidence sits in the same regime: representations that are *present and causally relevant, but
approximate*. 0.903 edit accuracy is far from `nmiss=0`. The authors themselves describe the target as
*limitivist*: networks approximate a symbol system without realising it exactly.

The paper also bounds what a fit can say. A good fit with a rich scheme does not show the model uses that scheme, since
a bidirectional scheme can degenerately realise a left-to-right one (their §3.5). A role scheme is also a hypothesis
about **representation**, whereas a rosetta rule is a claim about the **decision**.

## The gate (pre-registered 2026-10-03)

**Model and site.** Qwen2.5-0.5B-Instruct. Residual at the final query position at layers {25%, 50%, 75%, last}.
Activations are build-time only; a Hugging Face forward pass is acceptable for this gate. The fit is a minimal
reimplementation, `W(Σ f⊗r)+b` trained with MSE (≈50 lines of torch). The authors' code is partial.

**Role schemes, fixed now.**
1. left-to-right;
2. right-to-left;
3. bidirectional;
4. predecessor;
5. Wickelroles;
6. bag-of-words (the null);
7. the suffix-match relation the certified guard uses: the role of a token is its offset after the most recent earlier
   occurrence of the query's last three tokens.

**Fit score.** A fit is scored by **last-layer substitution decision agreement**: replace the final-layer residual with
the fit and check that the model's argmax is unchanged. MSE is not used. Scores come from a held-out split, with
withheld (token, slot) pairs as in `docs/copy-generalization.md`.

### H1 — sanity: does the best fit name the rule we already certified?

**Domains** (both have certified rosetta rules):
- the guarded-copy family (`reference/benchmarks/qwen25_05b_guarded_seed1` design);
- the recursion ladder (`dl/reach_circuit.dl`).

**Pass:** on the copy domain, scheme 7 or predecessor (4) ranks first by substitution agreement at some layer, and
beats bag-of-words by at least 20 points.

**Kill:**
- a scheme with no relation to the certified rule ranks first at every layer; or
- every scheme is within 5 points of bag-of-words.

H1 is necessary but not sufficient. It shows only that fit ranking can recover a known answer.

### H2 — value: does it propose a rule the miner missed?

**Domain:** the MiMo / Qwen copy-stress conditions where rule search has stalled. Raw copy is defeated by near-match
distractors, and `rule_search_expansion.md` added 42 candidates and admitted none.

**Procedure:**
1. Fit the schemes on development data only.
2. Translate the top-ranked scheme into Datalog guard candidates keyed on its role relation. There may be at most 10
   candidates, written before any test reference is seen.
3. Run them through the **existing** selection-then-frozen-test protocol unchanged (`py/benchmark_guarded_copy.py`).

**Pass:** at least one proposed candidate is admitted on development data **and** earns an exact frozen-firing-domain
certificate on fresh test references. The existing 42-candidate search admitted none.

**Kill:** zero admitted candidates. Record a measured no-build, as for J-Lens.

### What does not count

- Fit MSE, probe accuracy, or explained variance.
- Edit (constituent-surgery) accuracy below 1.0, however high.
- Any certificate computed on development data, or on a domain chosen after seeing references.
- A win that depends on a role scheme added after this document was written. That would be a new pre-registration.

### Budget

CPU or single-GPU, at most one working day of compute. If the gate needs more, that is a finding about cost, recorded
as such.

## If it passes

The minimal hook is a role-relation **candidate source** feeding the existing selection. It emits Datalog
`role(inst,pos,r)` definitions and guard templates. Selection and certification stay unchanged, and the runtime
artifact stays Soufflé-only: role relations are computed from `tok` facts, never from a model. Nothing else in
rosetta is built.

## Status

`open`. No fit has been run. The withheld-pair protocol and binding-blind baseline (`dl/binding_baseline.dl`) were
built because they are useful without DISCOVER. They are the evaluation half of this gate, not the proposer.
