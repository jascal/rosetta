# Does a compiled TPR certify decisions that n-grams and idioms cannot? — pre-registration

**Status: written 2026-10-03, committed and pushed before any code for this study existed.**

**Motivation.** pil #136 showed that a certificate-trained TPR substitute is *provably* faithful (pairwise
certificate) on 64–96% of held-out contexts. But it needs GPT-2's unembedding at runtime, and on list copy a
3-token Datalog copy rule already does the job exactly.

The real question: as a **compiled, Soufflé-only layer behind the n-gram and idiom layers**, does a learned TPR
certify GPT-2 decisions that those layers do not? Here "certify" means rosetta's standard: `dl/equiv.dl`, with
`nmiss = 0` over a frozen finite domain.

## 1. Tasks and oracle

**Model and oracle.**
- GPT-2 small.
- **Certification references** come from rosetta's build-time oracle: the fieldrun bundle
  `fieldrun/bundles/gpt2.fieldrun.{bin,json}` (sha256 `4edbcffe…` / `322d7962…`), served with `--serve`, using
  `oracle.serve_decide`.
- The TPR is trained on Hugging Face GPT-2's decode inputs and decisions, train split only.
- Any HF/fieldrun disagreement is reported. Certificates are always against the fieldrun references.

**SVO (passive probe).** `The S V the O. The O was V by the`, 12 GPT-2 tokens.
- S ≠ O from the occupations list (single-token, ≤ 40). V is a regular `-ed` verb (single-token, ≤ 16). These are
  pil `certified_substitutes.py`'s lists.
- Stimulus seed **31**. All S-V-O combinations are shuffled, and the first 12,000 are taken.

**COPY (closed-vocabulary copy stress).**
- Each context is built from a **closed pool** of the first 120 single-token nouns of pil's `NOUNS`: closed, so
  that the TPR's fillers are seen in training.
- A distinct 8-token sequence `s`, then two exposures, then the query prefix `s[:4]`. Target: `s[4]`.
- Three layouts, 4,000 contexts each:
  - `plain`: contiguous copies;
  - `noise8`: 8 unrelated pool tokens after each copy;
  - `near_match8`: 8-token gaps ending in the query's last two tokens followed by a decoy, as in
    `copy-generalization.md`.
- Stimulus seed **32**. All tokens within a context are distinct.

**Splits** (by context, seed 33): 60% train / 10% dev / 30% test. Test contexts containing a (filler, role) pair
unseen in train are dropped, and the count is reported.

## 2. Layers (router order; the first layer that fires decides)

**(a) n-gram.** rosetta's train-only n-gram baseline (`benchmark_induction.learn_baseline`, `dl/ngram_train.dl`).
- **Guard:** keep only rules that fire on ≥ 1 dev context and are correct on all of their dev contexts.

**(b) idiom.**
- **SVO:** a Datalog *copy-subject* rule: output the token at position 1.
  - **Guard:** fire iff the verb (token at position 2) is in a whitelist.
  - Whitelist = verbs with ≥ 3 dev contexts on which the rule was correct every time.
- **COPY:** rosetta's frozen guarded copy circuit, `reference/benchmarks/qwen25_05b_guarded_seed1/selected/
  circuits.dl`, used **unchanged**.
  - Its own guard was dev-selected in that benchmark, so transferring it unchanged to GPT-2 is part of the test.
  - No new guard is selected.

**(c) compiled TPR.**
- **Roles** are parsed **in Datalog from `tok` facts**:
  - SVO: positions 1/2/4 → subject/verb/object, after a template check of the fixed tokens.
  - COPY: role = distance from the end of the context, `len − 1 − pos`.
- **Model:** the certificate-trained TPR from pil, with the `cert` objective of `certified_substitutes.py`,
  d_F = 32, fit seed 0, trained on the train split.
- **Compilation:** to weighted facts `w(v, filler, role) = round(2¹⁶ · ⟨U_v, W(e_F ⊗ e_R)⟩)` and
  `bias(v) = round(2¹⁶ · ⟨U_v, b⟩)`, over a candidate set `C`.
  - `C` = the task's fillers ∪ every token that is GPT-2's decision on some train context.
  - The runtime decision is `argmax_{v∈C} bias(v) + Σ_{pairs} w(v, f, ρ)`, computed in Soufflé. No weights,
    unembedding or Python at runtime.
- **Guard:** fire iff the compiled score margin (top − second over `C`) ≥ `θ`.
  - `θ` = the smallest threshold under which every firing dev context is correct, i.e. 1 + the largest dev margin
    among dev errors.
  - This is computed by a Datalog aggregate over dev facts.

**(d) abstain** otherwise.

**Freezing.**
- Every guard is selected on train + dev only.
- Each layer's **test firing domain** is computed from `tok` facts alone and recorded with hashes **before any test
  reference is read**.

## 3. Certification and metrics

- **Per layer and for the composite router:** `run_equiv` over the frozen test firing domain.
  - Clean iff `nmiss = 0` (and `nuncov = 0`, which holds by construction) → `proved` over that finite domain.
  - Any mismatch fails that certificate. Counterexamples are reported, and nothing in that domain is counted as
    certified.
- **Residual `R`:** the test contexts on which layers (a)+(b), as deployed, do **not** decide GPT-2's answer
  (they abstain or are wrong). It is split by the reference's category:
  - SVO: `R_obj` (the reference is the object token) vs `R_other` (the reference is not a token of the sentence).
  - COPY: `R_ctx` (the reference occurs in the context) vs `R_out` (it does not).
- **Primary metric:** the *additional certified coverage*. It is `|{c ∈ R : layer (c) decides c}| / |R|`, counted
  **only if layer (c)'s test certificate is clean**; otherwise it is 0. It is reported per split of `R`.
- **Secondary:**
  - composite certified coverage of the whole test set, against (a) alone and (a)+(b);
  - layer (c)'s agreement on `R` within its firing domain, labelled `empirical`, not certified.

## 4. Decision rules (fixed now)

- **Q1 (a compiled TPR certifies what n-grams and idioms cannot):** passes **per task** iff layer (c)'s test
  certificate is clean **and** the additional certified coverage is ≥ 0.05 of `R` and ≥ 10 contexts.
  - **Overall: yes** if Q1 passes on both tasks, **partial** if on one, **no** if on neither.
- **Q2 (where it helps):** for each task where Q1 passes, report which residual split it covers. This is
  descriptive.
- **Failure reporting:** if layer (c)'s certificate fails, Q1 fails for that task, and its mismatches are listed.
  There is no re-selection after seeing test references.

## 5. Not claimed

- Certificates are proofs over these finite test domains only.
- The TPR is learned from data, so its rules are not hand-authored; they are not claimed to be human-legible.
- This is not a statement about GPT-2 on natural text.

## 6. Artifacts

- `py/benchmark_compiled_tpr.py` (dataset generation, layers, certification).
- `dl/tpr_runtime_*.dl`.
- pil `experiments/compile_tpr.py` (training + fact export).
- Outcome: `docs/compiled-tpr-outcome.md`. This file is not edited after the first run.
