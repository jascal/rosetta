# An expert build is a reproducible experiment

*Design doc. Companion to [`CONVERGENCE.md`](./CONVERGENCE.md) — which made rosetta the *sole builder*; this defines
**what a build is** and how its quality is established. Forward-looking; claims here are `design`/`open` until the
artifact (a scorecard, an `equiv.dl` certificate) backs them.*

> **Quickstart**: for the current best-practice package checklist and the three ingestion
> routes (document / teacher / feedback), see [examples/UNIFIED_QUICKSTART.md](./examples/UNIFIED_QUICKSTART.md).

## The reframe

A bounded expert is not the output of a pipeline — it is a **claim**:

> over domain **D**, this expert answers at coverage **C** with precision **P**, abstains on the rest, leaks **L**
> off-domain, and scores **B** on benchmark **X**.

rosetta's job *as the builder* is to make that claim **true and backed by an artifact** — the **scorecard** — the same
way a minimized circuit's faithfulness is backed by `dl/equiv.dl`. An unbacked quality claim is exactly the tag-discipline
violation the program forbids.

So the full build has **three first-class inputs**, not one:

1. **Corpus design** — what the expert should know: the content, its structure, its coverage of the domain.
2. **Experimental design** — how quality is measured (held-out + off-domain sets, metrics, thresholds) and what it is
   *aimed at* (benchmark targets).
3. The **model substrate** (optional → a model-free expert).

The package is the *output*; the **scorecard** is the *result*; a **benchmark target** is the *hypothesis*.

## Why validation lives in rosetta, never sgiandubh

- **sgiandubh is a leaf** — it serves a package and knows nothing else; it has no ground truth to grade against, and
  putting eval there re-introduces the hybrid CONVERGENCE.md dissolved.
- **The builder owns the ground truth** — the model (oracle), the corpus, and the ability to hold out a test split.
  Only the builder can grade.
- **Tag discipline makes it mandatory** — every quality claim is `empirical`/`proved` *with the artifact that backs it*.
  "precision 0.97 @ coverage 0.8" is an `empirical` artifact; the cover's faithfulness is a `proved` (`equiv.dl`) one.
  Both are produced at build, by rosetta.

## The build spec (declarative, versioned, reproducible)

A build is defined by a spec, so corpus design + experimental design + targets are explicit and re-runnable — not
buried in a shell invocation:

```toml
# expert.toml — a declarative expert build (a reproducible experiment)

[corpus]                              # CORPUS DESIGN — what the expert knows
text      = "corpora/logic_kb.txt"
questions = "corpora/logic_questions.txt"
citation  = "Open Logic Project (CC BY 4.0)"

[model]                               # the substrate (omit + use [adapter] instead → model-free expert)
bundle    = "$BUNDLE"                 # named, never hard-coded
fieldrun  = "$FIELDRUN"               # the extractor, named (no PATH discovery)

[experiment]                          # EXPERIMENTAL DESIGN — how quality is graded
holdout    = 0.2                      # auto in-domain split (the always-on baseline)
off_domain = "probes/negatives.txt"   # leak/abstain measurement — content OUTSIDE the corpus
testset    = "logic_test.jsonl"       # optional owner set — preferred for deployment claims

[[benchmark]]                         # BENCHMARK TARGETS — first-class, extensible, optional
name   = "olp-qa"
set    = "bench/olp_qa.jsonl"
target = 0.85

[gate]                                # HARD-FAIL thresholds
min_precision = 0.90
max_leak      = 0.05
```

`rosetta build-expert expert.toml` → the **package** + **`scorecard.json`** (+ a per-benchmark report). It **exits
non-zero with no shippable package** if the scorecard misses `[gate]` or a benchmark `target`.

A **model-free** expert (a frozen spec) omits `[model]` and uses `[adapter]` — no cover, no curated answers, retrieval-only:

```toml
# riscv.toml — a model-free expert (retrieval over a structured source, no model)
[corpus]
citation = "RISC-V ISA Manual (CC BY 4.0)"

[adapter]                             # structured-source → citable passages (no model, no fieldrun)
name   = "normrules"
source = "norm-rules.json"

[experiment]
off_domain = "probes/negatives.txt"   # no holdout split — there's no cover/curated tier to hold out

[gate]
max_leak = 0.05                        # no min_precision: rules are returned verbatim; gate on leak/citation
```

## The scorecard (the result)

Graded on the experimental design, over the deployable package's *whole* cascade (not just the cover):

| metric | meaning | tag |
|---|---|---|
| **coverage / precision / abstain** | the reject-option triple, **per tier** (cover / curated / retrieval) | `empirical` |
| **confident-wrong rate** | answered confidently but wrong — the loose-match hallucination | `empirical` |
| **off-domain-leak rate** | answered (not abstained) on an off-domain probe — the "boiling point → Cantor" class | `empirical` |
| **gram-parity** | cover's gated n-grams ⊇ `build_gram` (the gate that retires `gram`) | `empirical` |
| **cover faithfulness** | `dl/equiv.dl` certificate over the stated domain | `proved` |
| **benchmark score(s)** | per `[[benchmark]]`, vs its `target` | `empirical` |

Per-tier attribution is required: every answer is tagged with the tier that produced it, so a failure localizes to a
tier (and thus to a builder) rather than being a single opaque number. This is where the **cover-first composition**
([`CONVERGENCE.md`](./CONVERGENCE.md)) meets the scorecard: each tier carries its own coverage/precision, and the
**cover tier additionally carries a `proved` faithfulness row** (its `dl/equiv.dl` certificate) — so "smart tier
generalizes faithfully" is an artifact, not an assertion.

A `scorecard.json` (illustrative values):

```json
{
  "model": "rosetta-expert-logic",
  "domain": "Open Logic Project",
  "tiers": {
    "curated":   { "coverage": 0.41, "precision": 0.98, "n": 16 },
    "cover":     { "coverage": 0.33, "precision": 0.95, "faithful": "proved", "equiv": "nmiss=0 nuncov=0" },
    "retrieval": { "coverage": 0.11, "precision": 0.82 }
  },
  "abstain": 0.15,
  "confident_wrong": 0.01,
  "off_domain_leak": 0.02,
  "gram_parity": { "recall": 1.0, "confident_disagree": 0 },
  "benchmarks": [ { "name": "olp-qa", "score": 0.86, "target": 0.85, "pass": true } ],
  "gate": { "min_precision": 0.90, "max_leak": 0.05, "pass": true },
  "testset": "auto-baseline"
}
```

## Measured result — the logic expert (the ablation) · `empirical`

The illustrative scorecard above is now backed by a real one. The decisive question for the *whole* model-distilled
arm — **does the model tier earn its place, or is every working expert really document-derived?** — was run on the
Open Logic Project expert (gemma-4-e4b-it distilled over 28 questions × ~251 decode steps, in
[`examples/logic/`](./examples/logic/)). Three package variants that differ **only** in which tiers are present were
served through the real sgiandubh binary with identical flags and graded on one 50-row testset
([`testset.jsonl`](./examples/logic/testset.jsonl): 10 verbatim / 11 paraphrase / 12 held-out / 17 off-domain).
Reproduce: `.venv/bin/python examples/logic/run_ablation.py` → [`scorecard.json`](./examples/logic/scorecard.json).

*Provenance of the model tier.* [`curated_faq.json`](./examples/logic/curated_faq.json) is the frozen distilled FAQ:
`fieldrun --export-logic-corpus` runs **gemma-4-e4b-it** greedily (EOS-stopping, ~251 steps) over each of the 28
questions, and `pack.answers.from_export` concatenates the predicted tokens into one answer per question — **no human
review, editing, or reranking**. So the "model tier" being ablated is the model's own raw greedy generation, distilled
at build and frozen (which is why one answer — "quantifier" — is a garbled self-correction, kept verbatim).

Answered / n per subset (`ii_document` = retrieval over the **raw** OLP dump; `iii_both` = the shipped cascade):

| variant | verbatim | paraphrase | held-out | off-domain leak |
|---|---|---|---|---|
| **require-citation ON** (shipped default) | | | | |
| i_curated (FAQ only) | 10/10 | 12/12 | 9/12 | 0/16 |
| ii_document (retrieval only) | **0/10** | **0/12** | **0/12** | 0/16 |
| iii_both | 10/10 | 12/12 | 9/12 | 0/16 |
| **require-citation OFF** (raw retrieval capability) | | | | |
| i_curated | 10/10 (topical 10) | 12/12 (12) | 9/12 (7) | 0/16 |
| ii_document | 7/10 (**topical 2**) | 9/12 (**2**) | 9/12 (**3**) | **5/16** |
| iii_both | 10/10 (10) | 12/12 (12) | 10/12 (7) | 5/16 |

**Verdict (bounded): on this domain the model tier decisively earns its place — but strictly as a distilled FAQ, not
as a generalizing reasoner.** This *refutes* the prior tentative read that "no shipped expert derives value from the
model." Concretely:

- **In-distribution** (verbatim + paraphrase, n=22): the curated tier answers all 22 with clean, citation-bearing
  definitional prose (manual read: ~21/22 genuinely correct — the lone miss is a garbled "quantifier" distillation).
  Retrieval over the **raw dump** cannot cite these under the shipped `--require-citation` policy (0/22), and even
  uncited it is on-target only ~2/22 (it returns section headers / tangential mentions from the PDF-to-text noise).
  So the model tier's real contribution is **converting a noisy raw corpus into a clean, cited, leak-free answer set**
  that raw-dump retrieval cannot match.
- **Off-domain** (n=16): the curated tier leaks **0** (structural — it only fires on a lexical match to a distilled
  question, and off-domain queries don't match). Uncited retrieval leaks **5/16** (grabs logic fragments on overlap
  words — e.g. "chemical formula for salt" hits the corpus word "formula").
- **Held-out** (n=12, questions never distilled): the curated tier does **not** generalize — it fuzzy-matches to the
  nearest distilled question (shared content words ≥ τ=0.25) and confidently returns *that* answer. The concrete
  drift, from the served outputs:

  | held-out query | fuzzy-matched distill Q | returned (wrong) answer |
  |---|---|---|
  | "What is the **completeness** theorem?" | "…the **soundness** theorem?" | the Soundness Theorem definition |
  | "What is the **compactness** theorem?" | "…the **soundness** theorem?" | the Soundness Theorem definition |
  | "What is the **Löwenheim–Skolem** theorem?" | "…the **soundness** theorem?" | the Soundness Theorem definition |
  | "What is a **conjunction** in logic?" | "What is a **tautology** …?" | the tautology definition |
  | "…difference between a **theory and a model**?" | "…**valid** vs a **sound** argument?" | the valid-vs-sound answer |

  Manual read: only ~2–3/12 genuinely correct; the lenient any-key `topical` metric over-credits the rest. **This is a
  confident-wrong failure, not coverage** — the fuzzy matcher accepts a shared noun ("theorem", "argument") as a match.

That last failure is **calibratable**, and the fix is measured, not asserted: sweeping the curated tier's Jaccard
threshold (`--tau`) shows that raising it from the 0.25 default to **≈0.40** makes the out-of-FAQ held-out questions
**abstain** (held-out answered 9/12 → 2/12, and the 2 survivors are a verbatim distill question + a near-paraphrase)
while fully preserving the FAQ (verbatim 10/10, paraphrase 11/12). τ≈0.40 trades confident-wrong nearest-neighbour
answers for honest abstention — the bounded-expert bias — and is the recommended default for a distilled-FAQ expert
(the analogue of the riscv cos/margin calibration; the server default lives in sgiandubh).

**The fair-baseline follow-up (P1-b) — the win is robust.** The document tier above was the **raw** OLP dump, so
"model tier > raw-dump retrieval" left open whether a *cleanly-adapted* corpus would close the gap. Tested it:
[`clean_kb.py`](./examples/logic/clean_kb.py) filters TOC/boilerplate and attaches `[OLP §N.N]` section handles (1200
→ 1048 citable passages), and [`run_clean_baseline.py`](./examples/logic/run_clean_baseline.py) re-grades curated vs
raw-doc vs clean-doc ([`scorecard_clean.json`](./examples/logic/scorecard_clean.json)). The clean corpus fixes the
*citation* problem — under `--require-citation` it now answers **25/34** (raw answered 0/34) — but **not** the
*relevance* problem: it is on-target only **7/25** and still leaks **3/16**, versus curated's **29/31** on-target and
**0** leak. So the document tier's failure was never just missing citation handles — passage-retrieval (GloVe cosine +
lexical coverage) returns a passage that *mentions the query words*, not one that *answers the question*, for these
"what is X" definitional queries. **The model-distilled tier earns its place even against a cleaned, citeable document
baseline.** The raw→clean→curated progression at a glance (require-citation ON, the shipped policy):

| document source | answered (all cited) | on-target | off-domain leak |
|---|---|---|---|
| raw OLP dump (no handles) | 0/34 | — | 0/16 |
| cleaned (+ `[OLP §N.N]` handles) | 25/34 | 7/25 (28%) | 3/16 |
| **curated (model FAQ)** | **31/34** | **29/31 (94%)** | **0/16** |

Cleaning recovers *answering* (0→25) but not *relevance* (on-target stays ~28%). **The main remaining uncertainty** is
retriever strength: this uses sgiandubh's as-shipped GloVe-cosine + lexical-coverage retriever, so a stronger dense
retriever or a reranker could narrow the document-tier gap — that is the natural next baseline, and the one axis on
which this verdict could still move.

**Connection to the minimization arm.** A high-precision, leak-free, abstaining expert like this is also a clean
*oracle*: it answers a bounded question set faithfully and refuses elsewhere, so it can seed the circuit work — a
scoped query set whose model behaviour is worth decompiling, or a source of causally-checkable idioms for `circuits.dl`.
Putting a real model *circuit* (not just a distilled answer) into a served package — pythia160m induction — is the
first step across that bridge and is tracked separately.

**Caveats (state the domain):** (1) `topical` is a lenient any-key proxy corrected by manual reads above. (2) One
model (gemma-4-e4b-it), one corpus (OLP), 50 rows — `empirical` over exactly that.

## Test-set construction (decided)

A scorecard is only as honest as the set it's graded on (*"recalibrate on a representative test set, not a toy corpus"*).
So:

- **Always (baseline):** an auto in-domain **holdout** split of the corpus/questions **+** a generic **off-domain
  probe** set → a baseline scorecard ships with **every** package.
- **Optional, preferred (deployment-grade):** an owner-supplied **representative `testset`** → the scorecard that
  substantiates real deployment claims.

The baseline catches gross regressions; the owner set substantiates the claim. Both must be **held out** from everything
the build trains on (cover, curated, retrieval) — leakage would inflate the claim.

**Off-domain probes** (the negative set) are **versioned alongside the spec** (referenced by `[experiment].off_domain`,
committed with the expert), and must be genuinely *outside* the domain — a fixed generic negative set (general-knowledge
/ other-domain queries), not adversarial paraphrases of in-domain content. What counts as a *fair* generic negative set
is tracked in Open Questions.

## Gate (decided)

The scorecard **gates the build**: `build-expert` exits non-zero (no package) when precision < `min_precision`, leak >
`max_leak`, or a benchmark misses its `target`. Thresholds are config (`[gate]` / `--min-precision` / `--max-leak`),
tuned per deployment — not a silent default. A shipped expert provably met its bar.

## Corpus design as a *measured* concern

Corpus design stops being a guess. rosetta emits a **corpus report**: vocab / section coverage, dedup; and — once a
testset or benchmark exists — **gap analysis**: which test/benchmark items the corpus does not actually support (so they
can only ever abstain or be wrong). This closes the loop: **corpus design informed by the experimental design**,
measured, not asserted.

## Benchmarks — a first-class slot, not a framework

`[[benchmark]]` names an external eval set + a target; the scorecard reports the score and the gate can require it. We
design the **slot and the report** now and wire a *concrete* benchmark when one is chosen — no speculative benchmark
harness (a non-goal).

## Reuses rosetta's existing eval tooling

This is already rosetta's wheelhouse; `pack.eval` composes the existing primitives over the *deployable package* (all
tiers) on the held-out query set:

- `py/holdout_score.py` — train/holdout generalization ("is it us, not the model").
- `py/abstain_cover.py` + [`ABSTAIN.md`](./ABSTAIN.md) — the coverage / precision / abstain frontier.
- `dl/equiv.dl` — the cover's faithfulness certificate.
- `py/probe_families.py` + [`CROSS_ARCH.md`](./CROSS_ARCH.md) — which reasoning families the cover carries.

## Where it fits

This **elevates CONVERGENCE.md's Phase 3** (validation) from an afterthought to the *defining* concern of the build.
Implementation surface (a later PR, after this design is agreed):

- `pack/spec.py` — load `expert.toml` → a build config.
- `pack/eval.py` — the scorecard (the metrics above) + the hard-fail gate, over the package's full cascade.
- the corpus report + gap analysis.
- the `[[benchmark]]` slot + report.

## Open questions (tagged)

- `open` **leakage** — guarantee the held-out / off-domain sets are unseen by *every* tier the build trains (cover,
  curated, retrieval), not just the cover (`holdout_score` covers the cover today).
- `open` **off-domain probes** — what is a *fair* generic negative set (genuinely outside the domain, not adversarial)?
- `open` **benchmark scoring** — exact-match vs semantic per benchmark; normalize where.
- `design` **spec format** — TOML vs JSON; keep it minimal and obvious.

## Non-goals

- Not a generic ML-benchmark framework — just the slot + report.
- Not moving any validation into sgiandubh — it stays a leaf.
- Not auto-fixing a failing expert — the gate reports/fails; improving corpus + experimental design is the human↔rosetta
  loop, deliberately.
