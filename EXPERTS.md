# An expert build is a reproducible experiment

*Design doc. Companion to [`CONVERGENCE.md`](./CONVERGENCE.md) — which made rosetta the *sole builder*; this defines
**what a build is** and how its quality is established. Forward-looking; claims here are `design`/`open` until the
artifact (a scorecard, an `equiv.dl` certificate) backs them.*

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
