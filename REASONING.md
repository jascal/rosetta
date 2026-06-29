# Authored-deductive reasoning — how a *model-free* expert reasons

*Design doc. The third reasoning mode. Companion to [`EXPERTS.md`](./EXPERTS.md) (a build is an experiment) and
[`CONVERGENCE.md`](./CONVERGENCE.md) (rosetta is the builder). Forward-looking; `design`/`open` until the artifacts
(ergo's `proved` rules, the materialized closure) back them.*

## The thesis

A **model-distilled** expert gets its reasoning from the **extracted cover** (causal idioms — the model already learned
it). A **model-free** expert has *no cover* — so where does *its* reasoning come from?

> **Authored deductive rules** (ergo's, verified sound) over **facts curated from the domain source**, with the
> deductive closure **materialized at build**. ergo's rules are the reasoning tier for model-free experts.

This is chosen by **measurement**, not by default: we distilled riscv with llama-3.2-1b and the cover's reasoning tier
came up empty (`idiom_learn`: 0 causal gate/compose idioms — a frozen spec is a lookup domain for a small model). So
riscv is model-free, and its reasoning — *if we want any* — is **authored**, not extracted.

## Three modes — no universal law, chosen per domain by measurement

| expert | reasoning source | when |
|---|---|---|
| model-distilled (e.g. logic) | **extracted cover** (causal idioms) | the model carries it (idioms confirmed) |
| model-free **+ derivable structure** (e.g. riscv) | **authored ergo rules + curated facts** | structure is factual/derivable; the model didn't learn it |
| model-free, no structure | retrieval / lookup only | the source is just text to cite |

## What ergo supplies (the verified rule library)

`ergo/core.dl` — authored **and soundness-verified** (`verify_soundness.py`; `proofs/core_rules.i.orca.md` → Isabelle).
All positive Horn ⇒ monotone + sound. The families a structural domain uses:

- **structural** — `extends`/`feature` **inheritance** (`feature(Sub,F) :- extends(Sub,Base), feature(Base,F)`) and
  `part_of`/`within` recursive containment — *the RISC-V case*.
- **predicate** — `subsumes` (is-a) transitivity + universal instantiation.
- **propositional** — modus ponens, hypothetical syllogism, conjunction intro/elim; modus tollens + contradiction flag
  (explicit, open-world `neg`).
- equality/congruence; arithmetic comparison transitivity.

(NAF / closed-world, disjunction, existentials are explicit **gaps** — see `ergo/GAPS.md`; fine for monotone structural
facts, flagged where CWA would change an answer.)

## How it builds (forward-closure at build — ergo INTEGRATION.md)

1. **Facts** — curate the domain's structural relations from the source: riscv `extends(rv64i, rv32i)`,
   `feature(rv32i, add)`, `part_of(insn, ext)`, `subsumes(specific_csr, csr)` — each **`vouched`** with provenance (the
   rule id / section it comes from).
2. **Closure at build** — run ergo's rules over the facts → the least fixpoint (all inherited features, transitive
   relations, derived props). Reason **forward at build**, not backward at query.
3. **Join the KB** — derived facts become passages in `knowledge.tsv`, **cited to their derivation**, so the existing
   retrieval path answers over the enriched KB unchanged. "Does RV64I have ADD?" → `feature(rv64i, add)` is now a
   materialized fact retrieval finds.

No NL→goal translation, no runtime engine — reasoning is a *build stage that emits cited facts*; the thin runtime stays
thin (it just retrieves more facts).

## rosetta is *aware* of it — opt-in, and wiring it takes care

rosetta recognizes authored reasoning as an **available tier for the model-free / retrieval case**, surfaced in the
build spec — but it is **off by default and deliberately wired**, because enabling it is real work, not a flag:

```toml
# in an expert.toml (model-free expert) — OPT-IN; see "takes care" below
[reasoning]                          # authored-deductive tier (ergo) — only for model-free experts
rules   = "ergo:core"                # the verified rule library (extends/feature, is-a, part_of, modus ponens)
facts   = "examples/riscv/facts.dl"  # CURATED domain facts (extends/feature/part_of/is-a), vouched to the spec
closure = "unbounded"               # forward-closure at build → derived facts join the KB (cited to their derivation)
```

**"Takes care" — why it's opt-in, not automatic:**
- the **facts** must be *curated/vouched* (the extends/feature/part_of graph) — this is human/extraction work, and a
  wrong base fact yields confidently-wrong derived facts (sound rules don't save bad inputs);
- the **rules** are sound but **open-world** — choosing them per domain (and respecting the NAF/disjunction gaps) is a
  judgment call;
- the **closure** must be re-run on any corpus/fact change (build-system reframe), and its output audited.

So rosetta *offers* the tier and validates it (the scorecard grades the derived-fact answers like any other), but does
**not** auto-derive reasoning for a model-free expert — you opt in, curate, and verify.

## Scope + tag discipline

- A **`pack/` reasoning tier** (deployment) — the minimization **core stays pure**. ergo is the authored-rule library
  (repositioned, not retired); `pack` consumes it for model-free experts.
- **`proved` rules** (ergo's verified soundness) + **`vouched` facts** (provenance to the source) ⇒ **sound derived
  facts**, each carrying its derivation as provenance. This is a *stronger* tier than the extracted cover (`empirical`):
  authored deduction is `proved`-backed.
- **Fidelity:** a model-free expert has no model to be faithful to → close **unbounded** (the full sound fixpoint — the
  expert can be *more correct* than any model). The "out-reason the model" capability is the point here, not a liability.

## Honest update to the ergo decision

We earlier retired ergo ("replaced by rosetta") on the premise that rosetta would *extract* all reasoning from the
model. The riscv measurement disproved that premise for model-free domains. So ergo is **repositioned, not retired**: the
authored-reasoning library for model-free experts. rosetta-as-builder picks the reasoning source **per domain, by
measurement** — extracted cover, authored deduction, or none.

## Open questions / non-goals

- `open` **fact sourcing** — hand-curate first (the ~40 riscv extensions + inheritance + key is-a is small and
  high-value); auto-extraction from prose (parsing/NER) is a later, separately-validated step.
- `open` **rule selection per domain** — riscv uses extends/feature/part_of/is-a (+ conditional); not all of `core.dl`.
- `open` **CWA boundary** — ergo is open-world; flag any query where closed-world assumption would change the answer.
- **non-goals:** not a general theorem prover; not runtime reasoning (build-time closure only); not moving reasoning into
  sgiandubh (derived facts ship as ordinary cited KB passages — the runtime stays a pure server).
