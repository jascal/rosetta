# AGENTS.md — rosetta orientation

## What rosetta is

rosetta minimizes a whole LLM into **provably-faithful Datalog**: it consumes the faithful whole-model program emitted by
`fieldrun export --logic-whole`, identifies the circuits the model computes, and certifies each one equivalent to the
model with a **Datalog query** (`dl/equiv.dl`). The output is a small `circuits.dl` plus a certificate. Read
[`README.md`](./README.md) first for the thesis and pipeline.

**Datalog is the implementation.** Anything that must be trusted — the equivalence proof, the causal ablation, the
circuit router — is a `.dl` program. Python (`py/`) is a thin driver: it runs `souffle`, stages facts, and reports the
verdict that Datalog computed. When extending rosetta, prefer expressing a transform *and its check* in Datalog over
doing it in Python; the whole point is that the certificate is a query result, not a claim.

## The runtime-independence invariant (non-negotiable)

The deliverable — the minimized `circuits.dl` — **runs in souffle alone: `token` facts in → `decide` out, with NO
fieldrun, no Python, no weights, no `whole.dl` at runtime.** fieldrun (and Python, and the corpus) are **build-time
only** — the oracle that produces and certifies the circuits, like a compiler's reference: used to build/verify, *gone*
once the artifact exists. Pragmatic use of fieldrun for refs during minimization is fine and expected; a fieldrun (or
any binary) dependency leaking into the *runtime* product is not.

Concretely this forbids one tempting shortcut: **the residual must never fall back to the model at runtime.** A context
the cover doesn't reach is handled by Datalog (a certified default / lookup, or an explicit abstain) — never by calling
fieldrun or `whole.dl`. "Complete standalone replacement" = 100% coverage by certified circuits + a Datalog residual
policy; until the tail is closed, ship a `circuits.dl` that is faithful-where-it-fires and abstains elsewhere, still
souffle-only.

## Place in the PIC program

rosetta is part of the PIC research instrument (see `../RESEARCH_MANIFESTO.md`), the **minimization arm** of the
certified-compression loop:

- `i-orca` — verifies (machine-checked proofs)
- `fieldrun` — analyzes (decompiles weights; emits `whole.dl`, the causal/recursion probes)
- `pil` — learns (margin-widening etc.)
- **rosetta** — minimizes (this repo): `whole.dl` → certified `circuits.dl`

It consumes `fieldrun`'s `whole.dl` and shares the formalism in `../pic` (`PIC_SPEC.md` is the source of truth). It does
**not** modify `fieldrun`; emitter changes (e.g. multi-instance `whole.dl`) belong in a `fieldrun` branch + PR.

## Tag discipline (inherited from the program)

Every claim is `proved` / `empirical` / `open`. In rosetta concretely:
- `proved` — a circuit with a clean `dl/equiv.dl` certificate (`nmiss=0 ∧ nuncov=0`) over a stated domain. State the
  domain — exhaustive over a finite domain is a proof *over that domain*, not unconditionally.
- `empirical` — coverage / size numbers measured over a corpus sample.
- `open` — a proposed circuit not yet certified, or a structure-search hypothesis.
Never promote a tag without the artifact (here, the `souffle` output) that backs it.

## What's logic, what's I/O

The dividing line: **every decision is Datalog; the host (Python) only does I/O** — stage input fact files, invoke
souffle, read output relations. souffle can't read a checkpoint, generate a corpus, spawn itself, or serialize a
runnable `.dl`, so a thin host shim is irreducible; but mining, detection, the cover, and certification are all `.dl`.
`dl/master.dl` shows the direction: it `#include`s the modules and runs detection + cover + certificate in ONE souffle
run. The last piece still outside it is computing `ref` (the forward) per context — that needs a **multi-instance
forward emit** (`ctx(inst,pos,id) → ref`, a fieldrun change); with it, the master takes weights + corpus in and emits
certified circuits out, end to end.

## Layout

- `dl/equiv.dl` — the keystone equivalence verifier (`tok`, `ref` in; `mismatch`/`uncovered`/`certified` out).
- `dl/ngram.dl` — automatic n-gram detection: shortest model-deterministic suffix per context (`minorder`, `orderhist`).
- `dl/induction.dl` — the first NON-n-gram detector: induction/copy (`[… A B … A] → B`), certified against `ref`.
- `dl/master.dl` — the process as one program: `#include`s the modules → detection + cover + certificate in one run.

### The circuit-detector library
Minimization = a growing library of certified circuit detectors, each a `dl/` module checked against the model's `ref`
(the same way `equiv.dl` certifies). `ngram.dl` is the recall floor; `induction.dl` the first computation circuit; next
are agreement, delimiter/bracket-matching, coreference. On a tiny model the n-gram cover looks complete; on a real one
it leaves a heavy long-order tail, and that tail is the map to the non-n-gram circuits the other detectors capture.
- `py/oracle.py` — `souffle` driver: `decide`, `certify`, `run_equiv`, `detect`, `run_master`, and `compiled` (native).
- `py/split_facts.py` — split inline-fact `whole.dl` → tiny `forward.dl` + `weights/*.facts` data modules (~100× faster).
- `py/minimize.py` — model-general: build a certified circuits program + auto-write the per-model certificate.
- `reference/<model>/` — the Rosetta Stones: `whole.dl`, certified `circuit.dl`/`circuits.dl`, `corpus.json`,
  `lexicon.json`, `CERTIFICATE.md`. `threx` is the seed (from the fieldrun whole-datalog experiment).
- `models/<model>/` — real models being minimized, same shape.
- `tests/` — keep the reference certificates clean.

## Build & test

Needs `souffle` on PATH and Python 3 (stdlib only so far).

```bash
python3 py/verify_threx.py        # end-to-end keystone: certify the threx composed circuit (expect CERTIFIED 25/25)
python3 -m pytest tests/ -q       # the reference certificates must stay clean
```

## Conventions

- A new model to minimize = a new `models/<name>/` dir with `whole.dl` + `corpus.json` (+ `lexicon.json` if symbolic).
- The goal is **breadth before depth**: minimize tens of small models to confirm the rewrite-rule/idiom library is
  complete, before scaling to large vocabularies (where the dense-Gram wall and multi-instance emission matter).
- Provenance: the threx reference and the discovery methods (suffix mining, causal localization, additive-structure
  search) were developed in `fieldrun/experiments/whole-datalog` (branch `feat/whole-datalog-poc`). That experiment is
  the lab notebook; rosetta is the distilled, Datalog-native tool.
