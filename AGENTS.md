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

## Layout

- `dl/equiv.dl` — the keystone equivalence verifier (`tok`, `ref` in; `mismatch`/`uncovered`/`certified` out).
- `dl/` — Datalog impl: equivalence, ablation, routing.
- `py/oracle.py` — `souffle` driver: `decide(whole, ctx)` and `certify(circuit, whole, instances)`.
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
