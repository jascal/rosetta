# `rosetta.pack` — the expert-packaging layer

The *factory* that turns a corpus (+ an optional model) into a **deployable bounded-expert package** for the thin
sgiandubh runtime to serve. rosetta is the sole builder; sgiandubh builds nothing (see
[`../../CONVERGENCE.md`](../../CONVERGENCE.md)). This layer **depends on** the minimization core (`py/`, via `cover.py`);
the core never imports `pack` (enforced by `tests/test_pack.py`).

## Two entries: a declarative spec, or `build_expert()`

**Declarative (preferred)** — a reproducible experiment from an `expert.toml` (see [`../../EXPERTS.md`](../../EXPERTS.md)):
```bash
.venv/bin/python build_expert.py examples/logic/expert.toml   # repo-root launcher (puts py/ on the path)
```
`pack.spec` loads `[corpus] [model] [adapter] [grounding] [experiment] [[benchmark]] [gate] [reasoning]` (`$ENV`
resolved), builds the package into `<spec_dir>/package/`, then — if `[gate]` is set — `pack.eval` **scores** it
(serves a held-out + off-domain set: coverage / precision / abstain / leak) and **hard-fails** the build below the gate.
The opt-in `[reasoning]` tier (REASONING.md) is *recognized* but deliberately not auto-wired.

**Programmatic** —
```python
from pack import build_expert
build_expert(out, *, corpus=…, bundle=…, questions=…, adapter=…, cover=…, fieldrun=…, …)
```

`build_expert()` is the single entry — it orchestrates the modules in the right order and emits the package. Three shapes:

| shape | inputs | produces |
|---|---|---|
| **model-distilled** | `bundle` + `questions` (+ `corpus`, `cover=True`) | curated answers + grounding (+ cover) |
| **model-free** | `adapter=…` + `adapter_source=…` | citable passages + grounding (no cover, no curated) |
| **corpus-only** | `corpus` | grounding + empty index |

Call the individual modules (`answers`, `grounding`, `cover`, `adapters.*`) only when scripting a non-standard build.
The extractor is **named explicitly** — `fieldrun=…` or `$FIELDRUN` (no hard-coded path, no PATH discovery), so a build
is reproducible.

## Document adapters — one contract, many sources (model-free)

A model-free "document expert" is a SOURCE through a registered adapter that yields one `Extraction` (citable passages +
the structural facts — `defines`/`statements`/`items`/`answers` — that drive the uniform strategy table). The builder and
runtime consume only the `Extraction`, so **adding a domain is writing one adapter**, never touching the builder, the
runtime, or ergo (as long as it reuses the existing intents `define`/`count`/`list`/`theorem`). See `adapters/base.py`.

| adapter | source | structural facts it emits (intents) |
|---|---|---|
| `normrules` | a spec's machine-readable normative rules (`norm-rules.json`) | passages + items (count/list) |
| `riscv_prose` | an AsciiDoc spec manual's prose | passages + defines |
| `pretext` | a PreTeXt / MathBook-XML textbook | passages + defines + statements (theorem) |
| `latexml` | LaTeXML HTML5 + MathML (arXiv / ar5iv) | passages + defines + statements |
| `glossary` | a term→definition file (`.tsv`/`.json`/colon-form) | passages + defines + items |
| `rfc` | an IETF RFC plain-text spec (`rfc-editor.org` `.txt`) | passages + defines + items (sections) |
| `manpage` | a rendered man page (`man x \| col -bx`) | passages + defines (cmd + long flags) + items (options) |
| `nh_legal` | a statute / legal code (RSA §-numbered text) | passages + defines (`"term" means …`) + items (sections) |
| `pedagogy` | teaching-template TOML | passages + answers (the `pedagogy` intent) |

Per-domain count/list citation handles are namespaced with `[reasoning] prefix` (e.g. `webnet:inventory`,
`cut:inventory`) — not the historical hard-coded `riscv:inventory`. Runnable examples: `examples/{glossary,manpage,rfc}/`.

## Composition — cover-first (why)

Every serving tier is ultimately a lookup, so *fast* is a given; the discriminators are **accurate** (abstain / cite /
no-hallucinate) and **smart** (generalize). So the package is composed:

> **cover** (smart: causal idioms + gated n-grams) → **curated answers** (gated FAQ) → **grounding** (citation;
> retrieval hard-gated at the runtime) → **abstain**

- **gram is dropped** — the cover's gated n-grams subsume it; shipping bare n-grams trades accuracy for cheap coverage.
  `gram.py` is kept *only* to run the coverage-parity check, then deleted.
- **grounding is citation-first** — its job is to attach the source passage; retrieval-as-answer is a hard-gated fallback
  at the runtime (kills the wrong-but-cited / off-domain-leak failures).

## The emitted package

| file(s) | tier | built by | optional? |
|---|---|---|---|
| `manifest.json` | cover (causal idioms + gated n-grams) | `cover.py` (needs a model) | model-free → absent |
| `index.json` + `facts_*/` | curated Q&A | `answers.py` (needs a model) | model-free → empty |
| `knowledge.tsv` (+ `wordvec.txt`) | grounding / citation | `grounding.py` | present if a corpus is given |
| `rules.txt` | model-free citable passages | `adapters/normrules.py` | adapter builds only |

Optional components are detected by **presence** (a model-free expert ships no `manifest`/curated items). `build_expert`
emits the **package only** — it does *not* build the sgiandubh runtime binary (that stays sgiandubh's job; the package
is the only interface).

## A build is a reproducible experiment

What an expert build *is*, how its quality is measured (scorecard: coverage / precision / abstain, hard-fail gated), and
how corpus + experimental design + benchmark targets drive it: see [`../../EXPERTS.md`](../../EXPERTS.md). The
`pack.eval` scorecard + spec loader land in a later phase.
