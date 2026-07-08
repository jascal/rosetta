# Example expert sources & attribution

Two reference experts, **built and tested entirely inside rosetta** (CONVERGENCE.md: rosetta is the sole builder).
Both source corpora are CC-BY (commercial-OK), so the content is committed here for reproducibility.

## logic — `examples/logic/`
- **Source:** [Open Logic Project](https://openlogicproject.org/) — `logic_kb.txt` (knowledge base) + `logic_questions.txt`.
- **License:** CC BY 4.0.
- **Model:** gemma-4-e4b-it first (scorecard-driven swap).

## riscv — `examples/riscv/`
- **Source:** [RISC-V ISA Manual](https://github.com/riscv/riscv-isa-manual) (the ratified spec).
  - `rules.txt` — 3258 normative rules as citable passages (grounding/citation).
  - `rules_plain.txt` — the same text without `[norm:…]` markup (the gated-n-gram corpus).
  - `questions.txt` — 416 common RISC-V questions (kept for reference / a future test set).
  - **TODO (the "more content" lever — now the PRIMARY answer source):** add the manual **prose**
    (definitions/explanations), sourced from the manual (CC BY 4.0). The norm-rules are statements, not definitions.
- **License:** CC BY 4.0.
- **Mode: MODEL-FREE (measured).** We distilled riscv with llama-3.2-1b; the scorecard showed the cover's smart tier is
  empty (`idiom_learn`: 0 causal gate/compose idioms — a frozen spec is a *lookup* domain for a small model). So riscv
  is retrieval over the spec (cited) + gated corpus n-grams + abstain — no model. (Contrast: `logic` is model-distilled;
  it's a reasoning domain.) This is the two-mode demonstration, justified by measurement rather than assumption.

## glossary — `examples/glossary/`
- **Source:** `webnet.tsv` — a Web & Networking glossary authored for this example (~29 terms).
- **License:** CC0 (public domain dedication).
- **Mode: MODEL-FREE.** The `glossary` adapter maps each entry to a citable passage + an exact `define` + a count/list
  inventory item. Exercises **define + count + list** with the existing intent vocabulary (no model, no new ergo cue).

## manpage — `examples/manpage/`
- **Source:** `cut.man.txt` — the rendered `cut(1)` page (`man cut | col -bx`), GNU coreutils.
- **License:** the rendered text is a derived snapshot; regenerate locally on any box with `man` (see the example README).
- **Mode: MODEL-FREE.** The `manpage` adapter emits per-section passages, per-option passages, command/long-flag
  `define`s, and an options inventory (count/list). Options are found under DESCRIPTION *or* OPTIONS.

## rfc — `examples/rfc/`
- **Source:** `rfc8259.txt` — IETF RFC 8259, *The JSON Data Interchange Format*.
- **License:** freely reproducible under BCP 78 / the IETF Trust legal provisions.
- **Mode: MODEL-FREE.** The `rfc` adapter strips page furniture, splits numbered sections into cited passages, extracts
  quoted-term and genus–differentia `define`s, and builds a section inventory (count/list). Verbatim normative text.

## nh_family — `examples/nh_family/` (template)
- **Source:** NH RSA Title XLIII (public domain) — sourced into `raw/` per that example's `SOURCES.md`; no legal text is
  committed. The `nh_legal` statute adapter is implemented and unit-tested; the spec builds once `raw/` is populated.

## Off-domain probes — `examples/probes/negatives.txt`
A generic out-of-domain negative set (general knowledge, outside both logic and RISC-V) for the scorecard's
leak/abstain measurement. Genuinely off-domain, not adversarial paraphrases (see EXPERTS.md).

> Model bundles (the `.fieldrun.*` files) are **not** committed — they are large and reconstructible via fieldrun
> `convert`. The `expert.toml`s name them; set `$BUNDLE` / `$FIELDRUN` at build time.
