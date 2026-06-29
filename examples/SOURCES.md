# Example expert sources & attribution

Two reference experts, **built and tested entirely inside rosetta** (CONVERGENCE.md: rosetta is the sole builder).
Both source corpora are CC-BY (commercial-OK), so the content is committed here for reproducibility.

## logic — `examples/logic/`
- **Source:** [Open Logic Project](https://openlogicproject.org/) — `logic_kb.txt` (knowledge base) + `logic_questions.txt`.
- **License:** CC BY 4.0.
- **Model:** gemma-4-e4b-it first (scorecard-driven swap).

## riscv — `examples/riscv/`
- **Source:** [RISC-V ISA Manual](https://github.com/riscv/riscv-isa-manual) (the ratified spec).
  - `rules.txt` — 3258 normative rules as citable passages (from the spec's `norm-rules.json`).
  - `questions.txt` — 416 common RISC-V questions (per-instruction + curated concepts).
  - **TODO (corpus design):** add the manual **prose** (definitions/explanations) — the norm-rules are normative
    statements, not definitions, so "what is machine mode?" needs the prose. Source from the manual (CC BY 4.0).
- **License:** CC BY 4.0.
- **Model:** llama-3.2-1b (chosen for a denser cover).

## Off-domain probes — `examples/probes/negatives.txt`
A generic out-of-domain negative set (general knowledge, outside both logic and RISC-V) for the scorecard's
leak/abstain measurement. Genuinely off-domain, not adversarial paraphrases (see EXPERTS.md).

> Model bundles (the `.fieldrun.*` files) are **not** committed — they are large and reconstructible via fieldrun
> `convert`. The `expert.toml`s name them; set `$BUNDLE` / `$FIELDRUN` at build time.
