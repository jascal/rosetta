# examples/manpage — a Unix CLI reference as a model-free expert

A man page is already a structured reference. The adapter (`py/pack/adapters/manpage.py`) maps it onto the uniform
strategy table with the *existing* intents — no model, no new ergo cue, no runtime change:

| man-page structure | Extraction field | intent it serves |
|---|---|---|
| each prose paragraph (NAME, DESCRIPTION, …) | a citable `passage` | retrieval / cite |
| each option (a flag-tagged paragraph) | a citable `passage` (per option) | retrieval / cite |
| the command name | `defines(DESCRIPTION, name)` | **define** — "what is cut?" |
| each long flag `--bytes` | `defines(option, --bytes)` | **define** — "what does --bytes do?" |
| the options | `items(flag, "options")` | **count** / **list** — "how many options?", "list the options" |

Options are found wherever a paragraph *begins* with a flag — GNU coreutils lists them under DESCRIPTION, not a separate
OPTIONS section, so the adapter does not depend on the section name. Single-letter short flags are case-sensitive but the
runtime folds case, so only the case-stable `--long` flags become `define` entities.

## Source

`cut.man.txt` — the rendered `cut(1)` page, produced (universally reproducible on any box with `man`) by:

```bash
man cut | col -bx > cut.man.txt          # col -b strips backspace overstrike, -x expands tabs
```

The derived `package/` is the committed artifact. Point the adapter at any command's rendered page to build a different
CLI expert.

## Build

```bash
.venv/bin/python -m pack.build examples/manpage/expert.toml
```

## What it answers (deterministically, cited)

- **define** — "what is cut?" → `man:cut.description`; "what does --bytes do?" → `man:cut.opt.bytes`.
- **count** — "how many options does cut have?" → `cut:inventory:total`.
- **list** — "list the options" → `cut:inventory:options`.
- off-domain → **abstain**.
