# examples/rfc — an IETF RFC as a model-free expert

An RFC is a specification: its normative text is exactly the *lookup* domain a bounded, cited, abstaining expert is for
(you never want a model to synthesize over a standard). The adapter (`py/pack/adapters/rfc.py`) reads the canonical
fixed-layout plain text, strips the page furniture (form feeds + running `[Page N]` headers/footers), splits the
numbered sections, and maps them onto the uniform strategy table with the *existing* intents:

| RFC structure | Extraction field | intent it serves |
|---|---|---|
| each paragraph, under its section | a citable `passage` (normative text verbatim) | retrieval / cite |
| `A "resource" is …` / `An object is …` | `defines(passage, term)` | **define** — "what is an object?" |
| the numbered sections | `items(section, "sections")` | **count** / **list** — "how many sections?", "list the sections" |

Define extraction is precision-first: a **quoted** term (`"…" is/means/refers to`) or a genus–differentia paragraph
opener (indefinite article + single-word subject, `An object is …`). Ordinary mid-sentence words never become entities.

## Source

`rfc8259.txt` — RFC 8259, *The JSON Data Interchange Format* (freely reproducible under BCP 78 / the IETF Trust). Fetch
any RFC the same way:

```bash
curl -O https://www.rfc-editor.org/rfc/rfc8259.txt
```

The derived `package/` is the committed artifact.

## Build

```bash
.venv/bin/python -m pack.build examples/rfc/expert.toml
```

## What it answers (deterministically, cited)

- **define** — "what is an array?" → `rfc8259:1#5` (the genus–differentia definition, cited).
- **count** — "how many sections does this RFC have?" → `rfc8259:inventory:total` (22).
- **list** — "list the sections" → `rfc8259:inventory:sections`.
- normative lookups ("what does the spec say about duplicate names?") → the verbatim section paragraph, cited.
- off-domain → **abstain**.
