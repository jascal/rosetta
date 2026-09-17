# examples/glossary — a term→definition glossary as a model-free expert

The simplest structured source there is. A glossary is *already* the exact structure the uniform strategy table wants,
so the mapping is exact — no model, no heuristics, no formatting loss:

| glossary structure | Extraction field | intent it serves |
|---|---|---|
| one entry (term + definition) | a citable `passage` | retrieval / cite |
| the term | `defines(passage, term)` | **define** — "what is TLS?" → its definition |
| the term, grouped by category | `items(term, category)` | **count** / **list** — "how many terms?", "list the Transport terms" |

All three intents are the *existing* strategy vocabulary (`define`/`count`/`list`), so there is no new ergo cue and no
runtime change — adding this domain was writing one adapter (`py/pack/adapters/glossary.py`) and this spec.

## Source

`webnet.tsv` — a Web & Networking glossary (CC0), one entry per line: `term <TAB> definition <TAB> category`.
The adapter also accepts `.json` (`{term: def}` or `[{term, definition, category}]`) and colon-form lines
(`Term: definition`). The derived `package/` is the committed artifact.

## Build

```bash
.venv/bin/python -m pack.build examples/glossary/expert.toml     # or: python py/pack/build.py examples/glossary/expert.toml
```

Produces `package/`: `strategy.tsv` (the uniform `answer <intent> <entity> <passage>` table, souffle-certified at build),
`knowledge.tsv` + `corpus.txt` (citation-first grounding), `index.json` (empty — model-free). Serve it as a `sgiandubh`
spoke; the runtime is one uniform lookup over `strategy.tsv` with grounding fall-through, then abstain.

## What it answers (deterministically, cited)

- **define** — "what is a socket?" → `webnet:socket` (its definition, cited).
- **count** — "how many terms are defined?" → `webnet:inventory:total` (29, closed-world over the glossary).
- **list** — "list the Security terms" → `webnet:inventory:Security` (the enumerated members).
- off-domain ("what is the capital of France?") → **abstain** (the entity is not in scope).
