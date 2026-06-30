# examples/pretext — open-textbook experts via the PreTeXt document adapter

Builds a document expert from a [PreTeXt](https://pretextbook.org/) / MathBook-XML open textbook. PreTeXt **tags** its
structure — `<definition>`, `<theorem>`, `<term>` — so the adapter gets exact `defines` (term → its passage) and named
`theorem` statements for free, feeding the uniform strategy table (`define` / `theorem` / count / list).

**Licensing:** these books are **GFDL / CC-BY-SA** (copyleft). The *book source* and the *derived corpus* are therefore
**not committed** here — clone the book and build locally (same policy as the RISC-V manual).

```bash
# Abstract Algebra (Judson, GFDL)
git clone --depth 1 https://github.com/twjudson/aata
AATA_SRC=$PWD/aata/src .venv/bin/python build_expert.py examples/pretext/aata.toml

# A First Course in Linear Algebra (Beezer, GFDL) — definition/theorem-dense
git clone --depth 1 https://github.com/rbeezer/fcla
FCLA_SRC=$PWD/fcla/src .venv/bin/python build_expert.py examples/pretext/fcla.toml
```

Then serve with sgiandubh (`--answer-from-corpus --require-citation`) and try:
`What is a group?` (define) · `State Lagrange's theorem` (theorem) · `What is a vector space?` (define).

The adapter is `pack.adapters.pretext`; adding another book is just another `[adapter] source=`. A different *source
type* (a spec, an arXiv/LaTeXML paper) is a different registered adapter — see `py/pack/adapters/`.
