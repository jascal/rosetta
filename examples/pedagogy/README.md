# examples/pedagogy — teaching templates as a model-free expert

Pedagogy is the third layer of the stack — **content** (bounded experts) / **scope** (which experts are in-bounds, the
syllabus boundary) / **pedagogy** (how to teach). A teaching template is *just another expert's content*: a cited
passage, selected by the same uniform `answer(intent, entity, …)` rule as everything else. No model selects it.

## The template file (`templates.toml`)

```toml
[[template]]
name  = "socratic-tutor"   # the citation handle (pedagogy:socratic-tutor)
style = "socratic"         # ┐ together form the selection ENTITY ("socratic intro")
level = "intro"            # ┘
system = "You are a Socratic tutor for {scope}. … toward {objectives}. … {content}"
```

## Selection (ergo, model-free)

- **Intent** comes from cue words in `ergo/strategy.dl` → `pedagogy`: `tutor, teach, mentor, coach, lesson, tutorial,
  session, advisor, advise, quiz, exam, learn, study, …`.
- **Entity** is `"<style> <level>"`, the bare `name`, or `style` alone — longest match wins.
- So "give me a **socratic** **intro** **tutor**" → intent `pedagogy` + entity `socratic intro` → `socratic-tutor`.

Adding a template is just another `[[template]]` block — it wires itself in (no code, no new ergo rule).

## Placeholder contract (filled by claymore at session start)

The hub (`POST /session` or a `session` object on a chat request) substitutes these before running:

| placeholder | filled with |
|-------------|-------------|
| `{scope}`      | the in-bounds experts, `name (domain), …` |
| `{objectives}` | the session's learning objectives |
| `{content}`    | caller-supplied material (the student's code/problem) |
| `{anything}`   | any key in the session's `variables` |

claymore also **appends a hard guardrail** (teach only from the in-scope expert tools; out-of-scope → say so) and
offers the LLM tools **only for the in-scope experts** — so a tutor cannot teach beyond its corpus. The template need
not restate the guardrail (the hub enforces it regardless).

## Build & serve

```bash
.venv/bin/python build_expert.py examples/pedagogy/expert.toml          # → a model-free pedagogy expert (dim=0)
sgiandubh examples/pedagogy/package <port> --answer-from-corpus         # serve it; claymore lists it as role:"pedagogy"
```
