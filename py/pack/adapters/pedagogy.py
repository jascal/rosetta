"""pack.adapters.pedagogy — teaching/mentoring TEMPLATES as a model-free expert (with ergo-governed selection).

A pedagogy template is a prompt scaffold that LEADS a tutoring/mentoring/advising session (Socratic, quiz, worked
example, advisor, …). Here it is just a citable passage; SELECTING one ("give me a Socratic, intro-level tutor") is the
SAME uniform strategy rule as everything else — intent=pedagogy, entity="<style> <level>" → the template passage. So
pedagogy rides the whole stack: no model, ergo predicates select the template (ergo/strategy.dl pedagogy cues), and the
templates are cited, retrievable, federatable like any expert's content. claymore starts a session by querying this
expert for the template, filling its {scope}/{objectives}/{content} variables, and scoping the tools to the content
experts.

Source = a TOML file of [[template]] entries: {name, style, level, system}. `system` is the prompt scaffold (it may use
{scope}, {objectives}, {content}, … placeholders the caller/hub fills at session start).
"""
import tomllib

from .base import Extraction, register


@register("pedagogy")
def adapt(source, *, prefix="pedagogy", citation="pedagogy templates", **_):
    with open(source, "rb") as f:
        spec = tomllib.load(f)
    templates = spec.get("template", [])
    if not templates:
        raise ValueError(f"pedagogy adapter: {source} has no [[template]] entries")
    passages, answers, seen = [], [], set()
    for t in templates:
        name = (t.get("name") or "").strip()
        style = (t.get("style") or "").strip().lower()
        level = (t.get("level") or "").strip().lower()
        system = " ".join((t.get("system") or "").split())
        if not (name and system):
            continue
        pid = f"{prefix}:{name}"
        facet = " ".join(x for x in (style, level) if x) or name
        passages.append((f"{pid} · {facet}", system))
        # select by "<style> <level>" (most specific), by name, or by style alone — longest-match wins at serve time
        for key in {k for k in (f"{style} {level}".strip(), name.lower(), style, level) if k}:
            if (key, pid) not in seen:
                seen.add((key, pid))
                answers.append(("pedagogy", key, pid))
    if not passages:
        raise ValueError(f"pedagogy adapter: {source} yielded no usable templates (each needs name + system)")
    return Extraction(passages, answers=answers, citation=citation)
