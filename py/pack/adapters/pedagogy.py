"""pack.adapters.pedagogy — teaching/mentoring TEMPLATES as a model-free expert (with ergo-governed selection).

A pedagogy template is a prompt scaffold that LEADS a tutoring/mentoring/advising session (Socratic, quiz, worked
example, advisor, …). Here it is just a citable passage; SELECTING one ("give me a Socratic, intro-level tutor") is the
SAME uniform strategy rule as everything else — intent=pedagogy, entity="<style> <level>" → the template passage. So
pedagogy rides the whole stack: no model, ergo predicates select the template (ergo/strategy.dl pedagogy cues), and the
templates are cited, retrievable, federatable like any expert's content. claymore starts a session by querying this
expert for the template, filling its {scope}/{objectives}/{content} variables, and scoping the tools to the content
experts.

Source = a TOML file of [[template]] entries: {name, style, level, system, opening?, applies_to?/experts?, subject?}.
`system` is the prompt scaffold (it may use {scope}, {objectives}, {content}, … placeholders the caller/hub fills at
session start). Two optional fields extend a template:

- `opening` — a second scaffold that GENERATES the tutor's FIRST message to the learner (the session kickoff: a greeting
  + the first guiding question / exam item). Same placeholders; the hub runs it to produce the opening turn so the tutor
  speaks first instead of waiting for the student.
- `suggest` — a scaffold that, after each tutor turn, GENERATES a few suggested next prompts for the learner to send.
  Same placeholders; the hub runs it (reflecting on the conversation) and shows the list. Authored here, not in the hub.
- `applies_to` / `experts` (list of spoke names) and/or `subject` (free text) — a SCOPE RESTRICTION: which content
  experts this tutor may be applied to. `applies_to` names experts exactly; `subject` is matched (lexically, at the hub)
  against each expert's domain. Empty = the tutor applies to every in-scope expert (the prior behaviour).

Both ride after a `[[TUTOR_META]]` sentinel as a JSON tail on the template passage, so they travel with the template in
ONE citable/federatable passage; the hub splits the tail off and ignores it for everything else.
"""
import json
import tomllib

from .base import Extraction, register

META_MARK = "[[TUTOR_META]]"   # sentinel separating the system scaffold from the JSON tutor-metadata tail


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
        opening = " ".join((t.get("opening") or "").split())
        suggest = " ".join((t.get("suggest") or "").split())
        if not (name and system):
            continue
        # optional scope restriction: which content experts this tutor may be applied to.
        applies = t.get("applies_to") or t.get("experts") or []
        if isinstance(applies, str):
            applies = [applies]
        applies = [a.strip() for a in applies if str(a).strip()]
        subject = " ".join((t.get("subject") or "").split())
        meta = {}
        if opening:
            meta["opening"] = opening
        if suggest:
            meta["suggest"] = suggest
        if applies:
            meta["applies_to"] = applies
        if subject:
            meta["subject"] = subject
        pid = f"{prefix}:{name}"
        facet = " ".join(x for x in (style, level) if x) or name
        # body = the system scaffold; the opener + scope restriction ride after the sentinel as a JSON tail so they
        # travel with the template in one passage (the hub splits it; an old hub just ignores an absent sentinel).
        body = system if not meta else f"{system} {META_MARK} {json.dumps(meta, ensure_ascii=False)}"
        passages.append((f"{pid} · {facet}", body))
        # select by "<style> <level>" (most specific), by name, or by style alone — longest-match wins at serve time
        for key in {k for k in (f"{style} {level}".strip(), name.lower(), style, level) if k}:
            if (key, pid) not in seen:
                seen.add((key, pid))
                answers.append(("pedagogy", key, pid))
    if not passages:
        raise ValueError(f"pedagogy adapter: {source} yielded no usable templates (each needs name + system)")
    return Extraction(passages, answers=answers, citation=citation)
