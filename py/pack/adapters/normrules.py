"""pack.adapters.normrules — a spec's machine-readable normative rules → an Extraction (model-free).

Each normative rule becomes one retrievable, citable passage (the rule id/chapter is the citation), and the spec's
`insn:X[]` cross-references become the count/list inventory items. The structured-source counterpart to pack.answers
(which needs a model): no model, no fieldrun. Source = RISC-V's norm-rules.json. A document-adapter instance.
"""
import json
import os
import re

from .base import Extraction, register

_INSN = re.compile(r"insn:([A-Za-z0-9._]+)\[\]")                   # the spec's insn:X[] instruction cross-reference


def _rules(norm_rules_path):
    try:
        d = json.load(open(norm_rules_path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"normrules adapter: cannot read {norm_rules_path} — {e}")
    if not isinstance(d, dict) or not isinstance(d.get("normative_rules"), list):
        raise ValueError(f"normrules adapter: {norm_rules_path} has no \"normative_rules\" list "
                         f"(top-level keys: {sorted(d)[:8] if isinstance(d, dict) else type(d).__name__})")
    for r in d["normative_rules"]:
        name = r.get("name", "")
        chap = r.get("chapter_name", "")
        text = re.sub(r"\s+", " ", " ".join(t.get("text", "") for t in r.get("tags", []))).strip()
        if text and name:
            yield name, chap, text


@register("normrules")
def adapt(source, *, citation="RISC-V ISA Manual (CC BY 4.0)", **_):
    """norm-rules.json → Extraction: one passage per rule + the insn:X[] inventory (for count/list)."""
    passages, items = [], set()
    for name, chap, text in _rules(source):
        passages.append((f"norm:{name} · {chap}", text))
        for insn in _INSN.findall(text):
            items.add((insn.lower(), chap or "unknown"))
    if not passages:
        raise ValueError(f"normrules adapter: {source} yielded 0 usable rules "
                         "(each needs a \"name\" and non-empty \"tags[].text\")")
    return Extraction(passages, items=sorted(items), citation=citation)


def to_corpus(norm_rules_path, out, *, model="rosetta-expert-spec"):
    """Back-compat: write <out>/rules.txt + rules_plain.txt. Returns (rules_txt, rules_plain, n)."""
    os.makedirs(out, exist_ok=True)
    rules_txt, rules_plain = os.path.join(out, "rules.txt"), os.path.join(out, "rules_plain.txt")
    n = 0
    with open(rules_txt, "w", encoding="utf-8") as ft, open(rules_plain, "w", encoding="utf-8") as fp:
        for name, chap, text in _rules(norm_rules_path):
            ft.write(f"[norm:{name} · {chap}] {text}\n")
            fp.write(text + "\n")
            n += 1
    if n == 0:
        raise ValueError(f"normrules adapter: {norm_rules_path} yielded 0 usable rules "
                         "(each needs a \"name\" and non-empty \"tags[].text\")")
    return rules_txt, rules_plain, n
