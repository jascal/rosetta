"""pack.adapters.normrules — a spec's machine-readable normative rules → corpus files (model-free).

Each normative rule becomes one retrievable, citable passage. The structured-source counterpart to pack.answers (which
needs a model): no model, no fieldrun — the rule ids/chapters are the citations. Emits into <out>/:
  rules.txt        "[norm:<name> · <chapter>] <text>"   (grounding --no-split: one passage per rule)
  rules_plain.txt  "<text>"                              (gram)

Faithful port of the former sgiandubh/tools/normrules2package.py.
"""
import json
import os
import re


def to_corpus(norm_rules_path, out, *, model="rosetta-expert-spec"):
    """norm-rules.json → <out>/rules.txt + <out>/rules_plain.txt. Returns (rules_txt, rules_plain, n)."""
    os.makedirs(out, exist_ok=True)
    d = json.load(open(norm_rules_path))
    rules = d["normative_rules"]
    rules_txt = os.path.join(out, "rules.txt")
    rules_plain = os.path.join(out, "rules_plain.txt")
    n = 0
    with open(rules_txt, "w", encoding="utf-8") as ft, open(rules_plain, "w", encoding="utf-8") as fp:
        for r in rules:
            name = r.get("name", "")
            chap = r.get("chapter_name", "")
            text = " ".join(t.get("text", "") for t in r.get("tags", []))
            text = re.sub(r"\s+", " ", text).strip()
            if not text or not name:
                continue
            ft.write(f"[norm:{name} · {chap}] {text}\n")
            fp.write(text + "\n")
            n += 1
    return rules_txt, rules_plain, n
