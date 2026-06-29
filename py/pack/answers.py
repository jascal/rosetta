"""pack.answers — fieldrun's emitted Datalog decisions → curated Q&A items (index.json + per-item facts).

The model-distilled curated tier: fieldrun extracts a corpus into per-decision `.dl` programs; this assembles them into
`index.json` + per-item `candidate.facts`/`contrib.facts` (the decision the runtime's semiring decode re-derives). An
item's ANSWER is the concatenated per-step predicted text (faithful — straight from the extraction).

Faithful port of the former sgiandubh/tools/dl2package.py, as importable `from_export` / `from_manifest`.
"""
import glob
import json
import os
import re

CAND = re.compile(r'candidate\((\d+)\)')
CONTRIB = re.compile(r'contrib\("([^"]+)",\s*(\d+),\s*(-?[\d.eE+]+)\)')
PRED = re.compile(r'model predicts:\s*"([^"]*)"\s*\[(\d+)\]')
ROUTE = re.compile(r'route:\s*(\S+)\s+margin:\s*([+-]?[\d.eE]+)')
STEP = re.compile(r'(\d+)\.dl$')
PROMPT = re.compile(r'p(\d+)_\d+\.dl$')

# fieldrun writes token text Rust-debug-escaped ({:?}); undo those escapes (real Unicode left untouched).
_ESC = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\', '0': '\0', "'": "'"}


def _unescape(s):
    return re.sub(r'\\(.)', lambda m: _ESC.get(m.group(1), m.group(1)), s)


def _parse_dl(path):
    txt = open(path, encoding="utf-8").read()
    cands = [int(x) for x in CAND.findall(txt)]
    contribs = [(b, int(t), float(w)) for b, t, w in CONTRIB.findall(txt)]
    m = PRED.search(txt)
    rm = ROUTE.search(txt)
    route = rm.group(1) if rm else "—"
    margin = float(rm.group(2)) if rm else None
    return cands, contribs, (_unescape(m.group(1)) if m else ""), route, margin


def _build_item(out, item_id, query, citation, dl_files):
    steps = sorted(dl_files, key=lambda p: int(STEP.search(p).group(1)) if STEP.search(p) else 0)
    if not steps:
        return None
    answer = ""
    first_cands, first_contribs = [], []
    routes, margins = [], []
    for i, s in enumerate(steps):
        cands, contribs, ptext, route, margin = _parse_dl(s)
        answer += ptext
        routes.append(route)
        if margin is not None:
            margins.append(margin)
        if i == 0:
            first_cands, first_contribs = cands, contribs
    fdir = os.path.join(out, f"facts_{item_id}")
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, "candidate.facts"), "w") as f:
        for c in first_cands:
            f.write(f"{c}\n")
    with open(os.path.join(fdir, "contrib.facts"), "w") as f:
        for b, t, w in first_contribs:
            f.write(f"{b}\t{t}\t{w}\n")
    item = {"id": item_id, "query": query, "answer": answer.strip(), "citation": citation,
            "facts": os.path.basename(fdir)}
    order = {"RETRIEVED": 0, "SELECTED": 1, "COMPOSED": 2}        # provenance summary (worst tier + thinnest margin)
    tiered = [r for r in routes if r in order]
    if tiered:
        item["route"] = max(tiered, key=lambda r: order[r])
        item["n_composed"] = sum(1 for r in routes if r == "COMPOSED")
        item["n_steps"] = len(routes)
    if margins:
        item["margin"] = round(min(margins), 3)
    return item


def _cite_for(cite, idx):
    if cite is None:
        return ""
    e = cite[idx] if isinstance(cite, list) and idx < len(cite) else (cite.get(str(idx), "") if isinstance(cite, dict) else "")
    if isinstance(e, dict):
        return e.get("citation") or e.get("section") or ""
    return e or ""


def from_export(out, corpus, dl_dir, *, citation="", cite=None, model="sgiandubh"):
    """Corpus mode: one item per prompt, decisions = that prompt's p{NNNNN}_{SS}.dl. Writes <out>/index.json + facts_*/."""
    os.makedirs(out, exist_ok=True)
    lines = [ln.strip() for ln in open(corpus, encoding="utf-8") if ln.strip()]  # matches fieldrun's non-empty filter
    citemap = json.load(open(cite)) if cite else None
    groups = {}
    for f in glob.glob(os.path.join(dl_dir, "*.dl")):
        m = PROMPT.search(os.path.basename(f))
        if m:
            groups.setdefault(int(m.group(1)), []).append(f)
    items = []
    for idx in sorted(groups):
        query = lines[idx] if idx < len(lines) else f"[prompt {idx}]"
        cit = _cite_for(citemap, idx) or citation
        item = _build_item(out, f"p{idx:05}", query, cit, groups[idx])
        if item:
            items.append(item)
    json.dump({"model": model, "items": items},
              open(os.path.join(out, "index.json"), "w"), indent=1, ensure_ascii=False)
    return items


def from_manifest(out, manifest, *, model="sgiandubh"):
    """Explicit mode: a JSON list of {id, query, citation, dl(glob)}. Writes <out>/index.json + facts_*/."""
    os.makedirs(out, exist_ok=True)
    man = json.load(open(manifest))
    model = man.get("model", model)
    items = []
    for it in man["items"]:
        item = _build_item(out, it["id"], it["query"], it.get("citation", ""), glob.glob(it["dl"]))
        if item:
            items.append(item)
    json.dump({"model": model, "items": items},
              open(os.path.join(out, "index.json"), "w"), indent=1, ensure_ascii=False)
    return items


def empty_index(out, *, model="sgiandubh"):
    """A model-free expert has no distilled items — write an empty index.json (served by retrieval)."""
    os.makedirs(out, exist_ok=True)
    json.dump({"model": model, "items": []},
              open(os.path.join(out, "index.json"), "w"), indent=1, ensure_ascii=False)
    return []
