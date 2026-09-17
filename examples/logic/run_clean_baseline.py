#!/usr/bin/env python
"""examples/logic/run_clean_baseline.py — P1-b: does the model-tier win survive a CLEANED document baseline?

The P1 ablation (run_ablation.py) used the RAW OLP dump as the document tier — a weak baseline (no citation handles,
TOC noise). This gives retrieval a fair shot: it grades a document tier built over logic_kb_clean.txt (clean_kb.py:
filtered + `[OLP §N.N]` section handles) against the SAME testset, side by side with the curated FAQ and the raw
document tier. If clean-document closes the gap, the P1 verdict flips to "a good adapter makes the model tier
redundant"; if the model tier still wins, the verdict is robust.

Run: .venv/bin/python examples/logic/run_clean_baseline.py  ->  examples/logic/scorecard_clean.json
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "py"))
import run_ablation as RA  # noqa: E402  — reuse serve/ask/grade/SUBSETS + the calibrated flags
from pack import answers, grounding  # noqa: E402

RAW_KB = os.path.join(HERE, "logic_kb.txt")
CLEAN_KB = os.path.join(HERE, "logic_kb_clean.txt")
FROZEN_FAQ = os.path.join(HERE, "curated_faq.json")


def _doc_pkg(work, name, kb):
    """A document-only package (empty curated index + grounding over `kb`)."""
    d = os.path.join(work, name)
    os.makedirs(d, exist_ok=True)
    answers.empty_index(d, model="sgiandubh-open-logic")
    n, v, _ = grounding.build(kb, d, dim=300, corpus_vectors=False, no_split=True)   # clean = one citable passage/line
    return d, n, v


def _curated_pkg(work):
    d = os.path.join(work, "curated")
    os.makedirs(d, exist_ok=True)
    shutil.copyfile(FROZEN_FAQ, os.path.join(d, "index.json"))
    return d


def main():
    if not os.path.exists(BIN := RA.BIN):
        sys.exit(f"sgiandubh binary not found: {BIN} (set $SGIANDUBH)")
    if not os.path.exists(CLEAN_KB):
        import clean_kb
        clean_kb.clean()
        print(f"[clean] generated {CLEAN_KB}")
    rows = [json.loads(ln) for ln in open(RA.TESTSET, encoding="utf-8") if ln.strip()]
    work = tempfile.mkdtemp(prefix="logic_clean_")
    try:
        variants = {"curated": _curated_pkg(work)}
        praw, nraw, _ = _doc_pkg(work, "doc_raw", RAW_KB)
        pcln, ncln, _ = _doc_pkg(work, "doc_clean", CLEAN_KB)
        variants["doc_raw"] = praw
        variants["doc_clean"] = pcln
        print(f"[build] doc_raw passages={nraw}  doc_clean passages={ncln}")
        grids = {}
        for tag, cite in [("require_citation_on", True), ("require_citation_off", False)]:
            grids[tag] = {}
            for i, (name, pkg) in enumerate(variants.items()):
                p = RA.serve(pkg, 8155 + i, cite)
                try:
                    grids[tag][name] = RA.grade(rows, 8155 + i)
                finally:
                    p.kill()
        json.dump({"domain": "Open Logic Project", "question": "does the model-tier win survive a cleaned doc baseline?",
                   "grids": grids, "n": len(rows)},
                  open(os.path.join(HERE, "scorecard_clean.json"), "w"), indent=1)
        for tag in grids:
            print(f"\n{tag}")
            print(f"  {'variant':<12}" + "".join(f"{s.split('_')[1][:7]:>9}" for s in RA.SUBSETS) + "   overall")
            for name in variants:
                g = grids[tag][name]
                cells = []
                for s in RA.SUBSETS:
                    if s == "D_offdomain":
                        cells.append(f"{g[s]['answered']}/{g[s]['n']}")
                    else:
                        cells.append(f"{g[s]['answered']}/{g[s]['n']}·{g[s]['topical']}t")
                ind_a = sum(g[s]["answered"] for s in RA.SUBSETS[:3])
                ind_n = sum(g[s]["n"] for s in RA.SUBSETS[:3])
                ind_t = sum(g[s]["topical"] for s in RA.SUBSETS[:3])
                off = g["D_offdomain"]
                print(f"  {name:<12}" + "".join(f"{c:>9}" for c in cells) +
                      f"   rec {ind_a}/{ind_n} top {ind_t} leak {off['answered']}/{off['n']}")
        print("\n(topical = lenient any-key match; N/N·Kt = answered/n · K topical)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
