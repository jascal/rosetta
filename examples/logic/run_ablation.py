#!/usr/bin/env python
"""examples/logic/run_ablation.py — the decisive "does the model-distilled tier earn its place?" experiment.

Reproducible experiment (EXPERTS.md): builds the logic expert from the committed distill export (examples/logic/
package/_export, 28 questions × ~251 decode steps — no fieldrun/model needed at replay time) + the raw OLP dump
(logic_kb.txt), composes THREE ablation variants that differ ONLY in which tiers are present, serves each through the
real sgiandubh binary with IDENTICAL flags, and grades one 50-row testset — so the delta between variants IS the
measured value of the model-distilled curated tier.

  i_curated   = curated FAQ only  (index.json + facts_*; no knowledge.tsv)
  ii_document = retrieval only     (empty index.json; knowledge.tsv + wordvec over the raw dump)
  iii_both    = the shipped cascade (curated > retrieval)

Two citation policies are run: --require-citation ON (the shipped bounded-expert default) and OFF (raw retrieval
capability). A --tau sweep on the curated tier measures whether it can be calibrated to ABSTAIN on out-of-FAQ
questions instead of returning a confident nearest-neighbour answer.

Run:  FIELDRUN unused.  SGIANDUBH=../sgiandubh/build/sgiandubh  .venv/bin/python examples/logic/run_ablation.py
Writes: examples/logic/scorecard.json   (the measured result; committed alongside the testset)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "py"))
from pack import answers, grounding  # noqa: E402

BIN = os.environ.get("SGIANDUBH", os.path.join(ROOT, "..", "sgiandubh", "build", "sgiandubh"))
EXPORT = os.path.join(HERE, "package", "_export")             # fresh distill (needs fieldrun+gemma; gitignored, 55MB)
FROZEN_FAQ = os.path.join(HERE, "curated_faq.json")           # committed fallback: the distilled FAQ, frozen (~33KB)
QUESTIONS = os.path.join(HERE, "logic_questions.txt")
KB = os.path.join(HERE, "logic_kb.txt")
TESTSET = os.path.join(HERE, "testset.jsonl")
COS, MARGIN, COV = 0.70, 0.30, 0.6                     # repo-calibrated retrieval defaults (score_retrieval/server.cpp)
SUBSETS = ["A_verbatim", "B_paraphrase", "C_heldout", "D_offdomain"]


def build_variants(work):
    full = os.path.join(work, "full")
    os.makedirs(full, exist_ok=True)
    if os.path.isdir(EXPORT):                                  # fresh distill present → rebuild the curated tier from it
        items = answers.from_export(full, QUESTIONS, EXPORT,
                                    citation="Open Logic Project (CC BY 4.0)", model="sgiandubh-open-logic")
        print(f"[curated] rebuilt from {EXPORT}")
    else:                                                      # fall back to the committed frozen FAQ (self-contained)
        shutil.copyfile(FROZEN_FAQ, os.path.join(full, "index.json"))
        items = json.load(open(FROZEN_FAQ))["items"]
        print(f"[curated] frozen FAQ ({FROZEN_FAQ}) — no fieldrun/gemma needed")
    npass, nvec, _ = grounding.build(KB, full, dim=300, corpus_vectors=False, no_split=False)
    both = os.path.join(work, "iii_both")
    shutil.copytree(full, both)
    doc = os.path.join(work, "ii_document")
    shutil.copytree(full, doc)
    d = json.load(open(os.path.join(doc, "index.json")))
    d["items"] = []
    json.dump(d, open(os.path.join(doc, "index.json"), "w"))
    cur = os.path.join(work, "i_curated")
    shutil.copytree(full, cur)
    for f in ("knowledge.tsv", "wordvec.txt"):
        p = os.path.join(cur, f)
        if os.path.exists(p):
            os.remove(p)
    return {"i_curated": cur, "ii_document": doc, "iii_both": both}, len(items), npass, nvec


def serve(pkg, port, require_cite, tau=None):
    cmd = [BIN, pkg, str(port), "--answer-from-corpus", "--answer-cov", str(COV),
           "--answer-cos", str(COS), "--answer-margin", str(MARGIN)]
    if require_cite:
        cmd.append("--require-citation")
    if tau is not None:
        cmd += ["--tau", str(tau)]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return p
        except Exception:
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("sgiandubh did not come up")


def ask(port, q):
    b = json.dumps({"messages": [{"role": "user", "content": q}],
                    "response_format": {"type": "json_object"}}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", b, {"content-type": "application/json"}), timeout=20)
    d = json.loads(json.load(r)["choices"][0]["message"]["content"])
    return d.get("answer", ""), d.get("kind", ""), d.get("citation_id", "")


def grade(rows, port):
    subs = {s: {"n": 0, "answered": 0, "topical": 0, "distilled": 0, "retrieved": 0} for s in SUBSETS}
    for r in rows:
        ans, kind, _ = ask(port, r["q"])
        answered = kind != "abstain" and not ans.startswith("That isn't")
        s = subs[r["subset"]]
        s["n"] += 1
        if answered:
            s["answered"] += 1
            s["topical"] += bool(not r["key"] or any(k.lower() in ans.lower() for k in r["key"]))
            s[kind] = s.get(kind, 0) + 1
    return subs


def main():
    if not os.path.exists(BIN):
        sys.exit(f"sgiandubh binary not found: {BIN} (set $SGIANDUBH). Build it: (cd ../sgiandubh && ./build.sh)")
    rows = [json.loads(ln) for ln in open(TESTSET, encoding="utf-8") if ln.strip()]
    work = tempfile.mkdtemp(prefix="logic_abl_")
    try:
        pkgs, nitems, npass, nvec = build_variants(work)
        print(f"[build] curated items={nitems}  knowledge passages={npass}  vectors={nvec}")
        grids = {}
        for tag, cite in [("require_citation_on", True), ("require_citation_off", False)]:
            grids[tag] = {}
            for i, (name, pkg) in enumerate(pkgs.items()):
                p = serve(pkg, 8155 + i, cite)
                try:
                    grids[tag][name] = grade(rows, 8155 + i)
                finally:
                    p.kill()
        tau_sweep = {}
        for ti, tau in enumerate([0.25, 0.34, 0.40, 0.45, 0.50, 0.60]):
            p = serve(pkgs["i_curated"], 8165 + ti, True, tau=tau)
            try:
                subs = grade(rows, 8165 + ti)
            finally:
                p.kill()
            tau_sweep[f"{tau:.2f}"] = {s: [subs[s]["answered"], subs[s]["n"]] for s in SUBSETS}
        sc = {"domain": "Open Logic Project", "model": "gemma-4-e4b-it (distilled)",
              "testset": os.path.basename(TESTSET), "n": len(rows),
              "flags": {"answer_cos": COS, "answer_margin": MARGIN},
              "grids": grids, "tau_sweep_curated_require_citation": tau_sweep,
              "note": "topical = lenient any-key match; see EXPERTS.md for the manual-read correction (verbatim ~9/10 "
                      "genuine, held-out ~2-3/12 genuine — the rest are confident nearest-neighbour matches)."}
        json.dump(sc, open(os.path.join(HERE, "scorecard.json"), "w"), indent=1)
        print(f"[scorecard] -> {os.path.join(HERE, 'scorecard.json')}")
        for tag in grids:
            print(f"\n{tag}")
            for name in pkgs:
                g = grids[tag][name]
                ind_a = sum(g[s]["answered"] for s in SUBSETS[:3])
                ind_n = sum(g[s]["n"] for s in SUBSETS[:3])
                ind_t = sum(g[s]["topical"] for s in SUBSETS[:3])
                d = sum(g[s]["distilled"] for s in SUBSETS[:3])
                rr = sum(g[s]["retrieved"] for s in SUBSETS[:3])
                off = g["D_offdomain"]
                print(f"  {name:<12} recall {ind_a}/{ind_n}  topical {ind_t}/{ind_a}  "
                      f"[distilled {d}/retrieved {rr}]  off-domain leak {off['answered']}/{off['n']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
