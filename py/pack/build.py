"""pack.build — build_expert: the single entry that assembles a deployable bounded-expert package.

Replaces sgiandubh/tools/build_expert(s).sh. Composes the pack modules COVER-FIRST (CONVERGENCE.md), under the
fast/accurate/smart constraints:
  cover (smart: causal idioms + gated n-grams) > curated answers (gated FAQ) > grounding (CITATION; retrieval
  hard-gated at the runtime) > abstain.
No bare `gram` tier — it's subsumed by the cover's gated n-grams (shipping it would trade accuracy for cheap coverage).
Emits the *package*; does NOT build the runtime binary — the builder produces the package, the package is the only
interface to the thin sgiandubh runtime.
"""
import argparse
import os
import shutil
import subprocess

from . import answers, cover as cover_mod, grounding
from .adapters import normrules


def _fieldrun_bin():
    return shutil.which("fieldrun") or os.path.expanduser("~/code/fieldrun/target/release/fieldrun")


def build_expert(out, *, corpus=None, bundle=None, questions=None, steps=256, citation="",
                 model="rosetta-expert", adapter=None, adapter_source=None,
                 dim=300, corpus_vectors=False, no_split=False,
                 cover=False, minsupp=3, mindet=1.0, fieldrun=None):
    """Assemble a package at `out`. Three shapes:
      * model-free adapter  (adapter=…, adapter_source=…)  → citable passages + grounding, no curated items, no cover.
      * model-distilled     (bundle=…, questions=…)        → curated answers (+ optional cover with cover=True).
      * corpus-only         (corpus=…)                      → grounding + empty index.
    Returns the package dir."""
    os.makedirs(out, exist_ok=True)
    ground_corpus, ground_no_split = corpus, no_split

    # 1. items + the grounding corpus
    if adapter == "normrules":
        if not adapter_source:
            raise ValueError("adapter=normrules needs adapter_source (norm-rules.json)")
        rules_txt, _rules_plain, n = normrules.to_corpus(adapter_source, out, model=model)
        answers.empty_index(out, model=model)                 # model-free: served by cover/retrieval, no curated items
        ground_corpus, ground_no_split = rules_txt, True      # one passage per rule
        print(f"[adapter:normrules] {n} rules -> {os.path.basename(rules_txt)}")
    elif bundle and questions:
        export = os.path.join(out, "_export")
        fr = fieldrun or _fieldrun_bin()
        if not os.path.exists(fr):
            raise FileNotFoundError(f"fieldrun not found at '{fr}' — set fieldrun=…")
        print(f"[distill] fieldrun --export-logic-corpus (--steps {steps}) — the heavy step (EOS-stopping, avoids truncation)")
        subprocess.run([fr, "--bundle", bundle, "--export-logic-corpus", questions,
                        "--steps", str(steps), "--out", export], check=True)
        items = answers.from_export(out, questions, export, citation=citation, model=model)
        print(f"[answers] {len(items)} curated items")
        ground_corpus = corpus or questions
    else:
        answers.empty_index(out, model=model)

    # 2. grounding — CITATION-first (the runtime hard-gates retrieval-as-answer; see CONVERGENCE.md)
    if ground_corpus:
        npass, nvec, k = grounding.build(ground_corpus, out, dim=dim, corpus_vectors=corpus_vectors,
                                         no_split=ground_no_split)
        print(f"[grounding] {npass} passages, {nvec} vectors x {k}d (citation-first)")

    # 3. cover — the SMART tier (model-distilled only; extracted from the model, not asserted)
    if cover:
        if not bundle:
            raise ValueError("cover=True needs a model (bundle) — the cover is extracted from the model")
        mpath = cover_mod.build(out, minsupp=minsupp, mindet=mindet)
        print(f"[cover] manifest -> {mpath}")

    # NOTE: no `gram` tier — subsumed by the cover's gated n-grams (CONVERGENCE.md).
    print(f"package -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(prog="rosetta build-expert",
                                 description="assemble a deployable bounded-expert package (cover-first)")
    ap.add_argument("out")
    ap.add_argument("--corpus", help="grounding corpus (knowledge passages)")
    ap.add_argument("--bundle", help="model bundle stem (enables distilled answers + --cover)")
    ap.add_argument("--questions", help="question file for distilled curated answers")
    ap.add_argument("--steps", type=int, default=256, help="distill depth (EOS-stopping; 256 avoids truncation)")
    ap.add_argument("--citation", default="")
    ap.add_argument("--model", default="rosetta-expert")
    ap.add_argument("--adapter", choices=["normrules"], help="model-free structured-source adapter")
    ap.add_argument("--adapter-source", help="the adapter's source file (e.g. norm-rules.json)")
    ap.add_argument("--dim", type=int, default=300, help="grounding embedding dim (0 = lexical only)")
    ap.add_argument("--corpus-vectors", action="store_true", help="PPMI+SVD over the corpus instead of GloVe")
    ap.add_argument("--no-split", action="store_true")
    ap.add_argument("--cover", action="store_true", help="extract the cover, the smart tier (needs --bundle)")
    ap.add_argument("--minsupp", type=int, default=3)
    ap.add_argument("--mindet", type=float, default=1.0)
    ap.add_argument("--fieldrun")
    a = ap.parse_args()
    build_expert(a.out, corpus=a.corpus, bundle=a.bundle, questions=a.questions, steps=a.steps,
                 citation=a.citation, model=a.model, adapter=a.adapter, adapter_source=a.adapter_source,
                 dim=a.dim, corpus_vectors=a.corpus_vectors, no_split=a.no_split,
                 cover=a.cover, minsupp=a.minsupp, mindet=a.mindet, fieldrun=a.fieldrun)


if __name__ == "__main__":
    main()
