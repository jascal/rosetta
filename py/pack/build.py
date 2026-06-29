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
import json
import os
import shutil
import subprocess

from . import answers, cover as cover_mod, grounding
from .adapters import normrules


def _fieldrun_bin(explicit=None):
    """Resolve the fieldrun extractor binary explicitly: the `fieldrun` arg (--fieldrun), else $FIELDRUN. No hard-coded
    path / no PATH auto-discovery — the extractor must be named, so a build is reproducible and never picks up a stray binary."""
    fr = explicit or os.environ.get("FIELDRUN")
    if not fr:
        raise ValueError("fieldrun not specified — pass fieldrun=… (--fieldrun) or set $FIELDRUN (no hard-coded default)")
    if not os.path.exists(fr):
        raise FileNotFoundError(f"fieldrun not found at '{fr}'")
    return fr


def _make_corpus(out, text_file, bundle):
    """Scope the cover to the USER's corpus: tokenize `text_file` with the bundle's tokenizer → out/corpus.json, and
    drop out/bundle.tokenizer.json — the inputs the core's cover extraction (idiom_learn) reads. This is the fix for the
    cover-over-the-selected-corpus gap (without it the cover would extract over a generic/absent corpus)."""
    from tokenizers import Tokenizer
    tok_path = bundle + ".tokenizer.json"
    tok = Tokenizer.from_file(tok_path)
    ids = tok.encode(open(text_file, encoding="utf-8").read()).ids
    json.dump({"ids": ids}, open(os.path.join(out, "corpus.json"), "w"))
    shutil.copyfile(tok_path, os.path.join(out, "bundle.tokenizer.json"))
    return len(ids)


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
        fr = _fieldrun_bin(fieldrun)
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

    # 3. cover — the SMART tier (model-distilled only; extracted from the model OVER THE USER'S CORPUS, not asserted)
    if cover:
        if not bundle:
            raise ValueError("cover=True needs a model (bundle) — the cover is extracted from the model")
        cover_corpus = corpus or questions
        if not cover_corpus:
            raise ValueError("cover=True needs a corpus to extract over (corpus= or questions=)")
        ntok = _make_corpus(out, cover_corpus, bundle)            # scope the cover to the selected corpus (the gap fix)
        print(f"[cover] corpus.json: {ntok} tokens from {os.path.basename(cover_corpus)} (scoped to the selected corpus)")
        mpath = cover_mod.build(out, minsupp=minsupp, mindet=mindet)
        print(f"[cover] manifest -> {mpath}")

    # NOTE: no `gram` tier — subsumed by the cover's gated n-grams (CONVERGENCE.md).
    print(f"package -> {out}")
    return out


def build_from_spec(spec_path):
    """Build (and, if [gate] is set, score) an expert from a declarative expert.toml (EXPERTS.md). Output →
    <spec_dir>/package/ (or [build].out). Recognizes the opt-in [reasoning] tier (REASONING.md) but does not wire it."""
    from . import spec as spec_mod
    s = spec_mod.load_spec(spec_path)
    base = os.path.dirname(os.path.abspath(spec_path))
    kw = spec_mod.to_build_kwargs(s, base=base)
    out = (s.get("build", {}) or {}).get("out") or os.path.join(base, "package")
    out = out if os.path.isabs(out) else os.path.join(base, out)
    if s.get("reasoning"):                                        # "rosetta is aware of it" — opt-in, wiring takes care
        print("[reasoning] authored-deductive tier present in spec (opt-in) — recognized; "
              "wiring is a deliberate separate step (see REASONING.md)")
    build_expert(out, **kw)
    _score_if_gated(out, s, base)
    return out


def _score_if_gated(out, spec, base):
    """If [gate] is set and a manifest was produced, grade the package and HARD-FAIL on a miss (EXPERTS.md)."""
    g = spec.get("gate")
    if not g:
        return
    from . import eval as scorer
    mpath = next((p for p in (os.path.join(out, "manifest.json"), os.path.join(out, "package", "manifest.json"))
                  if os.path.exists(p)), None)
    if not mpath:
        print("[scorecard] no manifest (model-free grounding-only build) — gate skipped (see the model-free n-gram open item)")
        return
    holdout, off_domain = _eval_sets(out, spec, base)
    if not holdout:
        print("[scorecard] no holdout derivable (no tokenizer/corpus) — gate skipped")
        return
    sc = scorer.score(mpath, holdout, off_domain)
    scorer.write_scorecard(sc, os.path.join(out, "scorecard.json"))
    print(f"[scorecard] coverage {sc['coverage']:.0%}  precision {sc['precision']:.0%}  "
          f"abstain {sc['abstain']:.0%}  leak {sc['off_domain_leak']:.0%}  (n={sc['holdout_n']})")
    ok, reasons = scorer.gate(sc, min_precision=g.get("min_precision"), max_leak=g.get("max_leak"),
                              benchmarks=spec.get("benchmark"))
    if not ok:
        raise SystemExit("[gate] FAIL — no shippable package: " + "; ".join(reasons))
    print("[gate] PASS")


def _eval_sets(out, spec, base, W=8, frac=0.3):
    """Held-out (ctx, gold) pairs + off-domain contexts for the scorecard, tokenized with the built package's tokenizer.
    NOTE: a held-out *tail* of the same corpus the cover was built on has leakage — proper train/hold isolation (build
    the cover on TRAIN only) is the EXPERTS.md open item; this is the plumbed baseline."""
    tok_path = os.path.join(out, "bundle.tokenizer.json")
    text = (spec.get("corpus", {}) or {}).get("text")
    if not (os.path.exists(tok_path) and text):
        return [], []
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    text = text if os.path.isabs(text) else os.path.join(base, text)
    ids = tok.encode(open(text, encoding="utf-8").read()).ids
    wins = [(tuple(ids[i - W:i]), ids[i]) for i in range(W, len(ids))]
    hold = wins[int(len(wins) * (1 - frac)):]
    od = []
    odf = (spec.get("experiment", {}) or {}).get("off_domain")
    if odf:
        odf = odf if os.path.isabs(odf) else os.path.join(base, odf)
        if os.path.exists(odf):
            oids = tok.encode(open(odf, encoding="utf-8").read()).ids
            od = [tuple(oids[i - W:i]) for i in range(W, len(oids))]
    return hold, od


def main():
    import sys
    if len(sys.argv) == 2 and sys.argv[1].endswith(".toml"):     # rosetta build-expert expert.toml
        build_from_spec(sys.argv[1])
        return
    ap = argparse.ArgumentParser(prog="rosetta build-expert",
                                 description="assemble a deployable bounded-expert package (cover-first); "
                                             "pass an expert.toml for the declarative form")
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
    ap.add_argument("--fieldrun", help="path to the fieldrun extractor binary (required for distill/--cover; "
                    "else set $FIELDRUN). No hard-coded default.")
    a = ap.parse_args()
    build_expert(a.out, corpus=a.corpus, bundle=a.bundle, questions=a.questions, steps=a.steps,
                 citation=a.citation, model=a.model, adapter=a.adapter, adapter_source=a.adapter_source,
                 dim=a.dim, corpus_vectors=a.corpus_vectors, no_split=a.no_split,
                 cover=a.cover, minsupp=a.minsupp, mindet=a.mindet, fieldrun=a.fieldrun)


if __name__ == "__main__":
    main()
