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


def _resolve_rules(spec_rules):
    """Resolve a reasoning-rules ref to a .dl path: an explicit path, or 'ergo:<name>' → ../ergo/<name>.dl."""
    if spec_rules and os.path.exists(spec_rules):
        return spec_rules
    name = (spec_rules or "ergo:aggregate").split(":")[-1]
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # rosetta repo root
    cand = os.path.join(os.path.dirname(repo), "ergo", name + ".dl")                      # ../ergo/<name>.dl
    if not os.path.exists(cand):
        raise FileNotFoundError(f"reasoning rules not found: {spec_rules!r} (looked for {cand})")
    return cand


def _concat_corpus(out, *parts):
    """Concatenate corpus files (each `[id · chapter] text` one-per-line) into out/_corpus_combined.txt. Used to add
    a prose corpus alongside the rules corpus — both are no-split citable passages, so this is a straight append."""
    combined = os.path.join(out, "_corpus_combined.txt")
    with open(combined, "w", encoding="utf-8") as w:
        for src in parts:
            if not (src and os.path.exists(src)):
                continue
            text = open(src, encoding="utf-8").read()
            w.write(text if text.endswith("\n") else text + "\n")
    return combined


def build_expert(out, *, corpus=None, prose=None, bundle=None, questions=None, steps=256, citation="",
                 model="rosetta-expert", adapter=None, adapter_source=None, adapter_opts=None, documents=None,
                 dim=300, corpus_vectors=False, no_split=False,
                 cover=False, minsupp=3, mindet=1.0, fieldrun=None,
                 inventory=False, inventory_label="instruction", reasoning_rules=None):
    """Assemble a package at `out`. Three shapes:
      * document adapter    (adapter=…, adapter_source=…)  → an Extraction (passages + defines/statements/items) → a
                                                             grounding corpus + the uniform strategy table. No cover.
      * model-distilled     (bundle=…, questions=…)        → curated answers (+ optional cover with cover=True).
      * corpus-only         (corpus=…)                      → grounding + empty index.
    Any document SOURCE (spec, PreTeXt book, arXiv/LaTeXML paper, …) is just a registered adapter — see pack.adapters.
    Returns the package dir."""
    os.makedirs(out, exist_ok=True)
    ground_corpus, ground_no_split = corpus, no_split
    extraction = None

    # 1. the grounding corpus — from N document adapters (any registered sources), a distilled model, or a raw corpus.
    # `documents` (N docs of M adapter types) is the general form; a single [adapter] is the one-document shorthand.
    docs = documents or ([{"adapter": adapter, "source": adapter_source, "opts": adapter_opts or {}}] if adapter else [])
    if docs:
        from . import adapters as _adp
        exts, failed = [], []
        for i, d in enumerate(docs):
            fn = _adp.get(d["adapter"])               # unknown adapter NAME → hard fail (config error, not a bad doc)
            try:                                       # a bad SOURCE (missing/unparseable) at N=100 skips, doesn't abort
                if not d.get("source"):
                    raise ValueError("no source path")
                ext = fn(d["source"], **(d.get("opts") or {}))
            except Exception as e:                     # noqa: BLE001 — adapters raise varied errors; isolate the doc
                failed.append(f"#{i} (adapter={d['adapter']!r}, source={d.get('source')!r}): {e}")
                print(f"[document {i + 1}/{len(docs)}: {d['adapter']}] SKIPPED — {e}")
                continue
            print(f"[document {i + 1}/{len(docs)}: {d['adapter']}] {ext.summary()}")
            exts.append(ext)
        if not exts:
            raise ValueError("no documents could be built:\n  " + "\n  ".join(failed))
        if failed:
            print(f"[documents] WARNING: skipped {len(failed)}/{len(docs)} document(s) that failed to build")
        extraction = exts[0] if len(exts) == 1 else _adp.Extraction.merge(exts)
        ground_corpus = extraction.write_corpus(os.path.join(out, "corpus.txt"))
        ground_no_split = True
        answers.empty_index(out, model=model)                 # model-free: served by retrieval/strategy, no curated items
        if len(exts) > 1:
            print(f"[documents] merged {len(exts)} → {extraction.summary()}")
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

    # 1a. extra PROSE passages (riscv_prose adapter): definitional/conceptual paragraphs the rules corpus lacks, so
    # "what is a hart?" retrieves a definition instead of abstaining. Same no-split citable shape → straight concat.
    if prose:
        ground_corpus = _concat_corpus(out, ground_corpus, prose)
        ground_no_split = True
        print(f"[prose] +{sum(1 for _ in open(prose, encoding='utf-8'))} passages from {os.path.basename(prose)} "
              f"into the grounding corpus")

    # 1b. authored reasoning → the UNIFORM strategy table (REASONING.md). Gather the strategy facts from the adapter's
    # Extraction (defines/statements/items), or — for the committed-corpus path — derive them (inventory from the
    # corpus when [reasoning] is on; defines from prose). Materialize count/list aggregates (ergo, souffle, build-time)
    # into cited passages, then emit strategy.tsv. Build-time only; the runtime stays engine-free.
    if ground_corpus:
        from . import reasoning
        items = list(extraction.items) if (extraction and extraction.items) else \
            (reasoning.instruction_inventory(ground_corpus) if inventory else [])
        defines = list(extraction.defines) if (extraction and extraction.defines) else \
            (reasoning.extract_defines(prose) if prose else [])
        statements = list(extraction.statements) if extraction else []
        gen_answers = list(extraction.answers) if extraction else []   # generic strategy rows (any intent, e.g. pedagogy)
        mat = None
        if items:                                            # count/list aggregates → cited passages appended to corpus
            mat = reasoning.materialize(items, _resolve_rules(reasoning_rules))
            aug = os.path.join(out, "_corpus_with_inventory.txt")
            shutil.copyfile(ground_corpus, aug)
            with open(aug, "a", encoding="utf-8") as f:
                for sec, text in reasoning.inventory_passages(mat, label=inventory_label):
                    f.write(f"[{sec}] {text}\n")
            ground_corpus = aug
            print(f"[reasoning] inventory: {mat['total']} distinct {inventory_label}s across {len(mat['groups'])} groups")
        if mat or defines or statements or gen_answers:
            sdl = _resolve_rules("ergo:strategy")
            _t, ncue, nans = reasoning.strategy_tables(sdl, os.path.join(out, "strategy.tsv"), mat=mat,
                                                       label=inventory_label, defines=defines, theorems=statements,
                                                       answers=gen_answers)
            print(f"[reasoning] strategy: {ncue} cues, {nans} answer rows "
                  f"({len(defines)} define, {len(statements)} theorem, {len(gen_answers)} other) → strategy.tsv")

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


def _card_passage(card, target):
    """A document card → one citable CATALOG passage. The citation handle encodes (target expert, document) so a
    librarian answer points INTO the content expert: cite "lib:<target>:<handle>" → resolve via that expert's spoke."""
    handle, title = card.get("handle", ""), (card.get("title") or card.get("handle") or "document")
    summary, sections = card.get("summary", ""), card.get("sections") or []
    cid = f"lib:{target}:{handle}" if target else f"lib:{handle}"
    text = f"{title}. {summary}".strip()
    if sections:
        text += " Sections: " + "; ".join(sections) + "."
    return f"{cid} · {title[:70]}", _re_ws(text)


def _re_ws(s):
    return " ".join(s.split())


def build_librarian(out, *, documents, target="", dim=300, model="rosetta-librarian", extra_cards=None, label="document"):
    """Build a LIBRARIAN: a model-free CATALOG expert over a document collection. Runs each document's adapter, takes its
    card (or synthesizes one from its passages), and grounds the cards as citable passages — so "which document covers
    X?" retrieves a card whose handle (lib:<target>:<doc>) points INTO the content expert (`target`). The catalog is
    ALSO an inventory, so "how many documents / list the collection" route through ergo's aggregate.dl + strategy.dl
    (the same authored rules as any expert). `extra_cards` adds EXPERT-level cards (cross-hub granularity). No model."""
    from . import adapters as _adp, reasoning
    os.makedirs(out, exist_ok=True)
    cards = list(extra_cards or [])
    for i, d in enumerate(documents):
        fn = _adp.get(d["adapter"])
        try:
            ext = fn(d["source"], **(d.get("opts") or {}))
        except Exception as e:                                    # noqa: BLE001 — one bad doc skips, doesn't abort
            print(f"[librarian {i + 1}/{len(documents)}: {d['adapter']}] SKIPPED — {e}")
            continue
        if ext.cards:
            cards.extend(ext.cards)
        else:                                                    # adapter has no native card → synthesize from structure
            handle = (d.get("opts") or {}).get("prefix") or f"doc{i}"
            facets = []
            for sec, _t in ext.passages:                         # distinct section facets (after the "·")
                f = sec.split("·", 1)[1].strip() if "·" in sec else ""
                if f and f not in facets:
                    facets.append(f)
            cards.append({"handle": handle, "title": handle,
                          "summary": ext.passages[0][1][:300] if ext.passages else "", "sections": facets[:12]})
    if not cards:
        raise ValueError("librarian: no document cards could be built")
    catalog = os.path.join(out, "catalog.txt")
    with open(catalog, "w", encoding="utf-8") as f:
        for c in cards:
            sec, text = _card_passage(c, target)
            f.write(f"[{sec}] {text}\n")
    # the catalog IS an inventory → count/list via ergo aggregate.dl + strategy.dl (same authored rules as any expert),
    # so "how many <label>s / list the <label>s" route declaratively. group = label+"s" so the list intent's entity
    # ("documents") matches the natural query.
    items = [(c.get("title") or c.get("handle") or "untitled", label + "s") for c in cards]
    mat = reasoning.materialize(items, _resolve_rules("ergo:aggregate"))
    with open(catalog, "a", encoding="utf-8") as f:
        for sec, text in reasoning.inventory_passages(mat, label=label, prefix="lib:catalog", collection="the catalog"):
            f.write(f"[{sec}] {text}\n")
    reasoning.strategy_tables(_resolve_rules("ergo:strategy"), os.path.join(out, "strategy.tsv"),
                              mat=mat, label=label, prefix="lib:catalog")   # match the inventory passage ids
    answers.empty_index(out, model=model)
    npass, nvec, k = grounding.build(catalog, out, dim=dim, no_split=True)
    print(f"[librarian] {len(cards)} cards over target={target!r}, {mat['total']} {label}s catalogued (count/list via "
          f"ergo) → {npass} passages, {nvec} vectors x {k}d")
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
    ap.add_argument("--prose", help="extra citable passages concatenated into the grounding corpus (e.g. prose.txt)")
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
    build_expert(a.out, corpus=a.corpus, prose=a.prose, bundle=a.bundle, questions=a.questions, steps=a.steps,
                 citation=a.citation, model=a.model, adapter=a.adapter, adapter_source=a.adapter_source,
                 dim=a.dim, corpus_vectors=a.corpus_vectors, no_split=a.no_split,
                 cover=a.cover, minsupp=a.minsupp, mindet=a.mindet, fieldrun=a.fieldrun)


if __name__ == "__main__":
    main()
