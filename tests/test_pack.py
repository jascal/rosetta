"""rosetta.pack — the expert-packaging layer. Two guards:
  (1) the core/pack boundary: the minimization core (py/*.py) must never import pack (pack → core only);
  (2) build-expert assembles a model-free package cover-first, with NO bare gram tier (CONVERGENCE.md).
Hermetic: no model, no network (lexical grounding via dim=0)."""
import glob
import json
import os
import re
import shutil
import sys

import pytest

_STRAT_DL = """
.decl cue(word:symbol, intent:symbol)
cue("many","count"). cue("list","list"). cue("what","define").
.output cue
.decl answer(intent:symbol, entity:symbol, section:symbol)
.decl defines(section:symbol, term:symbol)
.input defines
answer("define", T, S) :- defines(S, T).
.output answer
"""

_AGG_DL = """
.decl item(name:symbol, group:symbol)
.input item
.decl group(g:symbol)
group(G) :- item(_, G).
.decl uitem(name:symbol)
uitem(I) :- item(I, _).
.decl group_count(group:symbol, n:number)
group_count(G, N) :- group(G), N = count : { item(_, G) }.
.output group_count
.decl total_count(n:number)
total_count(N) :- N = count : { uitem(_) }.
.output total_count
.output item
"""

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "py"))


def test_core_never_imports_pack():
    """The boundary invariant: pack depends on core, never the reverse."""
    core = glob.glob(os.path.join(HERE, "py", "*.py"))           # top-level core modules; pack/ is a subdir, excluded
    offenders = [os.path.basename(f) for f in core
                 if re.search(r'^\s*(from|import)\s+pack\b', open(f, encoding="utf-8").read(), re.M)]
    assert not offenders, f"core modules import pack (boundary violation): {offenders}"


def test_build_model_free_expert(tmp_path):
    """build-expert assembles a model-free package: citable passages + grounding + empty index, and NO gram tier."""
    from pack import build_expert

    src = tmp_path / "norm.json"
    json.dump({"normative_rules": [
        {"name": "ecall_traps", "chapter_name": "Machine",
         "tags": [{"text": "ecall raises an environment-call exception."}]},
        {"name": "xlen_width", "chapter_name": "ISA", "tags": [{"text": "A base instruction is 32 bits wide."}]},
        {"name": "x0_zero", "chapter_name": "ISA", "tags": [{"text": "Register x0 is hardwired to zero."}]},
    ]}, open(src, "w"))
    out = tmp_path / "pkg"
    build_expert(str(out), adapter="normrules", adapter_source=str(src), dim=0, model="test-spec")  # dim=0 → lexical, no download

    idx = json.load(open(out / "index.json"))
    assert idx["model"] == "test-spec" and idx["items"] == []   # model-free → no curated items
    assert (out / "corpus.txt").exists()                        # the adapter's Extraction → canonical corpus
    kn = (out / "knowledge.tsv").read_text()                    # grounding (citation)
    assert "ecall" in kn and "x0" in kn
    assert not (out / "gram").exists()                          # cover-first: NO bare gram tier (CONVERGENCE.md)


def test_build_expert_cover_orchestration(tmp_path, monkeypatch):
    """Exercise the distilled + cover orchestration hermetically: stub the model-dependent steps (fieldrun distill,
    cover extraction) and assert build_expert wires curated answers + grounding(citation) + the cover, still no gram."""
    import pack.build as B

    out = tmp_path / "pkg"
    out.mkdir()
    corpus = tmp_path / "kb.txt"
    corpus.write_text("Logic is the study of valid inference. A tautology is always true regardless of its parts.\n")
    questions = tmp_path / "q.txt"
    questions.write_text("What is a tautology?\n")

    monkeypatch.setattr(B.subprocess, "run", lambda *a, **k: None)              # skip the real fieldrun distill

    def fake_answers(o, c, dl, *, citation="", cite=None, model="m"):           # pretend distill produced 1 curated item
        json.dump({"model": model, "items": [{"id": "p00000", "query": "q", "answer": "a",
                                              "citation": citation, "facts": ""}]},
                  open(os.path.join(o, "index.json"), "w"))
        return [1]
    monkeypatch.setattr(B.answers, "from_export", fake_answers)

    def fake_cover(model_dir, *, minsupp=3, mindet=1.0, **k):                   # pretend the core emitted a cover
        pk = os.path.join(model_dir, "package")
        os.makedirs(pk, exist_ok=True)
        json.dump({"model": "m", "n_rules": 1, "rules": [{"kind": "ngram", "ctx": [1], "out": 2}]},
                  open(os.path.join(pk, "manifest.json"), "w"))
        return os.path.join(pk, "manifest.json")
    monkeypatch.setattr(B.cover_mod, "build", fake_cover)

    made = {}

    def fake_make_corpus(o, text_file, bundle):                                 # the cover-corpus wiring (no real tokenizer)
        json.dump({"ids": [1, 2, 3]}, open(os.path.join(o, "corpus.json"), "w"))
        made["called"] = (text_file, bundle)
        return 3
    monkeypatch.setattr(B, "_make_corpus", fake_make_corpus)

    B.build_expert(str(out), corpus=str(corpus), bundle="dummy", questions=str(questions),
                   fieldrun=sys.executable, dim=0, cover=True, model="m")        # fieldrun=existing path; run() stubbed

    assert (out / "index.json").exists()                                       # curated answers wired
    assert (out / "knowledge.tsv").exists()                                    # grounding (citation) wired
    assert (out / "corpus.json").exists() and made["called"]                    # cover SCOPED to the corpus (the gap fix)
    assert made["called"][0] == str(corpus)                                     # …over the user's corpus, not a generic one
    assert (out / "package" / "manifest.json").exists()                        # cover wired
    assert not (out / "gram").exists()                                         # still no bare gram tier


def test_normrules_rejects_malformed_source(tmp_path):
    """The adapter fails with a clear error on a source missing normative_rules / yielding no usable rules."""
    from pack.adapters import normrules
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rules": []}))                                   # wrong top-level key
    with pytest.raises(ValueError, match="normative_rules"):
        normrules.to_corpus(str(bad), str(tmp_path / "o1"))
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"normative_rules": [{"chapter_name": "X"}]}))  # no name/text → 0 usable
    with pytest.raises(ValueError, match="0 usable"):
        normrules.to_corpus(str(empty), str(tmp_path / "o2"))


def test_distill_requires_explicit_fieldrun(tmp_path, monkeypatch):
    """A distilled build must NAME the extractor: no fieldrun=… and no $FIELDRUN → clear error, never a hard-coded path."""
    from pack import build_expert
    monkeypatch.delenv("FIELDRUN", raising=False)
    q = tmp_path / "q.txt"
    q.write_text("What is a tautology?\n")
    with pytest.raises(ValueError, match="fieldrun not specified"):
        build_expert(str(tmp_path / "pkg"), bundle="dummy", questions=str(q), dim=0)


def test_spec_load_and_kwargs(tmp_path, monkeypatch):
    """expert.toml → build kwargs: $ENV resolved, relative paths based, a [model] ⇒ cover=True, [grounding].dim honored."""
    from pack import spec as spec_mod
    monkeypatch.setenv("TESTBUNDLE", "/bundles/m")
    monkeypatch.setenv("TESTFR", "/bin/fieldrun")
    (tmp_path / "expert.toml").write_text(
        '[corpus]\ntext="kb.txt"\nquestions="q.txt"\ncitation="T (CC BY)"\n'
        '[model]\nbundle="$TESTBUNDLE"\nfieldrun="$TESTFR"\n'
        '[grounding]\ndim=0\n'
        '[experiment]\noff_domain="neg.txt"\n[gate]\nmin_precision=0.9\nmax_leak=0.05\n'
        '[reasoning]\nrules="ergo:core"\n')
    s = spec_mod.load_spec(str(tmp_path / "expert.toml"))
    assert s["model"]["bundle"] == "/bundles/m"                 # $ENV expanded
    assert s["reasoning"]["rules"] == "ergo:core"               # opt-in tier parsed (rosetta is aware of it)
    kw = spec_mod.to_build_kwargs(s, base=str(tmp_path))
    assert kw["corpus"] == os.path.join(str(tmp_path), "kb.txt")  # relative path based
    assert kw["bundle"] == "/bundles/m" and kw["cover"] is True   # a model ⇒ build a cover
    assert kw["dim"] == 0 and kw["citation"] == "T (CC BY)"


def _manifest(tmp_path):
    m = {"n_rules": 2, "rules": [
        {"kind": "ngram", "ctx": [11, 12], "out": 99, "basis": "observational", "cite": [1]},
        {"kind": "ngram", "ctx": [12], "out": 88, "basis": "observational", "cite": [2]}]}
    p = tmp_path / "manifest.json"
    json.dump(m, open(p, "w"))
    return str(p)


def test_eval_score_and_gate(tmp_path):
    """The scorecard grades a package by serving it; the gate hard-fails below thresholds."""
    from pack import eval as scorer
    mp = _manifest(tmp_path)
    holdout = [((0, 11, 12), 99), ((0, 9, 12), 88), ((5, 6, 7), 1)]   # 2 covered (both correct), 1 abstain
    off_domain = [(500, 501)]                                          # no rule → abstain → no leak
    sc = scorer.score(mp, holdout, off_domain)
    assert abs(sc["coverage"] - 2 / 3) < 1e-9 and sc["precision"] == 1.0
    assert sc["off_domain_leak"] == 0.0 and sc["confident_wrong"] == 0.0
    assert scorer.gate(sc, min_precision=0.9, max_leak=0.05) == (True, [])

    bad = scorer.score(mp, [((0, 11, 12), 77)], off_domain)           # answers 99 ≠ gold 77 → precision 0
    ok, reasons = scorer.gate(bad, min_precision=0.9, max_leak=0.05)
    assert ok is False and any("precision" in r for r in reasons)     # HARD-FAIL below the floor


def test_inventory_aggregates(tmp_path):
    """ergo count/list aggregates over an extracted instruction inventory → materialized count + cited KB passages."""
    if not shutil.which("souffle"):
        pytest.skip("souffle not installed (build-time aggregate tier)")
    from pack import reasoning
    corpus = tmp_path / "rules.txt"
    corpus.write_text(
        "[norm:a1 · RV32I] The insn:add[] and insn:sub[] instructions compute.\n"
        "[norm:m1 · M Extension] The insn:mul[] instruction; see also insn:add[].\n"
        "[norm:m2 · M Extension] insn:mulh[] gives the high bits.\n")
    inv = reasoning.instruction_inventory(str(corpus))
    assert ("add", "RV32I") in inv and ("mul", "M Extension") in inv and len(inv) == 5   # add,sub,mul,add(M),mulh

    rules = tmp_path / "aggregate.dl"
    rules.write_text(_AGG_DL)
    mat = reasoning.materialize(inv, str(rules))
    assert mat["total"] == 4                                       # DISTINCT: add, sub, mul, mulh (add deduped)
    assert mat["groups"]["RV32I"] == 2 and mat["groups"]["M Extension"] == 3

    passages = reasoning.inventory_passages(mat)
    assert any("4 distinct" in t for _, t in passages)            # the total, stated + cited
    aug, npx, _ = reasoning.augment_corpus_with_inventory(str(corpus), str(rules), str(tmp_path / "aug.txt"))
    assert npx >= 3 and "riscv:inventory:total" in open(aug).read()  # count passages appended (cited handles)


def test_build_from_spec_model_free(tmp_path):
    """build_from_spec on a model-free [corpus]-only spec (dim=0, hermetic) → package built; no manifest ⇒ gate skipped."""
    import pack.build as B
    (tmp_path / "kb.txt").write_text("A tautology is always true. An argument is valid if the form preserves truth.\n")
    (tmp_path / "neg.txt").write_text("What is the capital of France?\n")
    (tmp_path / "expert.toml").write_text(
        '[corpus]\ntext="kb.txt"\ncitation="T (CC BY)"\n'
        '[grounding]\ndim=0\n'
        '[experiment]\noff_domain="neg.txt"\n[gate]\nmax_leak=0.05\n')          # gate set, but no manifest → skipped, no crash
    out = B.build_from_spec(str(tmp_path / "expert.toml"))
    assert os.path.exists(os.path.join(out, "index.json"))                       # empty index (model-free)
    assert os.path.exists(os.path.join(out, "knowledge.tsv"))                    # grounding (citation)
    assert not os.path.exists(os.path.join(out, "manifest.json"))                # no cover (model-free)


def test_riscv_prose_adapter(tmp_path):
    """The prose adapter cleans AsciiDoc → citable passages: macros decoded, markup stripped, prose kept, junk dropped."""
    from pack.adapters import riscv_prose

    src = tmp_path / "src" / "unpriv"
    src.mkdir(parents=True)
    (src / "intro.adoc").write_text(
        "== Introduction\n\n"
        "(((hart, definition)))\n"
        "A component is termed a _core_ if it contains an instruction fetch unit, supporting multiple harts.\n\n"
        "The insn:mret[] instruction returns from a trap, and csr:mstatus[mpp] holds the previous mode here.\n\n"
        "[NOTE]\n====\n"
        "See <<chapter2, the next chapter>> and cite:[ref2020] for the full rationale on this design point.\n"
        "====\n\n"
        "|===\n| col | col\n| 1 | 2\n|===\n\n"        # a table → must be skipped
        "short\n",                                     # too short → dropped
        encoding="utf-8")

    prose_txt, _plain, n = riscv_prose.to_corpus(str(tmp_path / "src"), str(tmp_path / "out"),
                                                 chapters=["unpriv/intro.adoc"])
    body = open(prose_txt, encoding="utf-8").read()
    assert n >= 2
    assert "termed a core" in body and "_core_" not in body          # _italic_ stripped
    assert "MRET" in body and "insn:mret[]" not in body              # macro decoded
    assert "mstatus.mpp" in body                                     # csr:NAME[field] → NAME.field
    assert "(((" not in body and "cite:" not in body                 # index + cite macros gone
    assert "the next chapter" in body and "<<" not in body           # xref → display text; NOTE prose kept
    assert "| col |" not in body                                     # table skipped
    assert "] short" not in body                                     # too-short fragment dropped
    assert "[manual:intro_" in body                                  # citable handle (prose namespace)


def test_build_concats_prose_into_grounding(tmp_path):
    """[corpus] prose= concatenates extra citable passages into the grounding corpus (rules + prose, both no-split)."""
    from pack import build_expert

    src = tmp_path / "norm.json"
    json.dump({"normative_rules": [
        {"name": "x0_zero", "chapter_name": "ISA", "tags": [{"text": "Register x0 is hardwired to zero."}]},
    ]}, open(src, "w"))
    prose = tmp_path / "prose.txt"
    prose.write_text("[manual:intro_0 · Intro] A hart is a resource that fetches and executes instructions.\n")
    out = tmp_path / "pkg"
    build_expert(str(out), adapter="normrules", adapter_source=str(src), prose=str(prose), dim=0, model="t")

    kn = (out / "knowledge.tsv").read_text()
    assert "x0 is hardwired" in kn and "hart is a resource" in kn    # BOTH rules and prose grounded
    assert "manual:intro_0" in kn                                    # prose citation handle present


def test_strategy_tables_uniform(tmp_path):
    """The UNIFORM strategy table: count/list/define are all `answer <intent> <entity> <passage>` rows — one shape,
    no per-kind special-casing. The entity is what the query must NAME (the domain gate)."""
    if not shutil.which("souffle"):
        pytest.skip("souffle not installed (build-time strategy tier)")
    from pack import reasoning
    sdl = tmp_path / "strategy.dl"
    sdl.write_text(_STRAT_DL)
    mat = {"total": 4, "groups": {"RV32I": 2, "M Extension": 3}, "items": []}
    out = tmp_path / "strategy.tsv"
    _p, ncue, nans = reasoning.strategy_tables(str(sdl), str(out), mat=mat, label="instruction",
                                               defines=[("manual:intro_17", "hart")])
    rows = [ln.split("\t") for ln in out.read_text().splitlines()]
    cues = [r for r in rows if r[0] == "cue"]
    ans = [r for r in rows if r[0] == "answer"]
    assert len(cues) == ncue == 3
    # every answer row is the same shape: answer <intent> <entity> <passage>
    assert all(len(r) == 4 for r in ans)
    assert ["answer", "count", "instruction", "riscv:inventory:total"] in ans          # count → label names it
    assert ["answer", "list", "m extension", "riscv:inventory:M Extension"] in ans      # list → group name
    assert ["answer", "define", "hart", "manual:intro_17"] in ans                       # define → term (from defines)
    assert nans == len(ans) == 4                                                        # 1 count + 2 groups + 1 define


def test_extract_defines_parenthesized_only(tmp_path):
    """The build-time defines extractor takes a section's parenthesized abbreviation (the spec's canonical id) as the
    term its FIRST passage defines — and does NOT turn ordinary heading words ("cause", "mode") into entities (those
    would match unrelated queries like "what causes earthquakes")."""
    from pack import reasoning
    corpus = tmp_path / "prose.txt"
    corpus.write_text(
        "[manual:m_2 · Machine › Machine ISA (misa) Register] The misa CSR is a WARL register.\n"
        "[manual:m_9 · Machine › Machine ISA (misa) Register] The e bit is read-only.\n"   # same section, not first
        "[manual:m_213 · Machine › Machine Cause (mcause) Register] The mcause register holds the trap cause.\n"
        "[manual:i_7 · Intro › RISC-V Hardware Platform Terminology] A component is termed a core.\n")
    d = reasoning.extract_defines(str(corpus))
    byterm = {t: cid for cid, t in d}
    assert byterm["misa"] == "manual:m_2"            # parenthesized id → the section's FIRST passage
    assert byterm["mcause"] == "manual:m_213"
    assert "cause" not in byterm and "machine" not in byterm and "core" not in byterm   # no ordinary-word entities
    assert "register" not in byterm and "terminology" not in byterm


def test_adapter_system_registry_and_contract(tmp_path):
    """The document-adapter system: all sources are registered instances of the one Extraction contract."""
    from pack import adapters
    assert set(adapters.names()) >= {"normrules", "riscv_prose", "pretext", "latexml"}

    # normrules instance → passages + the insn inventory items, as an Extraction
    src = tmp_path / "norm.json"
    json.dump({"normative_rules": [
        {"name": "m1", "chapter_name": "M Extension", "tags": [{"text": "The insn:mul[] instruction multiplies."}]},
    ]}, open(src, "w"))
    ext = adapters.get("normrules")(str(src))
    assert ext.passages and ("mul", "M Extension") in ext.items     # items feed count/list
    corpus = ext.write_corpus(str(tmp_path / "c.txt"))
    assert "[norm:m1 · M Extension]" in open(corpus).read()         # citable handle preserved

    # latexml instance → a LaTeXML HTML snippet yields sectioned passages + a named definition/theorem
    htm = tmp_path / "paper.html"
    htm.write_text(
        '<h2 class="ltx_title ltx_title_section">Preliminaries</h2>'
        '<div class="ltx_para"><p class="ltx_p">We work over a field '
        '<math alttext="\\mathbb{F}">MATHML</math> throughout this paper.</p></div>'
        '<div class="ltx_theorem ltx_theorem_theorem"><p class="ltx_p">Theorem 1 (Euler). '
        'For coprime a and n, a power phi of n is one mod n.</p></div>')
    ext = adapters.get("latexml")(str(htm), prefix="px")
    assert any("field" in t for _s, t in ext.passages)             # prose extracted
    assert any("\\mathbb{F}" in t for _s, t in ext.passages)       # math kept as LaTeX (alttext), MathML dropped
    assert "MATHML" not in " ".join(t for _s, t in ext.passages)   # MathML subtree skipped
    assert ("px:theorem2", "euler") in [(p, n) for p, n in ext.statements] or \
           any(n == "euler" for _p, n in ext.statements)           # named theorem captured


def test_pretext_adapter_extraction(tmp_path):
    """The PreTeXt adapter (ElementTree) extracts: a <definition>'s title + inline <term> as exact defines, a titled
    <theorem> as a named statement, prose <p> as passages, and <m> math as inline LaTeX. Covers both tag dialects."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "groups.xml").write_text(
        '<chapter xml:id="grp"><title>Groups</title>'
        '<section><title>Definitions</title>'
        '<definition xml:id="def-group"><title>Group</title>'
        '<p>A <term>group</term> is a set with an associative binary operation and inverses.</p></definition>'
        '<theorem xml:id="thm-lagrange"><title>Lagrange</title>'
        '<statement><p>For a finite group, <m>|H|</m> divides <m>|G|</m>.</p></statement></theorem>'
        '<p>This running paragraph introduces the notion of a <define>subgroup</define> informally for the reader.</p>'
        '</section></chapter>', encoding="utf-8")
    from pack.adapters import pretext
    passages, defines, statements = pretext.extract(str(src), "bk")
    by = {t: pid for pid, t in defines}
    assert by["group"] == "bk:definition:def-group"          # definition title → exact define
    assert "subgroup" in by                                   # inline <define> in prose → define
    assert ("bk:theorem:thm-lagrange", "lagrange") in statements   # titled theorem → named statement
    body = dict((s.split(" · ")[0], t) for s, t in passages)
    assert "|H|" in body["bk:theorem:thm-lagrange"] and "|G|" in body["bk:theorem:thm-lagrange"]  # <m> LaTeX kept


def test_adapter_source_envvar_expands(tmp_path, monkeypatch):
    """[adapter] source = "$VAR" is expanded by the spec loader so clone-and-build (AATA_SRC=…) works."""
    from pack import spec as spec_mod
    monkeypatch.setenv("BOOK_SRC", "/clones/aata/src")
    (tmp_path / "e.toml").write_text('[adapter]\nname="pretext"\nsource="$BOOK_SRC"\nprefix="aata"\n')
    kw = spec_mod.to_build_kwargs(spec_mod.load_spec(str(tmp_path / "e.toml")), base=str(tmp_path))
    assert kw["adapter"] == "pretext" and kw["adapter_source"] == "/clones/aata/src"   # $VAR expanded, absolute kept
    assert kw["adapter_opts"] == {"prefix": "aata"}


def test_multi_document_build(tmp_path):
    """N documents of M adapter types compose into ONE expert: a normrules spec + a PreTeXt book → merged Extraction
    (passages namespaced, defines/items combined). [[document]] in the spec; build merges."""
    from pack import build_expert
    # doc 1: a normrules spec (items)
    norm = tmp_path / "norm.json"
    json.dump({"normative_rules": [
        {"name": "m1", "chapter_name": "M Extension", "tags": [{"text": "The insn:mul[] instruction multiplies."}]},
    ]}, open(norm, "w"))
    # doc 2: a tiny PreTeXt book (defines)
    bk = tmp_path / "bk"
    bk.mkdir()
    (bk / "c.xml").write_text(
        '<section><title>Sets</title><definition xml:id="d-set"><title>Set</title>'
        '<p>A <term>set</term> is an unordered collection of objects in this book.</p></definition></section>',
        encoding="utf-8")
    out = tmp_path / "pkg"
    build_expert(str(out), dim=0, model="multi",
                 documents=[{"adapter": "normrules", "source": str(norm)},
                            {"adapter": "pretext", "source": str(bk), "opts": {"prefix": "bk"}}])
    kn = (out / "knowledge.tsv").read_text()
    assert "norm:m1" in kn and "bk:definition:d-set" in kn          # BOTH documents' passages, namespaced, no collision
    strat = (out / "strategy.tsv").read_text()
    assert "\tset\tbk:definition:d-set" in strat                    # the book's define
    assert "\tmul\t" in strat or "answer\tcount\t" in strat         # the spec's inventory (count/list)


def test_multi_document_skips_bad_source(tmp_path, capsys):
    """At N-document scale, one unbuildable document is SKIPPED (with a warning), not fatal — the build proceeds on the
    rest. An unknown adapter NAME, by contrast, is a config error and fails hard."""
    from pack import build_expert
    good = tmp_path / "g.json"
    json.dump({"normative_rules": [{"name": "r1", "chapter_name": "C", "tags": [{"text": "x0 is hardwired to zero."}]}]},
              open(good, "w"))
    out = tmp_path / "pkg"
    build_expert(str(out), dim=0, model="m", documents=[
        {"adapter": "normrules", "source": str(good)},
        {"adapter": "normrules", "source": str(tmp_path / "missing.json")},   # bad source → skipped, not fatal
    ])
    assert "SKIPPED" in capsys.readouterr().out
    assert "x0 is hardwired" in (out / "knowledge.tsv").read_text()           # the good document still built
    # unknown adapter name is a hard config error
    import pytest as _pt
    with _pt.raises(KeyError):
        build_expert(str(tmp_path / "p2"), dim=0, documents=[{"adapter": "nope", "source": str(good)}])


def test_librarian_catalog(tmp_path):
    """A LIBRARIAN is a model-free catalog expert: each document → a card whose handle points INTO the content expert
    (lib:<target>:<doc>); the catalog is also an inventory (count/list via ergo). No model."""
    if not shutil.which("souffle"):
        pytest.skip("souffle not installed (the catalog inventory uses ergo aggregates)")
    from pack.build import build_librarian
    d = tmp_path / "docs"
    d.mkdir()
    for i, (pid, title) in enumerate([("p1", "Attention and Transformers"), ("p2", "Graph Coloring Bounds")]):
        (d / f"{pid}.html").write_text(
            f'<h1 class="ltx_title ltx_title_document">{title}</h1>'
            f'<div class="ltx_abstract"><p class="ltx_p">We study {title.lower()} in depth here.</p></div>'
            f'<div class="ltx_para"><p class="ltx_p">{title}: a long enough body paragraph to be a real passage here.</p></div>',
            encoding="utf-8")
    out = tmp_path / "lib"
    build_librarian(str(out), dim=0, target="arxiv", label="paper", documents=[
        {"adapter": "latexml", "source": str(d / "p1.html"), "opts": {"prefix": "p1"}},
        {"adapter": "latexml", "source": str(d / "p2.html"), "opts": {"prefix": "p2"}},
    ])
    cat = (out / "catalog.txt").read_text()
    assert "[lib:arxiv:p1 · Attention and Transformers]" in cat       # card cites INTO the content expert (target:doc)
    assert "There are 2 distinct papers in the catalog" in cat        # the catalog is an inventory (ergo count)
    strat = (out / "strategy.tsv").read_text()
    assert "answer\tcount\tpaper\tlib:catalog:total" in strat         # count/list routed via ergo strategy.dl


def test_pedagogy_templates_as_expert(tmp_path):
    """Pedagogy templates are a model-free expert: each template → a passage + a generic ('pedagogy', entity, passage)
    strategy row, so selection ('socratic intro') routes via the uniform table (ergo). No model."""
    if not shutil.which("souffle"):
        pytest.skip("souffle not installed (strategy tier)")
    from pack import build_expert
    (tmp_path / "t.toml").write_text(
        '[[template]]\nname="socratic-tutor"\nstyle="socratic"\nlevel="intro"\nsystem="You are a Socratic tutor for {scope}."\n'
        '[[template]]\nname="examiner"\nstyle="quiz"\nlevel="any"\nsystem="You are an examiner for {scope}."\n')
    out = tmp_path / "pkg"
    build_expert(str(out), dim=0, model="pedagogy",
                 documents=[{"adapter": "pedagogy", "source": str(tmp_path / "t.toml"), "opts": {"prefix": "ped"}}])
    kn = (out / "knowledge.tsv").read_text()
    assert "ped:socratic-tutor" in kn and "Socratic tutor" in kn          # template is a citable passage
    strat = (out / "strategy.tsv").read_text()
    assert "answer\tpedagogy\tsocratic intro\tped:socratic-tutor" in strat  # ergo-routed selection by style+level
    assert "answer\tpedagogy\tquiz\tped:examiner" in strat


def test_ergo_pinned_dependency_resolution(tmp_path, monkeypatch):
    """ergo is a pinned PUBLISHED dependency, not a hard sibling-dir: $ERGO_DIR overrides; a recorded pin (repo@ref)
    lets a build elsewhere fetch the rules. (The sibling-dev path + the pinned fetch aren't exercised here.)"""
    import pack.build as B
    monkeypatch.setenv("ERGO_DIR", str(tmp_path))
    assert B._ergo_dir() == str(tmp_path)                         # explicit override wins
    assert B.ERGO_REPO.endswith("/ergo") and B.ERGO_REF          # the pin is recorded (reproducible/deployed builds)
