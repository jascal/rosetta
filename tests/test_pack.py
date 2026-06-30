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
    assert (out / "rules.txt").exists()                         # adapter passages
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
