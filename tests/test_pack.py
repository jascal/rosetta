"""rosetta.pack — the expert-packaging layer. Two guards:
  (1) the core/pack boundary: the minimization core (py/*.py) must never import pack (pack → core only);
  (2) build-expert assembles a model-free package cover-first, with NO bare gram tier (CONVERGENCE.md).
Hermetic: no model, no network (lexical grounding via dim=0)."""
import glob
import json
import os
import re
import sys

import pytest

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

    B.build_expert(str(out), corpus=str(corpus), bundle="dummy", questions=str(questions),
                   fieldrun=sys.executable, dim=0, cover=True, model="m")        # fieldrun=existing path; run() stubbed

    assert (out / "index.json").exists()                                       # curated answers wired
    assert (out / "knowledge.tsv").exists()                                    # grounding (citation) wired
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
