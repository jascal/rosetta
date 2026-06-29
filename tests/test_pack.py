"""rosetta.pack — the expert-packaging layer. Two guards:
  (1) the core/pack boundary: the minimization core (py/*.py) must never import pack (pack → core only);
  (2) build-expert assembles a model-free package cover-first, with NO bare gram tier (CONVERGENCE.md).
Hermetic: no model, no network (lexical grounding via dim=0)."""
import glob
import json
import os
import re
import sys

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
