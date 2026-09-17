"""Document split precedes cover extraction; requested gates require actual evidence."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from pack import build, holdout  # noqa: E402
from pack.eval import gate  # noqa: E402


def test_split_groups_duplicates_and_detects_tampering(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("one two three\none  two three\nfour five six\nseven eight nine\n")
    out = tmp_path / "out"
    holdout.prepare(out, source, fraction=.5)
    train, test, manifest = holdout.load(out)
    assert len(train) + len(test) == 4
    assert not {holdout.group_id(x) for x in train} & {holdout.group_id(x) for x in test}
    assert not set(manifest["train_groups"]) & set(manifest["holdout_groups"])
    (out / "evaluation/train.txt").write_text("tampered\n")
    with pytest.raises(ValueError, match="changed after split"):
        holdout.load(out)


@pytest.mark.parametrize("fraction", [0, 1, -1, float("nan")])
def test_split_rejects_invalid_fraction(tmp_path, fraction):
    with pytest.raises(ValueError, match="fraction"):
        holdout.prepare(tmp_path, tmp_path / "unused", fraction=fraction)


def test_split_happens_before_cover_build(tmp_path, monkeypatch):
    source = tmp_path / "corpus.txt"
    source.write_text("first document\nsecond document\nthird document\n")
    spec = tmp_path / "expert.toml"
    spec.write_text('[corpus]\ntext="corpus.txt"\n[model]\nbundle="unused"\n'
                    '[experiment]\nholdout=0.34\n[gate]\nmin_precision=0.9\n')
    called = []

    def build_stub(out, **kwargs):
        train, test, _ = holdout.load(out)
        assert holdout.documents(kwargs["corpus"]) == train
        assert not set(train) & set(test)
        assert kwargs["cover"]
        called.append("build")

    monkeypatch.setattr(build, "build_expert", build_stub)
    monkeypatch.setattr(build, "_score_if_gated", lambda *args: called.append("score"))
    build.build_from_spec(spec)
    assert called == ["build", "score"]


def test_eval_windows_never_cross_documents(tmp_path, monkeypatch):
    source = tmp_path / "corpus.txt"
    source.write_text("1 2 3 4\n5 6 7 8\n9 10 11 12\n13 14 15 16\n")
    holdout.prepare(tmp_path, source, fraction=.5, window=2)
    (tmp_path / "bundle.tokenizer.json").write_text("{}")
    tokenizer = SimpleNamespace(encode=lambda line: SimpleNamespace(ids=list(map(int, line.split()))))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _: tokenizer)))
    _, documents, _ = holdout.load(tmp_path)
    pairs, off = build._eval_sets(tmp_path, {}, tmp_path)
    expected = []
    for doc in documents:
        ids = list(map(int, doc.split()))
        expected.extend([(tuple(ids[:2]), ids[2]), (tuple(ids[1:3]), ids[3])])
    assert pairs == expected
    assert off == []


def test_gate_fails_without_observations():
    sc = {"holdout_n": 0, "precision": None, "off_domain_leak": None}
    ok, reasons = gate(sc, min_precision=.9, max_leak=.05)
    assert not ok and len(reasons) == 3


def test_zero_answers_cannot_pass_precision_gate(tmp_path, monkeypatch):
    from pack import eval as scorer
    from test_pack import _manifest
    sc = scorer.score(_manifest(tmp_path), [((999,), 1)], [])
    assert sc["precision"] is None and sc["off_domain_leak"] is None
    assert not gate(sc, min_precision=0, max_leak=1)[0]
