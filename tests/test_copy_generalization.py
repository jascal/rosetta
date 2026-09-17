"""Fixed-artifact stress tests: no selection, no reference-dependent domain filtering."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from benchmark_copy_generalization import benchmark, generate  # noqa: E402
from certificate import replay, sha256  # noqa: E402

PRIOR = ROOT / "reference/benchmarks/qwen25_05b_guarded_seed1"
pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")


def test_groups_are_fresh_and_interventions_preserve_every_other_token():
    excluded = set(range(200, 400))
    rows = generate(seeds=(2, 3), lengths=(8, 12), groups=1, exclude=excluded)
    vocab, parts = {}, {}
    for row in rows:
        assert not set(row["ctx"]) & excluded
        vocab.setdefault(row["group"], set()).update(row["ctx"])
        parts.setdefault(row["group"], set()).add(row["part"])
        if row["kind"] == "intervention":
            original = rows[row["base"]]
            differences = [p for p, (a, b) in enumerate(zip(original["ctx"], row["ctx"])) if a != b]
            assert differences == row["positions"]
            assert len(differences) == row["exposures"]
            assert original["ctx"][-3:] == row["ctx"][-3:]
        if row["layout"] == "near_match8":
            suffix = row["ctx"][-3:]
            end_of_gap = len(row["ctx"]) - row["prefix"]
            assert row["ctx"][end_of_gap - 3:end_of_gap - 1] == suffix[-2:]
            assert row["ctx"][end_of_gap - 4] != suffix[0]
    assert all(len(p) == 1 for p in parts.values())
    for a in vocab:
        for b in vocab:
            if a != b:
                assert not vocab[a] & vocab[b]


@pytest.mark.parametrize("corrupt", [False, True])
def test_fixed_circuit_domains_precede_all_references(tmp_path, corrupt):
    rows = generate(seeds=(2,), lengths=(8,), groups=1)
    out = tmp_path / "run"
    prior_hash = sha256(PRIOR / "selected/circuits.dl")

    def oracle(row):
        frozen = json.loads((out / "frozen.json").read_text())
        assert frozen["files"]["guarded/circuits.dl"] == prior_hash
        assert frozen["artifacts"]["guarded"]["domain"] == [r["id"] for r in rows if r["part"] == "test"]
        return -1 if corrupt and row["id"] == 0 else row["expected"]

    report = benchmark(out, rows, oracle, {"source": "test-control"}, PRIOR)
    guarded = report["results"]["guarded"]
    assert guarded["sha256"] == prior_hash
    assert guarded["scores"]["test"]["wrong"] == int(corrupt)
    assert guarded["scores"]["test"]["answered"] == 16
    assert guarded["scores"]["off_domain"]["answered"] == 0
    cert = guarded["certificates"]["frozen_firing_domain"]
    assert cert["ndomain"] == 16 and cert["certified"] is (not corrupt)
    assert replay(out / cert["evidence"])["certified"] is (not corrupt)
    assert report["results"]["raw_copy"]["strata"]["layout"]["near_match8"]["wrong"] == 4
    for factor in ("seed", "length", "layout", "exposures", "kind"):
        assert sum(s["n"] for s in guarded["strata"][factor].values()) == 16
    if corrupt:
        assert guarded["counterexamples"][0]["id"] == 0


def test_constant_control_cannot_reselect_an_abstaining_rule(tmp_path):
    report = benchmark(tmp_path / "constant", generate(seeds=(2,), lengths=(8,), groups=1),
                       lambda row: 0, {"source": "constant-control"}, PRIOR)
    guarded = report["results"]["guarded"]
    assert guarded["sha256"] == sha256(PRIOR / "selected/circuits.dl")
    assert guarded["scores"]["test"]["wrong"] == 16
    assert not guarded["certificates"]["full_test"]["certified"]


@pytest.mark.parametrize("changed", ["dataset.json", "guarded/circuits.dl", "frozen.json"])
def test_mutation_after_freeze_is_rejected(tmp_path, changed):
    out = tmp_path / "run"

    def oracle(row):
        if row["id"] == 0:
            path = out / changed
            path.write_text(path.read_text() + "\n")
        return row["expected"]

    with pytest.raises(ValueError, match="changed"):
        benchmark(out, generate(seeds=(2,), lengths=(8,), groups=1), oracle, {"source": "mutation-control"}, PRIOR)
    assert not (out / "report.json").exists()


def test_missing_oracle_reference_fails_evaluation_audit(tmp_path):
    with pytest.raises(ValueError, match="evaluation audit failed"):
        benchmark(tmp_path / "missing", generate(seeds=(2,), lengths=(8,), groups=1),
                  lambda row: None if row["id"] == 0 else row["expected"], {"source": "missing-reference"}, PRIOR)
