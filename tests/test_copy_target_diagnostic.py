"""Controlled target substitutions preserve the background and cannot hide failed causal contrasts."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from diagnose_copy_targets import TARGETS, experiment, generate  # noqa: E402
from certificate import replay, sha256  # noqa: E402

PRIOR = ROOT / "reference/benchmarks/qwen25_05b_copy_generalization"
pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")


def small():
    return generate(seeds=(5,), lengths=(8,), prefixes=(4,), gaps=(8,))


def test_factorial_dataset_and_controlled_edits():
    rows = generate(exclude=range(200, 300))
    assert len(rows) == 192
    cells = {}
    for row in rows:
        cells.setdefault((row["seed"], row["length"], row["prefix"], row["gap"]), []).append(row)
    assert len(cells) == 24
    for variants in cells.values():
        assert {r["target"] for r in variants} == {t for t, _, _ in TARGETS}
        base = variants[0]
        for edited in variants[1:]:
            differences = [p for p, (a, b) in enumerate(zip(base["ctx"], edited["ctx"])) if a != b]
            assert differences == base["positions"] == edited["positions"]
            assert len(differences) == 2
            assert base["ctx"][-base["prefix"]:] == edited["ctx"][-edited["prefix"]:]


def test_datalog_reports_recovery_regression_and_exact_failure(tmp_path):
    out = tmp_path / "diagnostic"

    def oracle(row):
        f = json.loads((out / "frozen.json").read_text())
        assert f["files"]["guarded/circuits.dl"] == sha256(PRIOR / "guarded/circuits.dl")
        assert f["domain"] == list(range(8))
        return -1 if row["target"] in (11436, 1602) else row["expected"]

    report = experiment(out, small(), oracle, {"source": "counterfactual-control"}, PRIOR)
    assert report["scores"]["test"]["wrong"] == 2
    assert ["11436", "2579", "recovered", "1"] in report["paired_count"]
    assert ["34341", "1602", "regressed", "1"] in report["paired_count"]
    assert not report["certificate"]["certified"]
    assert replay(out / report["certificate"]["evidence"])["nmiss"] == 2


def test_authored_control_certifies_all_cases(tmp_path):
    report = experiment(tmp_path / "control", small(), lambda row: row["expected"], {"source": "copy-control"}, PRIOR)
    assert report["certificate"]["certified"]
    assert report["scores"]["test"]["correct"] == 8
    assert all(r[2] == "both_correct" for r in report["paired_count"])


@pytest.mark.parametrize("defect", ["background-edit", "undeclared-source", "missing-reference"])
def test_invalid_intervention_or_reference_fails_audit(tmp_path, defect):
    rows = small()
    if defect == "background-edit":
        rows[1]["ctx"][0] = 190000
    elif defect == "undeclared-source":
        rows[1]["positions"] = rows[1]["positions"][:1]

    def oracle(row):
        return None if defect == "missing-reference" and row["id"] == 0 else row["expected"]

    with pytest.raises(ValueError, match="evaluation audit failed"):
        experiment(tmp_path / "bad", rows, oracle, {"source": "invalid-control"}, PRIOR)
