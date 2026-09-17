"""Held-out references cannot choose a circuit; certificates exercise actual Souffle."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from benchmark_induction import benchmark, evaluate, generate, write_copy  # noqa: E402
from certificate import replay  # noqa: E402
from minimize import emit  # noqa: E402
from oracle import run_equiv  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")


def test_lexical_emitter_handles_mixed_context_lengths(tmp_path):
    candidate = tmp_path / "circuits.dl"
    emit(str(candidate), {(2,): 7, (4,): 8}, False,
         {1: "one", 2: "two", 3: "three", 4: "four", 7: "seven", 8: "eight"})
    result = run_equiv(candidate, [[2], [1, 2], [1, 3, 4]], [7, 7, 8])
    assert result["certified"], result
    (tmp_path / "tok.facts").write_text("0\t0\ttwo\n1\t0\tone\n1\t1\ttwo\n2\t0\tone\n2\t1\tthree\n2\t2\tfour\n")
    subprocess.run(["souffle", "circuits.symbols.dl", "-F", ".", "-D", "."], cwd=tmp_path,
                   check=True, capture_output=True)
    assert set((tmp_path / "cdecide.csv").read_text().splitlines()) == {"0\tseven", "1\tseven", "2\teight"}


def test_generation_groups_and_interventions():
    rows = generate(groups=(1, 1, 1))
    seen = {}
    for row in rows:
        seen.setdefault(row["part"], set()).update(row["ctx"])
        if row["kind"] == "intervention":
            base = rows[row["base"]]
            assert (row["part"], row["group"]) == (base["part"], base["group"])
            assert [i for i, (a, b) in enumerate(zip(row["ctx"], base["ctx"])) if a != b] == [row["position"]]
            assert row["ctx"][-3:] == base["ctx"][-3:]
    for a in seen:
        for b in seen:
            if a != b:
                assert not seen[a] & seen[b]


@pytest.mark.parametrize("corrupt_test", [False, True])
def test_freeze_before_test_and_exact_certificate(tmp_path, corrupt_test):
    rows = generate(groups=(1, 1, 1))
    out = tmp_path / "benchmark"

    def oracle(row):
        if row["part"] in ("test", "off_domain"):
            frozen = json.loads((out / "frozen.json").read_text())
            assert frozen["selected_length"] == [1]
            assert len(frozen["artifact_sha256"]) == 3
        if corrupt_test and row["part"] == "test":
            return -1
        return row["expected"]

    report = benchmark(out, rows, oracle, {"source": "authored-test-control"})
    assert report["selected_length"] == [1]
    assert report["results"]["baseline"]["scores"]["test"]["answered"] == 0
    chosen = report["results"]["selected"]
    assert chosen["scores"]["test"]["answered"] == 6
    assert chosen["scores"]["test"]["correct"] == (0 if corrupt_test else 6)
    assert chosen["test_certificate"]["certified"] is (not corrupt_test)
    assert replay(out / chosen["test_certificate"]["evidence"])["certified"] is (not corrupt_test)


def test_negative_control_rejects_copy(tmp_path):
    report = benchmark(tmp_path / "negative", generate(groups=(1, 1, 1)), lambda row: 0,
                       {"source": "constant-control"})
    assert report["selected_length"] == []
    assert report["results"]["raw_copy"]["scores"]["test"]["wrong"] == 6
    assert report["results"]["selected"]["scores"]["test"]["abstained"] == 6
    assert not report["results"]["raw_copy"]["test_certificate"]["certified"]


@pytest.mark.parametrize("defect", ["group", "context", "missing-reference"])
def test_datalog_rejects_leakage_or_missing_reference(tmp_path, defect):
    rows = generate(groups=(1, 1, 1))
    train = next(r for r in rows if r["part"] == "train")
    test = next(r for r in rows if r["part"] == "test")
    if defect == "group":
        test["group"] = train["group"]
    elif defect == "context":
        test["ctx"] = train["ctx"][:]
    refs = {r["id"]: r["expected"] for r in rows}
    if defect == "missing-reference":
        del refs[test["id"]]
    candidate = write_copy(tmp_path / "copy", 1)
    with pytest.raises(ValueError, match="invalid/leaking"):
        evaluate(candidate, rows, refs, tmp_path / "evidence", {})


@pytest.mark.parametrize("length,expected", [(1, 90), (2, 80)])
def test_runtime_relocated_without_model_or_python(tmp_path, length, expected):
    candidate = write_copy(tmp_path / "source", length)
    runtime = tmp_path / "relocated"
    runtime.mkdir()
    for filename in ("circuits.dl", "run.dl"):
        shutil.copyfile(candidate.parent / filename, runtime / filename)
    # Last one-token match has successor 90; last two-token match has successor 80.
    ctx = [1, 2, 80, 3, 2, 90, 1, 2]
    (runtime / "tok.facts").write_text("".join(f"0\t{i}\t{t}\n" for i, t in enumerate(ctx)) +
                                       "1\t0\t11\n1\t1\t12\n")
    subprocess.run(["souffle", "-F", str(runtime), "-D", str(runtime), str(runtime / "run.dl")],
                   check=True, capture_output=True, text=True)
    assert (runtime / "cdecide.csv").read_text().strip() == f"0\t{expected}"
    assert (runtime / "abstain.csv").read_text().strip() == "1"
