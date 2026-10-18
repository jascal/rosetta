"""The replay audit must be able to fail.

An audit that always passes is worse than no audit, because it converts an
unverified claim into a recorded one. These tests run the committed auditor over
run A's real retained evidence — where the correct answer is "all 25 verdicts
preserved, hashes unchanged" — and then corrupt copies to prove it reports
mismatches instead of absorbing them.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from audit_native_trace2 import audit_evidence, audit_freeze, audit_standalone

RUN_A = ROOT / "reference/benchmarks/qwen25_native_trace"
pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")


def test_auditor_reproduces_run_a_over_its_real_evidence():
    evidence = audit_evidence(RUN_A)
    assert len(evidence) == 25
    assert all(e["verdict_preserved"] for e in evidence), \
        [e for e in evidence if not e["verdict_preserved"]]
    assert all(e["all_relations_identical"] for e in evidence), \
        [e for e in evidence if not e["all_relations_identical"]]


def test_failed_equivalence_checks_stay_failed():
    # Run A certified nothing. If the auditor reported these as passing it would
    # be laundering the run's central negative result.
    evidence = audit_evidence(RUN_A)
    equiv = [e for e in evidence if "equiv-" in e["evidence"]]
    assert equiv, "expected the retained equivalence checks to still be there"
    assert all(e["certified"] is False for e in equiv), \
        [e for e in equiv if e["certified"]]


def test_standalone_check_runs_each_arm_in_souffle_alone():
    standalone = audit_standalone(RUN_A)
    assert [s["arm"] for s in standalone] == ["behavior", "trace", "shuffled"]
    for entry in standalone:
        assert entry["ran"], entry
        assert entry["only_program_and_token_domain_facts"] is True
        # every arm abstained on run A, so the standalone artifact must too
        assert entry["all_domain_instances_abstain"] is True


def test_frozen_hashes_are_unchanged():
    unchanged, detail = audit_freeze(RUN_A)
    assert unchanged is True, detail
    assert detail["checked_files"] > 0 and detail["checked_sources"] > 0


def test_auditor_detects_a_tampered_output(tmp_path):
    study = tmp_path / "study"
    (study / "certificate-evidence").mkdir(parents=True)
    source = next(p for p in (RUN_A / "certificate-evidence").glob("*/*")
                  if p.is_dir() and (p / "outputs" / "score.csv").exists())
    copy = study / "certificate-evidence" / "stage" / "check"
    shutil.copytree(source, copy)
    baseline = audit_evidence(study)
    assert all(e["all_relations_identical"] for e in baseline), baseline

    out = copy / "outputs" / "score.csv"
    out.write_text(out.read_text() + "999999\t999999\t0\n")
    after = audit_evidence(study)
    assert after and not after[0]["all_relations_identical"], after
    assert not after[0]["verdict_preserved"], after
    # Either path is a correct detection: the recorded-output comparison, or the
    # evidence manifest's own integrity check firing first.
    assert after[0]["mismatched_relations"] or after[0].get("error"), after


def test_auditor_detects_a_tampered_fact_file(tmp_path):
    study = tmp_path / "study"
    (study / "certificate-evidence").mkdir(parents=True)
    source = next(p for p in (RUN_A / "certificate-evidence").glob("*/*") if p.is_dir())
    copy = study / "certificate-evidence" / "stage" / "check"
    shutil.copytree(source, copy)
    facts = next((copy / "facts").glob("*.facts"))
    facts.write_text(facts.read_text() + "123456\t123456\n")
    after = audit_evidence(study)
    assert after and after[0]["verdict_preserved"] is False
    assert "certificate evidence changed" in after[0]["error"]


def test_frozen_audit_names_the_file_that_changed(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    frozen = json.loads((RUN_A / "frozen.json").read_text())
    frozen["sha256"] = {"dataset.json": "0" * 64}
    frozen["source_sha256"] = {}
    (study / "frozen.json").write_text(json.dumps(frozen))
    unchanged, detail = audit_freeze(study)
    assert unchanged is False
    assert "file:dataset.json" in detail["changed"]


def test_auditor_cli_exits_nonzero_on_mismatch(tmp_path):
    study = tmp_path / "study"
    (study / "certificate-evidence").mkdir(parents=True)
    source = next(p for p in (RUN_A / "certificate-evidence").glob("*/*")
                  if p.is_dir() and (p / "outputs" / "score.csv").exists())
    copy = study / "certificate-evidence" / "stage" / "check"
    shutil.copytree(source, copy)
    out = copy / "outputs" / "score.csv"
    out.write_text(out.read_text() + "999999\t999999\t0\n")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "py/audit_native_trace2.py"), str(study)],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["mismatches"]