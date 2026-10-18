"""The oracle attestation, and proof that it can refuse.

Run B introduces no new oracle: it runs the same validated `0948cbe` executable
and the same f32 bundle as run A, so run A's upstream FP32/eager preflight is
run B's gate. That is only sound if the identity is actually checked, so these
tests cover both the matching case and — more importantly — the refusal paths.

The identity check against the real pinned binary is skipped when that binary is
not on this machine. It is a /tmp artefact, not something CI can have; the
refusal-path tests run unconditionally.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from run_native_trace2 import model_gate

RUN_A = ROOT / "reference/benchmarks/qwen25_native_trace"
PREFLIGHT = ROOT / "reference/benchmarks/native_trace_preflight"
PINNED = Path("/tmp/rosetta-native-qwen-f32")

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")


class Args:
    """Stands in for the parsed CLI arguments model_gate consumes."""
    def __init__(self, binary, bundle, preflight=PREFLIGHT):
        self.binary = binary
        self.bundle = bundle
        self.preflight = preflight


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def oracle(tmp_path):
    """A study directory plus a stand-in binary/bundle whose hashes we control."""
    out = tmp_path / "study"
    out.mkdir()
    binary = tmp_path / "fieldrun"
    binary.write_text("#!/bin/sh\nexit 0\n")
    stem = tmp_path / "model"
    (tmp_path / "model.fieldrun.bin").write_bytes(b"weights")
    (tmp_path / "model.fieldrun.json").write_bytes(b'{"arch":"rope"}')
    return out, Args(binary, stem)


def _preflight_matching(tmp_path, binary, stem):
    """A preflight report whose recorded hashes match the stand-in oracle."""
    directory = tmp_path / "preflight"
    directory.mkdir()
    (directory / "reference.json").write_text("{}\n")
    (directory / "report.json").write_text(json.dumps({
        "passed": True,
        "binary_sha256": _sha(binary),
        "bundle_sha256": {
            ".fieldrun.bin": _sha(str(stem) + ".fieldrun.bin"),
            ".fieldrun.json": _sha(str(stem) + ".fieldrun.json"),
        },
    }, indent=2) + "\n")
    return directory


def test_matching_oracle_is_recorded_and_repeatable(oracle, tmp_path):
    out, args = oracle
    args.preflight = _preflight_matching(tmp_path, args.binary, args.bundle)
    model_gate(out, args)
    recorded = json.loads((out / "model.json").read_text())
    assert recorded["binary_sha256"] == _sha(args.binary)
    # a second call must agree; identity changing mid-experiment is the failure
    model_gate(out, args)
    assert json.loads((out / "model.json").read_text()) == recorded


def test_refuses_when_the_binary_differs(oracle, tmp_path):
    out, args = oracle
    args.preflight = _preflight_matching(tmp_path, args.binary, args.bundle)
    model_gate(out, args)
    args.binary.write_text("#!/bin/sh\nexit 1\n")           # a different build
    with pytest.raises(ValueError, match="differs from validated oracle"):
        model_gate(out, args)


def test_refuses_when_the_bundle_differs(oracle, tmp_path):
    out, args = oracle
    args.preflight = _preflight_matching(tmp_path, args.binary, args.bundle)
    model_gate(out, args)
    (tmp_path / "model.fieldrun.bin").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="differs from validated oracle"):
        model_gate(out, args)


def test_refuses_a_failed_preflight(oracle, tmp_path):
    out, args = oracle
    directory = _preflight_matching(tmp_path, args.binary, args.bundle)
    report = json.loads((directory / "report.json").read_text())
    report["passed"] = False
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    args.preflight = directory
    with pytest.raises(ValueError, match="gate failed"):
        model_gate(out, args)


def test_refuses_when_identity_changes_mid_experiment(oracle, tmp_path):
    out, args = oracle
    args.preflight = _preflight_matching(tmp_path, args.binary, args.bundle)
    model_gate(out, args)
    # simulate a run that swapped the oracle after recording it
    recorded = json.loads((out / "model.json").read_text())
    recorded["bundle_sha256"][".fieldrun.bin"] = "0" * 64
    (out / "model.json").write_text(json.dumps(recorded, indent=2) + "\n")
    with pytest.raises(ValueError, match="changed within experiment"):
        model_gate(out, args)


def test_run_a_oracle_identity_is_unchanged_and_reused():
    """Run B inherits run A's oracle; that claim is only checkable from the record."""
    recorded = json.loads((RUN_A / "model.json").read_text())
    assert recorded["binary_sha256"] == (
        "54de8fc8fd5acec025a0cda6bd5a73202f08b4d2d3e30c0ea982c6c24567ea62")
    assert recorded["bundle_sha256"] == {
        ".fieldrun.bin": "3a2fb8e5062e9f281dffd9919eb81e164d5001b22395b1c87e2467cc67f77f8d",
        ".fieldrun.json": "5c013a3a24ef13276d3c4d3fbd0a9007525bf4e9d720608b8db0176a391aaa12",
    }
    # and the preflight it attests to is still exactly the one on disk
    assert _sha(PREFLIGHT / "report.json") == recorded["preflight_report_sha256"]
    assert _sha(PREFLIGHT / "reference.json") == recorded["reference_sha256"]


@pytest.mark.skipif(not (PINNED / "fieldrun-0948cbe").exists(),
                    reason="pinned oracle copy is not on this machine")
def test_pinned_oracle_on_this_machine_matches_the_preflight():
    binary = PINNED / "fieldrun-0948cbe"
    stem = PINNED / "model"
    report = json.loads((PREFLIGHT / "report.json").read_text())
    assert _sha(binary) == report["binary_sha256"]
    for suffix in (".fieldrun.bin", ".fieldrun.json"):
        assert _sha(str(stem) + suffix) == report["bundle_sha256"][suffix]