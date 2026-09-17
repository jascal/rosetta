"""Certificate obligations and retained evidence, exercised against real Souffle."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from certificate import replay  # noqa: E402
from oracle import run_equiv, run_master  # noqa: E402
from temperature import certify_T, check_argmax, emit_T, finalize, trange  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")


def crisp(tmp_path, extra=""):
    p = tmp_path / "candidate.dl"
    p.write_text(".decl cdecide(inst:number,out:number)\ncdecide(I,7) :- tok(I,_,_).\n" + extra)
    return p


@pytest.mark.parametrize("contexts,refs,missing", [([[1]], [None], 1), ([[1], []], [7, None], 1), ([], [], 0)])
def test_exact_rejects_missing_and_empty_domain(tmp_path, contexts, refs, missing):
    r = run_equiv(crisp(tmp_path), contexts, refs)
    assert "error" not in r, r
    assert not r["certified"]
    assert r["ndomain"] == len(contexts)
    assert r["nmissing"] == missing


def test_exact_rejects_reference_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="reference count"):
        run_equiv(crisp(tmp_path), [[1], [2]], [7])


def test_master_rejects_missing_and_empty_domain():
    assert not run_master([[1]], [None], 1)["certified"]
    assert not run_master([], [], 1)["certified"]
    assert run_master([[1]], [7], 1)["certified"]


def test_exact_rejects_extra_and_ambiguous_predictions(tmp_path):
    r = run_equiv(crisp(tmp_path, "cdecide(99,7).\n"), [[1]], [7])
    assert not r["certified"] and r["ninvalid"] == 1
    r = run_equiv(crisp(tmp_path, "cdecide(0,8).\n"), [[1]], [7])
    assert not r["certified"] and r["nmiss"] == 1


def test_evidence_replays_and_detects_tampering(tmp_path):
    r = run_equiv(crisp(tmp_path), [[1]], [7], evidence_dir=tmp_path / "evidence")
    assert r["certified"], r
    assert replay(r["evidence"])["certified"]
    facts = Path(r["evidence"]) / "facts" / "ref.facts"
    facts.write_text("0\t8\n")
    with pytest.raises(ValueError, match="evidence changed"):
        replay(r["evidence"])


def distribution(tmp_path, logits=None):
    p = tmp_path / "circuits.dl"
    emit_T(str(p), {(1,): logits or [(7, 0.0), (8, 0.0)]}, 1)
    return p


def test_argmax_adapter_replays_and_rejects_ties(tmp_path):
    p = distribution(tmp_path, [(7, 1.0), (8, 0.0)])
    r = check_argmax(p, [[1]], {0: 7}, [0], 0.5, evidence_dir=tmp_path / "evidence")
    assert r["certified"] and r["nmatch"] == 1, r
    assert replay(r["evidence"])["certified"]
    p = distribution(tmp_path)
    r = check_argmax(p, [[1]], {0: 7}, [0], 0.5)
    assert not r["certified"] and r["nmiss"] == 1, r


@pytest.mark.parametrize("refs,idxs", [({}, [0]), ({}, [])])
def test_argmax_adapter_rejects_missing_and_empty_domain(tmp_path, refs, idxs):
    r = check_argmax(distribution(tmp_path), [[1]], refs, idxs, 0.5)
    assert "error" not in r and not r["certified"], r


@pytest.mark.parametrize("missing_logits", [False, True])
def test_unified_cover_preserves_complete_certificate_obligations(tmp_path, monkeypatch, missing_logits):
    import exercise_confirm
    import temperature
    from types import SimpleNamespace

    (tmp_path / "corpus.json").write_text(json.dumps({"ids": [7, 8]}))
    (tmp_path / "logit_cache.json").write_text(json.dumps({} if missing_logits else {"7": [[8, 0.0]]}))
    monkeypatch.delenv("FIELDRUN_SERVE", raising=False)
    monkeypatch.setattr(exercise_confirm, "instances", lambda *a: [[7]])
    monkeypatch.setattr(exercise_confirm, "induction_exercise", lambda *a, **kw: ([[50, 51, 50]], [0]))
    monkeypatch.setattr(exercise_confirm, "CIRCUITS", {})
    monkeypatch.setattr(temperature, "build_sym", lambda *a: {})
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: SimpleNamespace(ids=[]))
    result = exercise_confirm.emit_unified_cover(str(tmp_path), lambda c: 51, lambda cs: None,
                                                 tokenizer, 100, T_lo=1.0, T=1.0)
    assert result["certified"] is (not missing_logits)
    report = json.loads((tmp_path / "unified-certificate.json").read_text())
    assert not report["interval_certified"]
    assert report["checks"][0]["ndomain"] == 1
    assert report["checks"][0]["nmissing"] == int(missing_logits)
    for check in report["checks"]:
        assert replay(check["evidence"])["certified"] == check["certified"]


@pytest.mark.parametrize("crisp_mode", [False, True])
@pytest.mark.parametrize("missing_reference", [False, True])
def test_idiom_cli_certifies_requested_domain(tmp_path, monkeypatch, crisp_mode, missing_reference):
    import idiom_learn

    (tmp_path / "corpus.json").write_text(json.dumps({"ids": [1, 2, 3]}))
    cache = {"1": [[7, 0.0]]}
    if not missing_reference:
        cache["2"] = [[7, 0.0]]
    (tmp_path / "logit_cache.json").write_text(json.dumps(cache))
    monkeypatch.delenv("FIELDRUN_SERVE", raising=False)
    monkeypatch.setattr(sys, "argv", ["idiom_learn.py", "2", "1", str(tmp_path), "--certify"]
                        + (["--crisp"] if crisp_mode else []))
    monkeypatch.setattr(idiom_learn, "model_refs", lambda *a: [7, None if missing_reference else 7])
    monkeypatch.setattr(idiom_learn, "ref_source", lambda *a: ("test", lambda c: 7))
    # Isolate certificate wiring from candidate search; use the real emitter and Datalog checker.
    for name in ("learn_gates", "learn_compose", "learn_relational"):
        monkeypatch.setattr(idiom_learn, name, lambda *a, **kw: [])
    monkeypatch.setattr(idiom_learn, "learn_skeleton", lambda *a, **kw: ([], set()))
    assert idiom_learn.main() == int(missing_reference)
    checks = list((tmp_path / "certificate-evidence").glob("*/certificate.json"))
    assert len(checks) == (1 if crisp_mode else 3)
    for path in checks:
        result = json.loads(path.read_text())["result"]
        assert result["ndomain"] == 2
        assert result["nmissing"] == int(missing_reference)
        assert result["certified"] is (not missing_reference)


def test_distribution_tv_computed_in_datalog(tmp_path):
    p = distribution(tmp_path, [(7, math.log(3)), (8, 0.0)])
    refs = {0: [(7, 0.0), (8, 0.0)]}
    r = certify_T(p, [[1]], refs, [0], 1.0, .2)
    assert "error" not in r, r
    assert r["worst"] == pytest.approx(.25)
    assert not r["certified"] and r["nmiss"] == 1
    assert certify_T(p, [[1]], refs, [0], 1.0, .3)["certified"]


@pytest.mark.parametrize("refs,idxs", [({}, [0]), ({0: [(7, 0.0)]}, [0, 1]), ({}, [])])
def test_distribution_never_drops_missing_obligations(tmp_path, refs, idxs):
    r = certify_T(distribution(tmp_path), [[1], [1]], refs, idxs, 1.0, .02)
    assert "error" not in r, r
    assert not r["certified"]
    assert r["ndomain"] == len(idxs)


def test_distribution_extra_support_and_strict_error_boundary(tmp_path):
    # Disjoint reference and candidate supports: TV = 1, even though each normalizes.
    p = distribution(tmp_path, [(9, 0.0)])
    r = certify_T(p, [[1]], {0: [(7, 0.0)]}, [0], 1.0, 1.0)
    assert not r["certified"] and r["worst"] == 1.0, r


def test_distribution_rejects_invalid_probability_mass(tmp_path):
    p = tmp_path / "bad.dl"
    p.write_text(".decl cdist(inst:number,token:number,p:float)\ncdist(0,7,0.99).\n")
    r = certify_T(p, [[1]], {0: [(7, 0.0)]}, [0], 1.0, .02)
    assert not r["certified"] and r["ninvalid"] == 1, r


@pytest.mark.parametrize("temperature,epsilon", [(0.0, .02), (-1.0, .02), (1.0, 0.0), (1.0, 2.0)])
def test_distribution_rejects_invalid_check_parameters(tmp_path, temperature, epsilon):
    p = tmp_path / "constant.dl"
    p.write_text(".decl cdist(inst:number,token:number,p:float)\ncdist(0,7,1.0).\n")
    r = certify_T(p, [[1]], {0: [(7, 0.0)]}, [0], temperature, epsilon)
    assert not r["certified"], r


def test_distribution_rejects_conflicting_reference_logits(tmp_path):
    r = certify_T(distribution(tmp_path), [[1]], {0: [(7, 0.0), (7, 1.0)]}, [0], 1.0, .9)
    assert not r["certified"] and r["ninvalid"] == 1, r


def test_finalize_retains_finite_grid_and_scoped_evidence(tmp_path):
    refs = {0: [(7, 0.0), (8, 0.0)]}
    ok = finalize(str(tmp_path), [[1]], refs, [0], [], {(1,): refs[0]}, set(), False,
                  1, {}, "synthetic", 1.0, .02, .7, "synthetic top-K oracle")
    assert ok
    report = json.loads((tmp_path / "certificate.json").read_text())
    assert report["temperatures"] == [.7, .85, 1.0]
    assert not report["interval_certified"] and not report["symbol_twin_certified"]
    assert "omitted model probability mass is not bounded" in report["scope"]
    assert all(replay(tmp_path / r["evidence"])["certified"] for r in report["checks"])


def test_temperature_grid_never_rounds_midpoint_outside_interval():
    assert all(.0001 <= t <= .0002 for t in trange(.0001, .0002))


def test_cache_only_cli_fails_on_missing_references(tmp_path):
    (tmp_path / "corpus.json").write_text(json.dumps({"ids": [1, 2, 3]}))
    proc = subprocess.run([sys.executable, str(ROOT / "py" / "temperature.py"), "2", "1", str(tmp_path)],
                          capture_output=True, text=True)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "certificate.json").read_text())
    assert report["domain"] == [0, 1]
    assert all(not r["certified"] and r["nmissing"] == 2 for r in report["checks"])


@pytest.mark.parametrize("returncode", [0, 1])
def test_failed_or_incomplete_souffle_never_returns_a_certificate(tmp_path, monkeypatch, returncode):
    import certificate
    monkeypatch.setattr(certificate.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, returncode, "", "failed"))
    r = run_equiv(crisp(tmp_path), [[1]], [7])
    assert not r["certified"] and r["error"]


@pytest.mark.parametrize("method", ["decide", "logits"])
def test_failed_oracle_discards_partial_output(tmp_path, monkeypatch, method):
    import oracle
    import split_facts
    weights = tmp_path / "weights"
    weights.mkdir()
    monkeypatch.setattr(split_facts, "split", lambda _: (str(tmp_path / "forward.dl"), str(weights)))
    monkeypatch.setattr(oracle, "compiled", lambda _: None)

    def failed_forward(_program, _facts, out):
        (Path(out) / "decide.csv").write_text("7\n")
        (Path(out) / "logit.csv").write_text("7\t1.0\n")
        return subprocess.CompletedProcess([], 1, "", "partial forward failed")

    monkeypatch.setattr(oracle, "_run", failed_forward)
    assert getattr(oracle, method)("unused-whole.dl", [1]) is None


def test_standalone_runtime_abstains_without_model_files(tmp_path):
    distribution(tmp_path)
    facts = tmp_path / "facts"
    output = tmp_path / "out"
    facts.mkdir()
    output.mkdir()
    (facts / "tok.facts").write_text("0\t0\t1\n1\t0\t99\n")
    (facts / "temp.facts").write_text("1.0\n")
    proc = subprocess.run(["souffle", "run.dl", "-F", "facts", "-D", "out"], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (output / "abstain.csv").read_text().strip() == "1"
    assert all(row.startswith("0\t") for row in (output / "cdist.csv").read_text().splitlines())
