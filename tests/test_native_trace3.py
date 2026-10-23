"""Run C: the plan, the summary, and the CLI against a declared synthetic oracle.

These are authored controls, not run-C evidence. They exercise every part of the
experiment that the offline suite could otherwise reach for the first time during
the irreversible collection: the arm shapes of the plan, its fail-closed checks,
the flip arithmetic of the summary, the identity instrument, and the whole
inputs -> collect -> plan -> freeze -> intervene -> evaluate path.

The stub oracle is declared here and is NOT the model: it flips the answer only
when it receives the registered treated coordinates, so the H2 decision path can
be exercised without a checkpoint.
"""
import json
import shutil
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from certificate import check, sha256
from native_trace import DL
import run_native_trace3 as runc

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")

REGISTERED = [(7, 1898), (15, 788), (23, 412), (7, 3), (15, 680), (23, 1650),
              (7, 1835), (15, 992), (23, 1792)]

STUB = '''#!/usr/bin/env python3
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[sys.argv.index("--serve") + 1])
TREATED = REGISTERED
N_NEURONS = 4

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send({"ok": True})
        self.send_error(404)

    def do_POST(self):
        if self.path != "/probe/mlp":
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        ids, probe = request["ids"], request.get("probe", {})
        observations = []
        for position, token in enumerate(ids):
            for layer, neuron in probe.get("units", []):
                observations.append({
                    "layer": layer, "position": position, "neuron": neuron,
                    "gate": 0.5 + neuron, "value": 2.0 + neuron,
                    "product": (0.5 + neuron) * (2.0 + neuron)})
        # The treated coordinates, when zeroed as a group, move the model.
        patches = probe.get("patches", [])
        hit = [p for p in patches if (p["layer"], p["neuron"]) in TREATED and p["value"] == 0.0]
        answer = 900 + ids[-1]
        if len(hit) >= 3:
            answer = 42
        return self._send({"next": answer, "next_logit": 1.5,
                           "observations": observations,
                           "logits": [[t, 2.0] for t in probe.get("readouts", [])]})

HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Args:
    def __init__(self, binary, bundle, preflight, port):
        self.binary, self.bundle, self.preflight, self.port = binary, bundle, preflight, port


def _oracle(tmp_path, port):
    binary = tmp_path / "stub-fieldrun"
    binary.write_text(STUB.replace("REGISTERED", repr(REGISTERED)).lstrip())
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    stem = tmp_path / "model"
    (tmp_path / "model.fieldrun.bin").write_bytes(b"weights")
    (tmp_path / "model.fieldrun.json").write_bytes(b'{"arch":"rope"}')
    pre = tmp_path / f"preflight-{port}"
    pre.mkdir(exist_ok=True)
    (pre / "reference.json").write_text("{}\n")
    (pre / "report.json").write_text(json.dumps({
        "passed": True, "binary_sha256": sha256(binary),
        "bundle_sha256": {".fieldrun.bin": sha256(str(stem) + ".fieldrun.bin"),
                          ".fieldrun.json": sha256(str(stem) + ".fieldrun.json")},
    }, indent=2) + "\n")
    return Args(binary, stem, pre, port)


def cli(out, *args, ok=True):
    p = subprocess.run([sys.executable, str(ROOT / "py/run_native_trace3.py"),
                        args[0], str(out), *args[1:]], capture_output=True, text=True)
    assert (p.returncode == 0) == ok, p.stdout + p.stderr
    return p


def _plan_facts(units, rows, obs, rset):
    return {
        "unit": "".join(f"{rank}\t{l}\t{n}\n" for rank, l, n in units),
        "row": "".join(f"{i}\t{cls}\t{g}\t{ln}\n" for (i, cls, g, ln) in rows),
        "obs": "".join(f"{i}\t{l}\t{n}\t{v:.17g}\n" for (i, l, n, v) in obs),
        "rset": "".join(f"{k}\t{l}\t{n}\n" for (k, l, n) in rset),
        "klist": "".join(f"{k}\n" for k in range(1, 10)),
    }


def _synthetic_plan(tmp_path):
    units = [(rank, l, n) for rank, (l, n) in enumerate(REGISTERED, start=1)]
    rows = [(0, 0, "s-l8-0", 30), (1, 1, "s-l8-0", 12),
            (2, 0, "s-l12-1", 46), (3, 1, "s-l12-1", 26)]
    obs = [(i, l, n, 0.5 * i + 0.01 * n) for (i, _, _, _) in rows for (_, l, n) in units]
    widths = {l: 4864 for l in (7, 15, 23)}
    rset = runc.random_sets(widths)
    return units, rows, obs, rset


def test_treatment_is_run_b_frozen_evidence_and_refuses_drift(tmp_path, monkeypatch):
    units, widths = runc.treatment()
    assert [(u["layer"], u["neuron"]) for u in units] == REGISTERED
    assert widths == {7: 4864, 15: 4864, 23: 4864}
    drifted = tmp_path / "features.json"
    drifted.write_text(json.dumps([{"layer": 7, "neuron": 1, "width": 4864}]))
    monkeypatch.setattr(runc, "RUN_B_FEATURES", drifted)
    monkeypatch.setattr(runc, "RUN_B_FEATURES_SHA256", sha256(drifted))
    with pytest.raises(ValueError, match="differ from the registered set"):
        runc.treatment()


def test_plan_arms_are_exactly_as_preregistered(tmp_path):
    units, rows, obs, rset = _synthetic_plan(tmp_path)
    result = check(DL / "native_trace_runc_plan.dl", DL / "native_trace_runc_plan.dl",
                   _plan_facts(units, rows, obs, rset), {})
    assert result["certified"], result.get("error")
    rel = result["relations"]
    assert rel["invalid"] == []
    requests = rel["request"]
    patches = rel["patch"]
    # 21 requests per row: 9 zero, 9 random, mean, ident, one write.
    assert len(requests) == 21 * len(rows)
    per_request = {}
    for (i, mode, k, l, n, v) in patches:
        per_request.setdefault((i, mode, k), []).append((l, n))
    for k in map(str, range(1, 10)):
        assert len(per_request[("0", "zero", k)]) == int(k)
        assert len(per_request[("0", "rand", k)]) == int(k)
    assert len(per_request[("0", "mean", "9")]) == 9
    assert len(per_request[("0", "ident", "9")]) == 9
    assert len(per_request[("0", "write_control_into_copy", "0")]) == 9
    assert len(per_request[("1", "write_copy_into_control", "0")]) == 9
    # Z_3 is one neuron per layer: the dose-response is not depth-confounded.
    assert sorted(per_request[("0", "zero", "3")]) == [("15", "788"), ("23", "412"), ("7", "1898")]
    # Random sets are layer-matched to the treated sets and avoid them.
    treated = set(REGISTERED)
    for k in map(str, range(1, 10)):
        z = set(per_request[("0", "zero", k)])
        r = set(per_request[("0", "rand", k)])
        assert not (r & treated)
        assert len(r) == int(k)
        layers = lambda s: sorted(l for l, _ in s)  # noqa: E731
        assert layers(z) == layers(r)


def test_plan_fails_closed_on_a_tampered_control_set(tmp_path):
    units, rows, obs, rset = _synthetic_plan(tmp_path)
    facts = _plan_facts(units, rows, obs, rset)
    # A random set that touches a treated coordinate must invalidate the plan.
    facts["rset"] += f"1\t7\t1898\n"
    assert not check(DL / "native_trace_runc_plan.dl", DL / "native_trace_runc_plan.dl",
                     facts, {})["certified"]
    # A random set that is not layer-matched must invalidate the plan.
    facts = _plan_facts(units, rows, obs, rset)
    facts["rset"] = facts["rset"].replace("1\t7\t", "1\t15\t", 1)
    assert not check(DL / "native_trace_runc_plan.dl", DL / "native_trace_runc_plan.dl",
                     facts, {})["certified"]
    # A row missing one coordinate's value must invalidate the plan.
    facts = _plan_facts(units, rows, obs, rset)
    facts["obs"] = "".join(facts["obs"].splitlines(True)[:-1])
    assert not check(DL / "native_trace_runc_plan.dl", DL / "native_trace_runc_plan.dl",
                     facts, {})["certified"]


def _summary_facts(flip_zero_from=None, write_to_donor=False, break_identity=False, drop=None):
    rows = [(0, 0, 101), (1, 1, 202), (2, 0, 303), (3, 1, 404)]
    ref = {i: t for (i, c, t) in rows}
    requests = []
    for (i, c, t) in rows:
        for k in range(1, 10):
            requests.append((i, "zero", k))
        for k in range(1, 10):
            requests.append((i, "rand", k))
        requests.append((i, "mean", 9))
        requests.append((i, "ident", 9))
        requests.append((i, "write_control_into_copy" if c == 0 else "write_copy_into_control", 0))
    donor = {0: 1, 1: 0, 2: 3, 3: 2}
    results = []
    for (i, c, t) in rows:
        for (i2, m, k) in [x for x in requests if x[0] == i]:
            if drop is not None and (i2, m, k) == drop:
                continue
            answer = t
            if flip_zero_from is not None and m == "zero" and k >= flip_zero_from:
                answer = 999
            if write_to_donor and m == "write_copy_into_control":
                answer = ref[donor[i]]
            if break_identity and m == "ident":
                answer = 777
            results.append((i2, m, k, answer, 1.0, 2.0, 3.0))
    return {
        "result": "".join(f"{i}\t{m}\t{k}\t{a}\t{al}\t{rl}\t{tl}\n" for (i, m, k, a, al, rl, tl) in results),
        "base": "".join(f"{i}\t5.0\t1.0\n" for (i, c, t) in rows),
        "ref": "".join(f"{i}\t{t}\n" for i, t in ref.items()),
        "row": "".join(f"{i}\t{c}\t{t}\n" for (i, c, t) in rows),
        "donor": "".join(f"{d}\t{i}\n" for i, d in donor.items()),
        "request": "".join(f"{i}\t{m}\t{k}\n" for (i, m, k) in requests),
    }


def _summary(facts):
    return check(DL / "native_trace_runc_summary.dl", DL / "native_trace_runc_summary.dl",
                 facts, {})


def test_summary_reports_each_registered_verdict():
    null = _summary(_summary_facts())
    assert null["certified"], null.get("error")
    assert null["relations"]["h1_supported"]
    assert not null["relations"]["h2_supported"]
    assert not null["relations"]["h_suf_supported"]
    assert null["relations"]["instrument_ok"]

    redundancy = _summary(_summary_facts(flip_zero_from=3))
    assert redundancy["relations"]["h2_supported"]
    assert not redundancy["relations"]["h1_supported"]

    sufficient = _summary(_summary_facts(write_to_donor=True))
    assert sufficient["relations"]["h_suf_supported"]
    assert not sufficient["relations"]["h2_supported"]


def test_summary_breaks_on_a_broken_identity_and_on_a_missing_result():
    broken = _summary(_summary_facts(break_identity=True))
    assert broken["certified"]
    assert broken["relations"]["instrument_ok"] == []
    missing = _summary(_summary_facts(drop=(2, "mean", 9)))
    assert not missing["certified"]
    assert missing["relations"]["invalid"]


def test_cli_workflow_freezes_the_plan_before_any_intervention(tmp_path):
    out = tmp_path / "run"
    cli(out, "inputs", "--write")
    rows = json.loads((out / "dataset.json").read_text())
    assert len(rows) == 114
    assert sum(r["negative"] for r in rows) == 18
    args = _oracle(tmp_path, _free_port())
    flags = ["--binary", str(args.binary), "--bundle", str(args.bundle),
             "--preflight", str(args.preflight), "--port", str(args.port)]

    # The plan cannot be built before the unpatched collection exists.
    cli(out, "plan", ok=False)
    cli(out, "collect", *flags)
    assert len(json.loads((out / "references.json").read_text())) == 114
    cli(out, "plan")
    plan_doc = json.loads((out / "plan.json").read_text())
    assert plan_doc["nrequests"] == 21 * len(rows)
    assert len(plan_doc["donor"]) == len(rows)
    # A plan may not be frozen after interventions exist.
    (out / "interventions.jsonl").write_text("")
    cli(out, "freeze", ok=False)
    (out / "interventions.jsonl").unlink()
    cli(out, "freeze")
    frozen = json.loads((out / "frozen.json").read_text())
    assert frozen["status"] == "ready"
    # A frozen study refuses new inputs and new collections.
    cli(out, "inputs", "--write", ok=False)
    cli(out, "collect", *flags, ok=False)

    cli(out, "intervene", *flags)
    report_rows = [json.loads(line) for line in (out / "interventions.jsonl").read_text().splitlines()]
    assert len(report_rows) == 21 * len(rows)
    cli(out, "evaluate")
    report = json.loads((out / "report.json").read_text())
    assert report["tag"] == "empirical"
    assert report["verdict"]["instrument_ok"]
    # The stub moves the model only for a group ablation of the treated set.
    assert report["verdict"]["h2_redundancy"]
    assert not report["verdict"]["h1_off_path"]


def test_cli_refuses_to_reuse_a_changed_treatment(tmp_path):
    out = tmp_path / "run"
    cli(out, "inputs", "--write")
    protocol = json.loads((out / "protocol.json").read_text())
    protocol["treatment_sha256"] = "0" * 64
    (out / "protocol.json").write_text(json.dumps(protocol))
    p = cli(out, "plan", ok=False)
    assert "treatment digest" in p.stderr
