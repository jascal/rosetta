"""Run D: the ablation plan, the effect summary, and the CLI against a stub oracle.

Authored controls, not run-D evidence. They exercise the arm structure, the
fail-closed checks, the flip arithmetic, the identity and consistency
instruments, and the whole inputs -> collect -> plan -> freeze -> intervene ->
evaluate path -- before any of it runs against the real checkpoint.

The stub oracle is declared here and is NOT the model. It flips the answer only
when one particular layer's whole MLP is zeroed, so the H-B decision path can
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
import run_mlp_ablation as rund

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")

# A small stand-in model: 3 layers, 4 neurons each.
LAYERS = [0, 1, 2]
WIDTH = 4
CUMUL = [1, 2, 3]
WRITEBACK = [0, 2]
FLIP_LAYER = 1

STUB = '''#!/usr/bin/env python3
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[sys.argv.index("--serve") + 1])
WIDTH = __WIDTH__
FLIP_LAYER = __FLIP_LAYER__

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
            for layer in probe.get("layers", []):
                for neuron in range(WIDTH):
                    observations.append({
                        "layer": layer, "position": position, "neuron": neuron,
                        "gate": 0.5 + neuron, "value": 2.0 + neuron,
                        "product": (0.5 + neuron) * (2.0 + neuron)})
        # Zeroing this layer's whole MLP moves the model; nothing else does.
        zeroed = set()
        for p in probe.get("patches", []):
            if p["value"] == 0.0:
                zeroed.add((p["layer"], p["neuron"]))
        answer = 900 + ids[-1]
        if all((FLIP_LAYER, n) in zeroed for n in range(WIDTH)):
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
    source = STUB.replace("__WIDTH__", str(WIDTH)).replace("__FLIP_LAYER__", str(FLIP_LAYER))
    binary.write_text(source.lstrip())
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
    p = subprocess.run([sys.executable, str(ROOT / "py/run_mlp_ablation.py"),
                        args[0], str(out), *args[1:]], capture_output=True, text=True)
    assert (p.returncode == 0) == ok, p.stdout + p.stderr
    return p


def _study(tmp_path):
    """A tiny run-D study: three rows, three layers, width four."""
    out = tmp_path / "run"
    out.mkdir()
    rows = [{"id": i, "ctx": [i, i + 1, i + 2], "expected": 900 + i,
             "group": f"g{i % 2}", "part": "eval", "seed": 301,
             "negative": i == 2, "kind": "repeat", "layout": "plain",
             "exposures": 2, "length": 3, "prefix": 1,
             "gkey": f"seed301-length3-{i % 2}"} for i in range(3)]
    (out / "dataset.json").write_text(json.dumps(rows, indent=1) + "\n")
    # The stub answers 900 + the last context token, which is id + 2 here.
    (out / "references.json").write_text(
        json.dumps({str(r["id"]): 900 + r["id"] + 2 for r in rows}) + "\n")
    (out / "base.json").write_text(json.dumps(
        {str(r["id"]): {"ref_logit": 1.5, "target_logit": 1.0} for r in rows}) + "\n")
    (out / "donor.json").write_text(json.dumps({"0": 2, "1": 0, "2": 1}) + "\n")
    (out / "protocol.json").write_text(json.dumps({
        "seed": 301,
        "source_dataset_sha256": sha256(rund.RUN_C / "dataset.json"),
        "prereg_sha256": sha256(ROOT / "docs/native-trace-mlp-prereg.md"),
        "layers": LAYERS, "cumulative": CUMUL, "writeback_layers": WRITEBACK,
        "width": WIDTH, "counts": {"n": len(rows)},
    }, indent=2) + "\n")
    return out, rows


def test_plan_arms_are_exactly_as_registered(tmp_path):
    out, rows = _study(tmp_path)
    facts = rund.arm_facts(out)
    result = check(DL / "mlp_ablation_plan.dl", DL / "mlp_ablation_plan.dl", facts, {})
    assert result["certified"], result.get("error")
    rel = result["relations"]
    assert rel["invalid"] == []
    # 3 sweep + 3 cumulative + 2 ident + 2 donor requests per row.
    assert len(rel["req"]) == 10 * len(rows)
    assert rel["narms"] == [["3"]]
    per = {}
    for (i, arm, key) in rel["req"]:
        per.setdefault((arm, str(key)), 0)
        per[(arm, str(key))] += 1
    assert per[("sweep", "1")] == len(rows)
    assert per[("cumul", "3")] == len(rows)
    # The cumulative arm's request patches a prefix of the layers.
    layers = {}
    for (i, arm, key, l) in rel["req_layer"]:
        layers.setdefault((arm, str(key)), set()).add(int(l))
    assert layers[("cumul", "3")] == {0, 1, 2}
    assert layers[("cumul", "1")] == {0}
    assert layers[("sweep", "1")] == {1}


def test_plan_fails_closed_on_a_broken_arm(tmp_path):
    out, rows = _study(tmp_path)
    facts = rund.arm_facts(out)
    # A sweep that misses a layer.
    broken = dict(facts, sweep_arm="0\n1\n")
    assert not check(DL / "mlp_ablation_plan.dl", DL / "mlp_ablation_plan.dl", broken, {})["certified"]
    # A cumulative arm that is not a prefix.
    broken = dict(facts, cumul_member="1\t0\n2\t0\n2\t2\n3\t0\n3\t1\n3\t2\n")
    assert not check(DL / "mlp_ablation_plan.dl", DL / "mlp_ablation_plan.dl", broken, {})["certified"]
    # A neuron universe with a gap.
    broken = dict(facts, neuron="".join(f"{l}\t{n}\n" for l in LAYERS for n in range(WIDTH)
                                        if not (l == 1 and n == 2)))
    assert not check(DL / "mlp_ablation_plan.dl", DL / "mlp_ablation_plan.dl", broken, {})["certified"]
    # A write-back arm on a layer the other write-back arm does not use.
    broken = dict(facts, donor_arm="0\n")
    assert not check(DL / "mlp_ablation_plan.dl", DL / "mlp_ablation_plan.dl", broken, {})["certified"]


def _summary_facts(flip_sweep=None, flip_joint=False, flip_full=None, break_identity=False, missing=None):
    rows = [(0, 0, 101), (1, 1, 202), (2, 0, 303)]
    ref = {i: t for (i, c, t) in rows}
    donor = {0: 1, 1: 0, 2: 1}
    reqs = []
    for (i, c, t) in rows:
        for l in LAYERS:
            reqs.append((i, "sweep", l))
        for k in CUMUL:
            reqs.append((i, "cumul", k))
        for l in WRITEBACK:
            reqs.append((i, "ident", l))
            reqs.append((i, "donor", l))
    if flip_full is not None:
        for (i, c, t) in rows:
            reqs.append((i, "full", flip_full))
    results = []
    for (i, c, t) in rows:
        for (i2, a, k) in [x for x in reqs if x[0] == i]:
            if missing is not None and (i2, a, k) == missing:
                continue
            answer = t
            if a == "sweep" and flip_sweep is not None and k == flip_sweep:
                answer = 999
            if a == "cumul" and k == max(CUMUL) and flip_joint:
                answer = 999
            if a == "full" and flip_full is not None:
                answer = 999
            if a == "ident" and break_identity:
                answer = 777
            results.append((i2, a, k, answer, 1.0, 2.0, 3.0))
    return {
        "result": "".join(f"{i}\t{a}\t{k}\t{x}\t{al}\t{rl}\t{tl}\n" for (i, a, k, x, al, rl, tl) in results),
        "req": "".join(f"{i}\t{a}\t{k}\n" for (i, a, k) in reqs),
        "req_layer": "".join(f"{i}\t{a}\t{k}\t{l}\n" for (i, a, k) in reqs
                             for l in (range(k) if a == "cumul" else [k])),
        "ref": "".join(f"{i}\t{t}\n" for i, t in ref.items()),
        "row": "".join(f"{i}\t{c}\t{t}\n" for (i, c, t) in rows),
        "base": "".join(f"{i}\t5.0\t1.0\n" for (i, c, t) in rows),
        "donor": "".join(f"{d}\t{i}\n" for i, d in donor.items()),
        "cumul_arm": "".join(f"{k}\n" for k in CUMUL),
    }


def _summary(facts):
    return check(DL / "mlp_ablation_summary.dl", DL / "mlp_ablation_summary.dl", facts, {})


def test_summary_reports_each_registered_verdict():
    null = _summary(_summary_facts())
    assert null["certified"], null.get("error")
    assert null["relations"]["h_a_supported"]
    assert not null["relations"]["h_b_supported"]
    assert null["relations"]["instrument_ok"]
    assert null["relations"]["consistency_ok"]

    localized = _summary(_summary_facts(flip_sweep=1))
    assert localized["relations"]["h_b_supported"]
    assert localized["relations"]["sweep_flip"] == [["1"]]

    aggregate = _summary(_summary_facts(flip_joint=True))
    assert aggregate["relations"]["h_b_aggregate"]
    assert not aggregate["relations"]["h_a_supported"]

    positional = _summary(_summary_facts(flip_full=1))
    assert positional["relations"]["h_c_supported"]
    assert not positional["relations"]["h_a_supported"]


def test_summary_breaks_on_a_broken_instrument_or_a_missing_result():
    broken = _summary(_summary_facts(break_identity=True))
    assert broken["certified"]
    assert broken["relations"]["instrument_ok"] == []
    missing = _summary(_summary_facts(missing=(2, "sweep", 0)))
    assert not missing["certified"]
    assert missing["relations"]["invalid"]


def test_only_the_stage2_arm_covers_every_position(tmp_path):
    """The 0/1 allpos column arrives as text: "0" is truthy, so it must be
    compared numerically. Every stage-1 arm patches the answer position only."""
    out, rows = _study(tmp_path)
    facts = rund.arm_facts(out)
    result = check(DL / "mlp_ablation_plan.dl", DL / "mlp_ablation_plan.dl", facts, {})
    assert result["certified"], result.get("error")
    by_arm = {}
    for (i, arm, key, layer, mode, allpos) in result["relations"]["plan"]:
        by_arm.setdefault(arm, set()).add(int(allpos))
    assert by_arm["sweep"] == {0}
    assert by_arm["cumul"] == {0}
    assert by_arm["ident"] == {0}
    assert by_arm["donor"] == {0}
    # With a stage-2 layer registered, that arm alone covers every position.
    facts = rund.arm_facts(out, stage2_layers=(1,))
    result = check(DL / "mlp_ablation_plan.dl", DL / "mlp_ablation_plan.dl", facts, {})
    assert result["certified"], result.get("error")
    by_arm = {}
    for (i, arm, key, layer, mode, allpos) in result["relations"]["plan"]:
        by_arm.setdefault(arm, set()).add(int(allpos))
    assert by_arm["full"] == {1}


def test_patch_expansion_covers_exactly_the_neuron_universe(tmp_path):
    out, rows = _study(tmp_path)
    request = {"i": 0, "arm": "sweep", "key": 1, "layers": [1], "mode": "zero", "allpos": False}
    own = {1: {0: [1.0] * WIDTH}}
    patches = rund.build_patches(request, rows[0], own, {}, WIDTH)
    assert len(patches) == WIDTH
    assert {p["neuron"] for p in patches} == set(range(WIDTH))
    assert all(p["value"] == 0.0 and p["layer"] == 1 for p in patches)
    # A write-back arm whose observation file is short must fail closed.
    writeback = {"i": 0, "arm": "ident", "key": 1, "layers": [1], "mode": "own", "allpos": False}
    with pytest.raises(ValueError, match="expected"):
        rund.build_patches(writeback, rows[0], {1: {0: [1.0] * (WIDTH - 1)}}, {}, WIDTH)
    # And it must expand to the whole universe when the file is complete.
    assert len(rund.build_patches(writeback, rows[0], own, {}, WIDTH)) == WIDTH


def test_cli_workflow_localizes_a_layer_and_keeps_the_instruments(tmp_path):
    out, rows = _study(tmp_path)
    args = _oracle(tmp_path, _free_port())
    flags = ["--binary", str(args.binary), "--bundle", str(args.bundle),
             "--preflight", str(args.preflight), "--port", str(args.port)]
    # The plan is a function of the registered arms only, so it can be built
    # before the collection; the write-back arms read their values at
    # intervention time and fail closed if the observation file is short.
    cli(out, "plan")
    cli(out, "collect", *flags)
    plan_doc = json.loads((out / "plan.json").read_text())
    assert plan_doc["nrequests"] == 10 * len(rows)
    (out / "interventions.jsonl").write_text("")
    cli(out, "freeze", ok=False)
    (out / "interventions.jsonl").unlink()
    cli(out, "freeze")
    frozen = json.loads((out / "frozen.json").read_text())
    assert frozen["status"] == "ready"
    assert "interventions.jsonl" not in frozen["sha256"]
    cli(out, "intervene", *flags)
    cli(out, "evaluate")
    report = json.loads((out / "report.json").read_text())
    assert report["tag"] == "empirical"
    assert report["verdict"]["instrument_ok"]
    assert report["verdict"]["consistency_ok"]
    # The stub moves the model only when that one layer's whole MLP is zeroed.
    assert report["verdict"]["h_b_layer_localized"]
    assert not report["verdict"]["h_a_not_the_mlp_path"]
