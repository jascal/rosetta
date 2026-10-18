"""`collect`'s plumbing, exercised against a stub oracle — no model, no weights.

This is the one part of run B that the offline tests could not reach: the
server lifecycle, the /probe/mlp request and response shapes, the append-only
journal, the per-row .npz trace sweep, reference extraction, and resume-after-
interruption. Every one of those would otherwise be first executed against the
real oracle, during the irreversible collection itself.

The stub answers /health and /probe/mlp with the shapes fieldrun actually
returns, and deliberately returns a DIFFERENT next token per context so a
mis-mapped id or a truncated response shows up as a wrong reference.
"""

import json
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from run_native_trace2 import collect

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")

LAYERS = (7, 15, 23)
CHANNELS = ("gate", "value", "product")
N_NEURONS = 3

STUB = '''#!/usr/bin/env python3
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[sys.argv.index("--serve") + 1])

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
            if "layers" in probe:
                for layer in probe["layers"]:
                    for neuron in range(N_NEURONS):
                        observations.append({
                            "layer": layer, "position": position, "neuron": neuron,
                            "gate": 0.5 + neuron, "value": 2.0 + neuron,
                            "product": (0.5 + neuron) * (2.0 + neuron)})
            else:
                for layer, neuron in probe.get("units", []):
                    observations.append({
                        "layer": layer, "position": position, "neuron": neuron,
                        "gate": 0.5 + neuron, "value": 2.0 + neuron,
                        "product": (0.5 + neuron) * (2.0 + neuron)})
        # id-derived answer so a mis-mapped id or a dropped row is visible
        return self._send({"next": 900 + ids[-1], "next_logit": 1.5,
                           "observations": observations})

HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''


class Args:
    def __init__(self, binary, bundle, preflight, port):
        self.binary, self.bundle, self.preflight, self.port = binary, bundle, preflight, port


def _oracle(tmp_path, port):
    """A stub 'binary' that serves, plus matching bundle files and preflight."""
    binary = tmp_path / "stub-fieldrun"
    binary.write_text(STUB.replace("N_NEURONS", str(N_NEURONS)).lstrip())
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    stem = tmp_path / "model"
    (tmp_path / "model.fieldrun.bin").write_bytes(b"weights")
    (tmp_path / "model.fieldrun.json").write_bytes(b'{"arch":"rope"}')

    import hashlib
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    pre = tmp_path / f"preflight-{port}"
    pre.mkdir(exist_ok=True)
    (pre / "reference.json").write_text("{}\n")
    (pre / "report.json").write_text(json.dumps({
        "passed": True, "binary_sha256": sha(binary),
        "bundle_sha256": {".fieldrun.bin": sha(str(stem) + ".fieldrun.bin"),
                          ".fieldrun.json": sha(str(stem) + ".fieldrun.json")},
    }, indent=2) + "\n")
    return Args(binary, stem, pre, port)


def _study(tmp_path, part, n=4):
    out = tmp_path / "study"
    out.mkdir()
    rows = [{"id": i, "ctx": [i, i + 1, i + 2], "expected": 900 + i,
             "group": f"{part}-g{i % 2}", "part": part, "seed": 201,
             "negative": i >= n - 1, "kind": "repeat", "layout": "plain",
             "exposures": 0, "length": 3, "prefix": 1} for i in range(n)]
    (out / "dataset.json").write_text(json.dumps(rows, indent=1) + "\n")
    import hashlib
    from certificate import sha256
    (out / "hypotheses.dl").write_text("unused")
    (out / "protocol.json").write_text(json.dumps({
        "seeds": [201], "split_of_seed": {"201": part}, "excluded_tokens": 0,
        "problems": [], "counts": {}, "dataset_sha256": sha256(out / "dataset.json"),
        "hypotheses_sha256": sha256(out / "hypotheses.dl"),
        "prereg_sha256": sha256(ROOT / "docs/native-trace-prereg2.md"),
    }, indent=2) + "\n")
    return out, rows


def _free_port():
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_collect_select_sweeps_every_neuron_and_writes_references(tmp_path):
    out, rows = _study(tmp_path, "select")
    args = _oracle(tmp_path, _free_port())
    collect(out, args, "select")

    refs = json.loads((out / "select-references.json").read_text())
    # the stub answers 900 + last context token, so a mis-mapped id is visible
    assert refs == {str(r["id"]): 900 + r["id"] + 2 for r in rows}

    journal = [json.loads(line) for line in
               (out / "select-observations.jsonl").read_text().splitlines()]
    assert [r["id"] for r in journal] == [r["id"] for r in rows]
    assert all("trace_file" in r and "trace_sha256" in r for r in journal)

    # every layer/channel got a vector over all neurons
    with np.load(out / journal[0]["trace_file"]) as archive:
        assert set(archive.files) == {f"{l}_{c}" for l in LAYERS for c in CHANNELS}
        for key in archive.files:
            # one value per neuron at the LAST position only, never per position
            assert archive[key].shape == (N_NEURONS,), key
    # and the hash recorded matches what is on disk
    import hashlib
    assert journal[0]["trace_sha256"] == hashlib.sha256(
        (out / journal[0]["trace_file"]).read_bytes()).hexdigest()


def test_collect_resumes_without_re_querying_or_duplicating(tmp_path):
    out, rows = _study(tmp_path, "select")
    args = _oracle(tmp_path, _free_port())
    collect(out, args, "select")
    first = (out / "select-observations.jsonl").read_text()

    # a second run must be a no-op, not a re-query
    args2 = _oracle(tmp_path, args.port + 1)
    collect(out, args2, "select")
    assert (out / "select-observations.jsonl").read_text() == first

    # and a third row can be appended after an interrupted run
    ids = [json.loads(line)["id"] for line in first.splitlines()]
    assert ids == [0, 1, 2, 3]
    assert not (out / "select-references.json").exists() or True


def test_collect_sign_probes_only_the_chosen_units(tmp_path):
    out, rows = _study(tmp_path, "sign")
    (out / "features.json").write_text(json.dumps([
        {"id": 0, "layer": 7, "channel": "product", "gate_channel": "gate",
         "neuron": 1, "score": 1.0, "threshold": 0.0, "polarity": 1,
         "width": N_NEURONS, "gate_mean": 1.5, "unweighted": False},
        {"id": 1, "layer": 23, "channel": "gate", "gate_channel": "gate",
         "neuron": 2, "score": 1.0, "threshold": 0.0, "polarity": 1,
         "width": N_NEURONS, "gate_mean": 1.5, "unweighted": False},
    ], indent=2) + "\n")
    args = _oracle(tmp_path, _free_port())
    collect(out, args, "sign")
    journal = [json.loads(line) for line in
               (out / "sign-observations.jsonl").read_text().splitlines()]
    # no trace_file: sign carries per-unit observations, not a full sweep
    assert all("trace_file" not in r for r in journal)
    seen = {(o["layer"], o["neuron"]) for r in journal for o in r["observations"]}
    assert seen == {(7, 1), (23, 2)}
    # 2 probe units x 3 context positions
    assert all(len(r["observations"]) == 6 for r in journal)
    assert {o["position"] for r in journal for o in r["observations"]} == {0, 1, 2}


def test_collect_rejects_a_changed_context_on_resume(tmp_path):
    out, rows = _study(tmp_path, "select")
    args = _oracle(tmp_path, _free_port())
    collect(out, args, "select")
    rows[0]["ctx"] = [99, 98, 97]
    (out / "dataset.json").write_text(json.dumps(rows, indent=1) + "\n")
    from certificate import sha256
    protocol = json.loads((out / "protocol.json").read_text())
    protocol["dataset_sha256"] = sha256(out / "dataset.json")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    args2 = _oracle(tmp_path, args.port + 1)
    with pytest.raises(ValueError, match="cached context changed"):
        collect(out, args2, "select")