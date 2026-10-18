"""Run-B driver.

Phase `inputs` generates the fresh splits from committed code. The remaining
phases mirror run A's staged I/O driver, but the two queries are
`native_trace_features2.dl` and `native_trace_select2.dl` and the splits are
select / sign / test rather than train / validation / test.

Model phases (`collect`, `final`) start a pinned fieldrun process and are not
implemented here yet; the offline phases below are, and every numeric or
statistical decision stays in Datalog.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import socket
import subprocess
import time
import urllib.error
import urllib.request

from benchmark_copy_generalization import generate
from benchmark_induction import digest, facts
from certificate import check, sha256
from native_trace import CHANNELS, DL, LAYERS, emit, grammar, program, write_json

ROOT = Path(__file__).resolve().parents[1]

SEEDS = (201, 202, 203)
LENGTHS = (8, 12, 24)
GROUPS = 2
SPLIT_OF_SEED = {201: "select", 202: "sign", 203: "test"}
RANKING_SPLIT = "sign"
ARMS = {0: "behavior", 1: "trace", 2: "shuffled"}

# The historical datasets run A excluded, per its own protocol.json. Run B must
# also exclude run A's frozen inputs, so no input is shared with the completed run.
HISTORICAL = (
    "reference/benchmarks/qwen25_05b/dataset.json",
    "reference/benchmarks/qwen25_05b_copy_generalization/dataset.json",
    "reference/benchmarks/qwen25_05b_guarded_seed1/dataset.json",
    "reference/benchmarks/qwen25_05b_int8_withheld_pairs/dataset.json",
    "reference/benchmarks/qwen25_05b_target_diagnostic/dataset.json",
    "reference/benchmarks/qwen25_05b_withheld_pairs/dataset.json",
)
RUN_A = (
    "reference/benchmarks/qwen25_native_trace/dataset.json",
    "reference/benchmarks/qwen25_native_trace/discovery.json",
)


def _rows(path):
    data = json.loads((ROOT / path).read_text())
    return data["rows"] if isinstance(data, dict) and "rows" in data else data


def _tokens(path):
    out = set()
    for row in _rows(path):
        out.update(row.get("ctx", ()))
    return out


def excluded_vocabulary():
    """Every token in the historical datasets, plus run A's own frozen inputs."""
    vocab = set()
    for path in HISTORICAL + RUN_A:
        vocab |= _tokens(path)
    return vocab


def build():
    exclude = excluded_vocabulary()
    raw = generate(seeds=SEEDS, lengths=LENGTHS, groups=GROUPS, exclude=exclude)

    rows = []
    for row in raw:
        seed = row["seed"]
        part = SPLIT_OF_SEED[seed]
        # generate() labels its two populations "test" (a copy context) and
        # "off_domain" (a negative control); run B records that as `negative`
        # and uses `part` only for the split.
        rows.append(dict(row, part=part, negative=(row["part"] == "off_domain")))
    rows.sort(key=lambda r: (list(SPLIT_OF_SEED).index(r["seed"]), r["id"]))
    for i, row in enumerate(rows):
        row["id"] = i
    return rows, exclude


def check_disjoint(rows, exclude):
    """Fail closed. A leak here would make the whole run incomparable to run A."""
    problems = []
    per_split_vocab, per_split_groups = {}, {}
    for row in rows:
        v = per_split_vocab.setdefault(row["part"], set())
        g = per_split_groups.setdefault(row["part"], set())
        v.update(row["ctx"])
        g.add(row["group"])
    names = sorted(per_split_vocab)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_tok = per_split_vocab[a] & per_split_vocab[b]
            shared_grp = per_split_groups[a] & per_split_groups[b]
            if shared_tok:
                problems.append(f"vocabulary shared between {a} and {b}: {sorted(shared_tok)[:8]}")
            if shared_grp:
                problems.append(f"groups shared between {a} and {b}: {sorted(shared_grp)[:4]}")
    for part, v in per_split_vocab.items():
        clash = v & exclude
        if clash:
            problems.append(f"{part} reuses excluded tokens: {sorted(clash)[:8]}")

    counts = {}
    for part in names:
        batch = [r for r in rows if r["part"] == part]
        counts[part] = dict(n=len(batch), negative=sum(r["negative"] for r in batch))
        if counts[part] != {"n": 114, "negative": 18}:
            problems.append(f"{part} is {counts[part]}, expected 114 rows with 18 negatives")
    return problems, counts


def read(path):
    return json.loads(path.read_text())


def input_gate(out):
    """Refuse to proceed unless the committed inputs are the ones on disk."""
    protocol = read(out / "protocol.json")
    if protocol.get("problems"):
        raise ValueError("input generation reported problems")
    if protocol["dataset_sha256"] != sha256(out / "dataset.json"):
        raise ValueError("dataset.json does not match protocol.json")
    if protocol["hypotheses_sha256"] != sha256(out / "hypotheses.dl"):
        raise ValueError("hypotheses.dl does not match protocol.json")
    if protocol["prereg_sha256"] != sha256(ROOT / "docs/native-trace-prereg2.md"):
        raise ValueError("preregistration changed")


def features(out):
    """Fit feature parameters on the SELECT split only.

    One layer/channel per run, as in run A, but the selection contrast is
    positive-copy-context vs negative control and the result carries gate_mean.
    The select split's references are never consulted here: the class comes from
    the `negative` flag, not from model agreement.

    Reads the full trace sweep (`select-traces/<id>.npz`, keys `<layer>_<channel>`)
    because every neuron must be scored, not just the probe units.
    """
    import numpy as np

    input_gate(out)
    rows = [r for r in read(out / "dataset.json") if r["part"] == "select"]
    chosen = []
    for layer in LAYERS:
        for channel in CHANNELS:
            # the SwiGLU gate is what opens the unit; a product channel is gated
            # by the gate channel at the same neuron
            gate_channel = "gate" if channel == "product" else channel
            column, gate_column, width = [], [], None
            for row in rows:
                path = out / "select-traces" / f"{row['id']}.npz"
                with np.load(path) as archive:
                    values = archive[f"{layer}_{channel}"]
                    opening = archive[f"{layer}_{gate_channel}"]
                if width is None:
                    width = len(values)
                if len(values) != width or len(opening) != width:
                    raise ValueError("neuron width changed")
                column.extend(f"{row['id']}\t{n}\t{float(v):.17g}\n"
                              for n, v in enumerate(values))
                gate_column.extend(f"{row['id']}\t{n}\t{float(v):.17g}\n"
                                   for n, v in enumerate(opening))
            staged = {
                "sample": "".join(f"{r['id']}\t{int(r['negative'])}\n" for r in rows),
                "unit": "".join(f"{n}\n" for n in range(width)),
                "observation": "".join(column),
                "gate": "".join(gate_column),
            }
            result = check(DL / "native_trace_features2.dl", out / "hypotheses.dl", staged, {})
            if not result["certified"] or result.get("error"):
                raise ValueError(f"feature query failed: {result}")
            gate_mean = {int(n): float(m) for n, m in result["relations"].get("gate_mean", [])}
            flagged = {int(n) for n, _ in result["relations"].get("unweighted", [])}
            for neuron, score, threshold, polarity in result["relations"]["chosen"]:
                neuron = int(neuron)
                chosen.append({
                    "id": len(chosen), "layer": layer, "channel": channel,
                    "gate_channel": gate_channel, "neuron": neuron,
                    "score": float(score), "threshold": float(threshold),
                    "polarity": int(polarity), "width": width,
                    "gate_mean": gate_mean.get(neuron), "unweighted": neuron in flagged,
                })
            for reason, npos, nneg in result["relations"].get("no_vote", []):
                print(f"  no vote ({reason}) at layer {layer} channel {channel}: "
                      f"{npos} positive / {nneg} negative", flush=True)
    write_json(out / "features.json", chosen)


def select(out):
    """Rank candidates on the SIGN split; select split fits parameters only."""
    input_gate(out)
    rows = [r for r in read(out / "dataset.json") if r["part"] in ("select", RANKING_SPLIT)]
    # References for BOTH staged splits. Selecting on sign only would leave every
    # select row unreferenced, which select2 correctly rejects as invalid.
    refs = {}
    for split in ("select", RANKING_SPLIT):
        path = out / f"{split}-references.json"
        if path.exists():
            refs.update({int(k): v for k, v in read(path).items()})
    missing = [r["id"] for r in rows if r["id"] not in refs]
    if missing:
        raise ValueError(f"missing references for staged rows: {missing[:8]}")
    features_ = read(out / "features.json")
    records = {r["id"]: r for r in
               map(json.loads, (out / f"{RANKING_SPLIT}-observations.jsonl").read_text().splitlines())}
    signing = [r for r in rows if r["part"] == RANKING_SPLIT]
    staged = facts(rows, refs)
    staged["sample"] = "".join(f"{r['id']}\t{r['group']}\t{r['part']}\t{digest(r['ctx'])}\n" for r in rows)
    staged["negmark"] = "".join(f"{r['id']}\n" for r in rows if r["negative"])
    staged["pair"] = "".join(f"{r['base']}\t{r['id']}\n" for r in rows if "base" in r)
    staged["feature"] = "".join(
        f"{f['id']}\t{f['threshold']:.17g}\t{f['polarity']}\t{f['gate_mean']:.17g}\n"
        for f in features_ if not f["unweighted"])
    trace, gate = "", ""
    for r in signing:
        last = len(r["ctx"]) - 1
        units = {(o["layer"], o["neuron"]): o for o in records[r["id"]]["observations"]
                 if o["position"] == last}
        for f in features_:
            if f["unweighted"]:
                continue
            unit = units.get((f["layer"], f["neuron"]))
            if unit is None:
                raise ValueError(f"missing probe unit L{f['layer']} n{f['neuron']} for {r['id']}")
            trace += f"{r['id']}\t{f['id']}\t{float(unit[f['channel']]):.17g}\n"
            gate += f"{r['id']}\t{f['id']}\t{float(unit[f['gate_channel']]):.17g}\n"
    # Registered shuffle: seed 204, a genuine permutation of the signing rows,
    # not a shift. A shift is not independent of the row ordering, which would
    # let the control inherit part of the trace arm's correlation.
    permuted = random.Random(204).sample([r["id"] for r in signing], len(signing))
    staged["trace"], staged["gate"] = trace, gate
    staged["permutation"] = "".join(f"{r['id']}\t{j}\n" for r, j in zip(signing, permuted))
    result = check(DL / "native_trace_select2.dl", out / "hypotheses.dl", staged, {})
    if not result["certified"] or result.get("error"):
        raise ValueError(f"selection query failed: {result}")
    write_json(out / "selection.json", result)
    if result["relations"]["degenerate"]:
        summary = result["relations"].get("signature_summary", [["?", "?", "?", "?"]])[0]
        print(f"  NON-DEGENERACY GATE FIRED: signature {summary[1]}/{summary[0]} good, "
              f"{summary[3]}/{summary[2]} negative controls good. Ranking did not run.")
        return
    summary = result["relations"]["signature_summary"][0]
    print(f"  signature {summary[1]}/{summary[0]} good, {summary[3]}/{summary[2]} negatives good")
    # Emit each arm's composite and every admitted individual. An empty selection
    # still emits, and still abstains: the Souffle-only artifact is the deliverable.
    for arm, name in ARMS.items():
        selected = [(int(rank), int(c)) for a, rank, c in result["relations"]["admitted"]
                    if int(a) == arm]
        emit(out / "artifacts" / name, selected)
        for rank, c in selected:
            emit(out / "artifacts" / f"{name}-candidate-{c}", [(rank, c)])


def model_gate(out, args):
    """The oracle must be the preflight-validated binary and bundle, unchanged."""
    report = read(args.preflight / "report.json")
    if not report["passed"]:
        raise ValueError("upstream native activation/intervention gate failed")
    identity = {
        "binary_sha256": sha256(args.binary),
        "bundle_sha256": {s: sha256(str(args.bundle) + s)
                          for s in (".fieldrun.bin", ".fieldrun.json")},
        "preflight_report_sha256": sha256(args.preflight / "report.json"),
        "reference_sha256": sha256(args.preflight / "reference.json"),
    }
    if identity["binary_sha256"] != report["binary_sha256"] or \
            identity["bundle_sha256"] != report["bundle_sha256"]:
        raise ValueError("model/binary differs from validated oracle")
    if (out / "model.json").exists():
        if read(out / "model.json") != identity:
            raise ValueError("model identity changed within experiment")
    else:
        write_json(out / "model.json", identity)


def post(port, ids, probe):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/probe/mlp",
        json.dumps({"ids": ids, "probe": probe}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    if "error" in result:
        raise ValueError(result)
    return result


@contextmanager
def server(out, args, stage):
    model_gate(out, args)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise ValueError("requested port already occupied; cannot attest another process")
    command = [str(args.binary.resolve()), "--bundle", str(args.bundle.resolve()),
               "--serve", str(args.port)]
    with (out / f"server-{stage}.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise ValueError(f"fieldrun exited {process.returncode}; see {log.name}")
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                        if json.load(r).get("ok"):
                            break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() - started > 300:
                        raise TimeoutError("fieldrun did not become ready")
                    time.sleep(0.2)
            write_json(out / f"server-{stage}.json",
                       {"command": command, "pid": process.pid,
                        "model_sha256": sha256(out / "model.json")})
            yield args.port
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def collect(out, args, stage):
    """Collect references, and native evidence where the phase needs it.

    select: a full sweep of every neuron at LAYERS, because feature selection
            scores all of them.
    sign:   the chosen units only, and only after features.json exists.
    test:   references only, and only after the freeze boundary.
    """
    import numpy as np

    input_gate(out)
    if stage == "test":
        validate_freeze(out)
    rows = [r for r in read(out / "dataset.json") if r["part"] == stage]
    if stage == "select":
        probe = {"layers": list(LAYERS)}
    elif stage == "sign":
        chosen = read(out / "features.json")
        probe = {"units": sorted({(f["layer"], f["neuron"]) for f in chosen})}
    else:
        probe = {}
    journal = out / f"{stage}-observations.jsonl"
    observations = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            if record["id"] in observations:
                raise ValueError("duplicate observation")
            observations[record["id"]] = record
    expected = {r["id"] for r in rows}
    if not set(observations) <= expected:
        raise ValueError("unexpected observation IDs")
    with server(out, args, stage) as port, journal.open("a") as stream:
        for j, row in enumerate(rows):
            i = row["id"]
            if i in observations:
                if observations[i]["input_sha256"] != digest(row["ctx"]):
                    raise ValueError("cached context changed")
                continue
            start = time.monotonic()
            answer = post(port, row["ctx"], probe)
            native = answer.pop("observations")
            record = {"id": i, "input_sha256": digest(row["ctx"]),
                      "seconds": time.monotonic() - start, **answer}
            if stage == "select":
                # The probe contract returns the LAST input position. Filter to it
                # explicitly: concatenating every returned position would silently
                # build one feature vector out of mixed positions, which reads as a
                # plausible-looking number rather than as an error.
                last = len(row["ctx"]) - 1
                vectors = {f"{l}_{c}": np.asarray(
                    [o[c] for o in native if o["layer"] == l and o["position"] == last],
                    dtype=np.float32)
                    for l in LAYERS for c in CHANNELS}
                tracefile = out / "select-traces" / f"{i}.npz"
                tracefile.parent.mkdir(exist_ok=True)
                np.savez_compressed(tracefile, **vectors)
                record.update(trace_file=str(tracefile.relative_to(out)),
                              trace_sha256=sha256(tracefile))
            elif stage == "sign":
                record["observations"] = native
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            observations[i] = record
            print(f"{stage}: {j+1}/{len(rows)}", flush=True)
    if set(observations) != expected:
        raise ValueError("incomplete observations")
    write_json(out / f"{stage}-references.json",
               {str(i): r["next"] for i, r in observations.items()})


def prediction_query(out, name, candidate, rows):
    verifier = out / "prediction-verifier.dl"
    verifier.write_text((DL / "guard_domain.dl").read_text() + "\n.output cdecide\n")
    staged = {"tok": facts(rows, {})["tok"],
              "sample": "".join(f"{r['id']}\n" for r in rows)}
    result = check(verifier, candidate, staged, {"ndomain": int})
    return result


def freeze(out):
    """Hash every input, decision, native observation and artifact at the boundary."""
    input_gate(out)
    if (out / "frozen.json").exists() or (out / "test-observations.jsonl").exists():
        raise ValueError("freeze already exists or test data collection began")
    if not read(out / "selection.json")["certified"]:
        raise ValueError("complete the selection query before freeze")
    rows = [r for r in read(out / "dataset.json") if r["part"] == "test"]
    artifacts = {}
    for path in sorted((out / "artifacts").glob("*/circuits.dl")):
        name = path.parent.name
        result = prediction_query(out, f"domain-{name}", path, rows)
        artifacts[name] = {
            "path": str(path.relative_to(out)), "sha256": sha256(path),
            "domain": [int(i) for i in result["relations"]["domain"]],
            "predictions": result["relations"]["cdecide"],
            "bytes": path.stat().st_size,
            "rule_lines": sum(":-" in line for line in path.read_text().splitlines())}
    if not set(ARMS.values()) <= set(artifacts):
        raise ValueError("missing arm composite")
    locked = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
              if p.is_file() and not p.name.startswith("server-")}
    sources = [Path(__file__), ROOT / "py/native_trace.py", ROOT / "py/certificate.py",
               ROOT / "py/benchmark_induction.py", ROOT / "py/benchmark_copy_generalization.py",
               ROOT / "docs/native-trace-prereg2.md",
               *sorted(DL.glob("native_trace_*.dl")), DL / "equiv.dl", DL / "guard_domain.dl"]
    write_json(out / "frozen.json", {
        "artifacts": artifacts, "sha256": locked,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "note": "No test oracle reference was requested before this manifest."})


def validate_freeze(out):
    frozen = read(out / "frozen.json")
    for name, expected in frozen["sha256"].items():
        if sha256(out / name) != expected:
            raise ValueError(f"frozen file changed: {name}")
    for name, expected in frozen["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    return frozen


def evaluate(out):
    """Equivalence checks over firing and full domains, then the summary report."""
    frozen = validate_freeze(out)
    rows = [r for r in read(out / "dataset.json") if r["part"] == "test"]
    refs = {int(k): v for k, v in read(out / "test-references.json").items()}
    certs, verdicts, predictions = {}, [], []
    for name, artifact in frozen["artifacts"].items():
        path = out / artifact["path"]
        certs[name] = {}
        for scope, ids in (("firing", artifact["domain"]), ("full", [r["id"] for r in rows])):
            subset = [r for r in rows if r["id"] in set(ids)]
            staged = facts(subset, refs)
            staged["domain"] = "".join(f"{i}\n" for i in ids)
            certs[name][scope] = check(
                DL / "equiv.dl", path, staged,
                {"ndomain": int, "nmiss": int, "nuncov": int,
                 "nmissing": int, "ninvalid": int})
        firing = certs[name]["firing"]
        verdicts.append(f"{name}\t" + "\t".join(
            str(firing[k]) for k in
            ("ndomain", "nmiss", "nuncov", "nmissing", "ninvalid"))
            + f"\t{int(firing['certified'])}\n")
        predictions.extend(f"{name}\t{i}\t{t}\n" for i, t in artifact["predictions"])
    staged = {"artifact": "".join(f"{n}\n" for n in frozen["artifacts"]),
              "sample": "".join(f"{r['id']}\t{int(r['negative'])}\n" for r in rows),
              "ref": facts(rows, refs)["ref"],
              "prediction": "".join(predictions), "verdict": "".join(verdicts)}
    summary = check(DL / "native_trace_report.dl", out / "hypotheses.dl", staged, {})
    costs = []
    for stage in ("select", "sign", "test"):
        journal = out / f"{stage}-observations.jsonl"
        if journal.exists():
            costs.extend(f"{stage}\t{i}\t{json.loads(line)['seconds']:.17g}\n"
                         for i, line in enumerate(journal.read_text().splitlines()))
    cost = check(DL / "native_trace_cost.dl", out / "hypotheses.dl",
                 {"observation": "".join(costs)})
    write_json(out / "report.json", {
        "tag": "empirical",
        "scope": "pinned f32 bundle, frozen finite domains, select/sign/test splits",
        "frozen_sha256": sha256(out / "frozen.json"),
        "summary": summary, "observation_cost": cost,
        "artifacts": frozen["artifacts"], "certificates": certs})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("inputs", "collect", "features", "select",
                                     "freeze", "final", "evaluate"))
    p.add_argument("out", type=Path)
    p.add_argument("--write", action="store_true",
                   help="for phase=inputs: write the study files; otherwise only report")
    p.add_argument("--binary", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--preflight", type=Path,
                   default=ROOT / "reference/benchmarks/native_trace_preflight_b")
    p.add_argument("--port", type=int, default=8392)
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.phase == "inputs":
        return _inputs(a)
    if a.phase in ("collect", "final"):
        if not (a.binary and a.bundle):
            p.error("oracle stages need --binary and --bundle")
        collect(a.out, a, "test" if a.phase == "final" else a.phase)
    else:
        {"features": features, "select": select, "freeze": freeze, "evaluate": evaluate}[a.phase](a.out)
    return 0


def _inputs(a):
    rows, exclude = build()
    problems, counts = check_disjoint(rows, exclude)

    report = {
        "seeds": list(SEEDS),
        "lengths": list(LENGTHS),
        "groups": GROUPS,
        "split_of_seed": SPLIT_OF_SEED,
        "excluded_tokens": len(exclude),
        "excluded_from": list(HISTORICAL + RUN_A),
        "counts": counts,
        "problems": problems,
    }
    if not a.write:
        print(json.dumps(report, indent=2))
        return 1 if problems else 0

    if problems:
        for line in problems:
            print(f"refusing to write: {line}")
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "dataset.json").write_text(json.dumps(rows, indent=1) + "\n")
    # The fixed preregistered candidate universe, generated rather than committed
    # as an opaque artifact. 24 bases x 56 guard compositions = 1,344 candidates.
    bases, universe = grammar()
    (a.out / "hypotheses.dl").write_text(
        "// Fixed preregistered universe. Selection, routing and admission stay in Datalog.\n"
        ".decl base(b:number,kind:number,k:number,m:number)\n"
        ".decl candidate(c:number,b:number,a:number,d:number)\n"
        + "".join(f"base({','.join(map(str, r))}).\n" for r in bases)
        + "".join(f"candidate({','.join(map(str, r))}).\n" for r in universe))
    protocol = dict(report)
    protocol["dataset_sha256"] = sha256(a.out / "dataset.json")
    protocol["hypotheses_sha256"] = sha256(a.out / "hypotheses.dl")
    protocol["prereg_sha256"] = sha256(ROOT / "docs/native-trace-prereg2.md")
    protocol["ncandidates"] = len(universe)
    protocol["nbase"] = len(bases)
    protocol["note"] = ("Generated by py/run_native_trace2.py inputs before any "
                        "run-B model request. No token is shared with the "
                        "historical datasets or with run A's frozen inputs.")
    (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({k: protocol[k] for k in
                      ("seeds", "split_of_seed", "excluded_tokens", "counts",
                       "ncandidates", "dataset_sha256", "hypotheses_sha256")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
