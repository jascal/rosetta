"""Staged I/O driver for docs/native-trace-prereg.md; no Python rule selection.

Stages: discovery, features, development, select, freeze, final, evaluate,
diagnostic. Model stages start their own pinned fieldrun process. A frozen
manifest is mandatory before any final oracle request.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import socket
import subprocess
import time
import urllib.error
import urllib.request

from benchmark_guarded_copy import guard_program
from benchmark_induction import digest, facts
from certificate import check, sha256
from native_trace import CHANNELS, DL, LAYERS, ROOT, emit, write_json

ARMS = {0: "behavior", 1: "trace", 2: "shuffled"}


def read(path):
    return json.loads(path.read_text())


def checked(out, name, verifier, candidate, staged, *, scalars=None, allow_failure=False):
    """Persist a Datalog query and its verdict. Admission is never reconstructed."""
    destination = out / "certificate-evidence" / name
    result = check(verifier, candidate, staged, scalars or {}, evidence_dir=destination,
                   provenance={"stage": name, "source": "native-trace experiment"})
    result["evidence"] = str(Path(result["evidence"]).relative_to(out))
    write_json(out / f"{name}.json", result)
    if result.get("error") or (not result["certified"] and not allow_failure):
        raise ValueError(f"{name} failed: {result.get('error', 'Datalog rejected inputs')}")
    return result


def input_gate(out):
    protocol = read(out / "protocol.json")
    for name in ("dataset", "discovery"):
        if sha256(out / f"{name}.json") != protocol[f"{name}_sha256"]:
            raise ValueError(f"frozen {name} changed")
    if sha256(out / "hypotheses.dl") != protocol["hypotheses_sha256"]:
        raise ValueError("candidate universe changed")
    if sha256(ROOT / "docs/native-trace-prereg.md") != protocol["prereg_sha256"]:
        raise ValueError("preregistration changed")


def model_gate(out, args):
    report = read(args.preflight / "report.json")
    if not report["passed"]:
        raise ValueError("upstream native activation/intervention gate failed")
    identity = {"binary_sha256": sha256(args.binary),
                "bundle_sha256": {s: sha256(str(args.bundle)+s) for s in (".fieldrun.bin", ".fieldrun.json")},
                "preflight_report_sha256": sha256(args.preflight / "report.json"),
                "reference_sha256": sha256(args.preflight / "reference.json")}
    if identity["binary_sha256"] != report["binary_sha256"] or identity["bundle_sha256"] != report["bundle_sha256"]:
        raise ValueError("model/binary differs from validated oracle")
    if (out / "model.json").exists():
        if read(out / "model.json") != identity:
            raise ValueError("model identity changed within experiment")
    else:
        write_json(out / "model.json", identity)


def post(port, ids, probe):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/probe/mlp",
        json.dumps({"ids": ids, "probe": probe}).encode(), {"Content-Type": "application/json"})
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
    command = [str(args.binary.resolve()), "--bundle", str(args.bundle.resolve()), "--serve", str(args.port)]
    with (out / f"server-{stage}.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise ValueError(f"fieldrun exited {process.returncode}; see {log.name}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                        if json.load(r).get("ok"):
                            break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic()-started > 300:
                        raise TimeoutError("fieldrun did not become ready")
                    time.sleep(0.2)
            write_json(out / f"server-{stage}.json", {"command": command, "pid": process.pid,
                       "model_sha256": sha256(out / "model.json")})
            yield args.port
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def prediction_query(out, name, candidate, rows):
    verifier = out / "prediction-verifier.dl"
    verifier.write_text((DL / "guard_domain.dl").read_text() + "\n.output cdecide\n")
    return checked(out, name, verifier, candidate,
        {"tok": facts(rows,{})["tok"], "sample": "".join(f"{r['id']}\n" for r in rows)}, scalars={"ndomain": int})


def collect(out, args, stage):
    import numpy as np
    input_gate(out)
    if stage == "final":
        validate_freeze(out)
    rows = read(out / ("discovery.json" if stage == "discovery" else "dataset.json"))
    if stage != "discovery":
        rows = [r for r in rows if (r["part"] == "test") == (stage == "final")]
    prior_predictions = {}
    chosen = read(out / "features.json") if stage == "development" else []
    if stage == "discovery":
        guard = out / "discovery-copy.dl"
        guard.write_text(guard_program([(1,3,2)]))
        audit = prediction_query(out, "discovery-predictions", guard, rows)
        prior_predictions = {int(i): int(t) for i,t in audit["relations"]["cdecide"]}
    journal = out / f"{stage}-observations.jsonl"
    observations = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            record = json.loads(line)  # interrupted partial writes fail, never silently disappear
            if record["id"] in observations:
                raise ValueError("duplicate observation")
            observations[record["id"]] = record
    expected = {r["id"] for r in rows}
    if not set(observations) <= expected:
        raise ValueError("unexpected observation IDs")
    with server(out, args, stage) as port, journal.open("a") as stream:
        for j,row in enumerate(rows):
            i = row["id"]
            if i in observations:
                if observations[i]["input_sha256"] != digest(row["ctx"]):
                    raise ValueError("cached context changed")
                continue
            probe = {}
            if stage == "discovery":
                probe = {"layers": LAYERS, "readouts": [prior_predictions[i]]}
            elif stage == "development" and row["part"] == "train":
                probe = {"units": sorted({(f["layer"],f["neuron"]) for f in chosen})}
            start = time.monotonic()
            answer = post(port, row["ctx"], probe)
            native = answer.pop("observations")
            record = {"id": i, "input_sha256": digest(row["ctx"]), "seconds": time.monotonic()-start, **answer}
            if stage == "discovery":
                vectors = {f"{l}_{c}": np.asarray([o[c] for o in native if o["layer"] == l], dtype=np.float32)
                           for l in LAYERS for c in CHANNELS}
                tracefile = out / "discovery-traces" / f"{i}.npz"
                tracefile.parent.mkdir(exist_ok=True)
                np.savez_compressed(tracefile, **vectors)
                record.update(trace_file=str(tracefile.relative_to(out)), trace_sha256=sha256(tracefile),
                              copy=prior_predictions[i])
            else:
                record["observations"] = native
            stream.write(json.dumps(record)+"\n")
            stream.flush()
            observations[i] = record
            print(f"{stage}: {j+1}/{len(rows)}", flush=True)
    if set(observations) != expected:
        raise ValueError("incomplete observations")
    write_json(out / f"{stage}-references.json", {str(i): r["next"] for i,r in observations.items()})


def features(out):
    import numpy as np
    input_gate(out)
    rows = read(out / "discovery.json")
    records = {r["id"]: r for r in map(json.loads, (out / "discovery-observations.jsonl").read_text().splitlines())}
    if set(records) != {r["id"] for r in rows}:
        raise ValueError("discovery observations incomplete")
    refs = {i: r["next"] for i,r in records.items()}
    chosen = []
    for layer in LAYERS:
        for channel in CHANNELS:
            column = []
            width = None
            for row in rows:
                record = records[row["id"]]
                path = out / record["trace_file"]
                if sha256(path) != record["trace_sha256"]:
                    raise ValueError("native evidence changed")
                with np.load(path) as archive:
                    values = archive[f"{layer}_{channel}"]
                if width is not None and len(values) != width:
                    raise ValueError("neuron width changed")
                width = len(values)
                column.extend(f"{row['id']}\t{n}\t{float(v):.9g}\n" for n,v in enumerate(values))
            staged = facts(rows, refs)
            staged.update(sample="".join(f"{r['id']}\n" for r in rows),
                          unit="".join(f"{n}\n" for n in range(width)), observation="".join(column))
            result = checked(out, f"feature-{layer}-{channel}", DL / "native_trace_features.dl",
                             out / "discovery-copy.dl", staged)
            for neuron, score, threshold, polarity in result["relations"]["chosen"]:
                chosen.append({"id": len(chosen), "layer": layer, "channel": channel, "neuron": int(neuron),
                               "score": float(score), "threshold": float(threshold), "polarity": int(polarity), "width": width})
            print(f"feature: layer {layer}, {channel}: {result['relations']['chosen']}", flush=True)
    write_json(out / "features.json", chosen)


def select(out):
    input_gate(out)
    rows = [r for r in read(out / "dataset.json") if r["part"] != "test"]
    refs = {int(k): v for k,v in read(out / "development-references.json").items()}
    features_ = read(out / "features.json")
    records = {r["id"]: r for r in map(json.loads, (out / "development-observations.jsonl").read_text().splitlines())}
    staged = facts(rows, refs)
    staged["sample"] = "".join(f"{r['id']}\t{r['group']}\t{r['part']}\t{digest(r['ctx'])}\n" for r in rows)
    staged["pair"] = "".join(f"{r['base']}\t{r['id']}\n" for r in rows if "base" in r)
    staged["feature"] = "".join(f"{f['id']}\t{f['threshold']:.17g}\t{f['polarity']}\n" for f in features_)
    lines = []
    for row in rows:
        if row["part"] == "train":
            record = records[row["id"]]
            native = {(o["layer"],o["neuron"]): o for o in record["observations"]}
            lines.extend(f"{row['id']}\t{f['id']}\t{native[(f['layer'],f['neuron'])][f['channel']]:.17g}\n" for f in features_)
    staged["trace"] = "".join(lines)
    train_ids = [r["id"] for r in rows if r["part"] == "train"]
    shuffled = train_ids.copy()
    random.Random(104).shuffle(shuffled)
    staged["permutation"] = "".join(f"{i}\t{j}\n" for i,j in zip(train_ids,shuffled))
    result = checked(out, "selection", DL / "native_trace_select.dl", out / "hypotheses.dl", staged)
    for arm,name in ARMS.items():
        selected = [(int(rank),int(c)) for a,rank,c in result["relations"]["admitted"] if int(a)==arm]
        emit(out / "artifacts" / name, selected)
        for rank,c in selected:
            emit(out / "artifacts" / f"{name}-candidate-{c}", [(rank,c)])


def diagnostic(out, args):
    import numpy as np
    input_gate(out)
    rows = {r["id"]: r for r in read(out / "discovery.json")}
    records = {r["id"]: r for r in map(json.loads, (out / "discovery-observations.jsonl").read_text().splitlines())}
    features_ = read(out / "features.json")
    units = sorted({(f["layer"],f["neuron"],f["width"]) for f in features_})
    labels = read(out / "feature-7-gate.json")["relations"]["label"]
    values = {}
    for i, record in records.items():
        path = out / record["trace_file"]
        if sha256(path) != record["trace_sha256"]:
            raise ValueError("changed discovery traces")
        with np.load(path) as archive:
            for l,n,_ in units:
                values[(i,l,n)] = {c: float(archive[f"{l}_{c}"][n]) for c in CHANNELS}
    staged = {"label": "".join(f"{i}\t{c}\n" for i,c in labels),
              "chosen_unit": "".join(f"{l}\t{n}\t{w}\n" for l,n,w in units),
              "product": "".join(f"{i}\t{l}\t{n}\t{v['product']:.17g}\n" for (i,l,n),v in values.items())}
    plan = checked(out, "diagnostic-plan", DL / "native_trace_diagnostic_plan.dl", out / "hypotheses.dl", staged)
    journal = out / "diagnostic-observations.jsonl"
    observations = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            key = tuple(record[k] for k in ("id","layer","neuron","mode"))
            if key in observations:
                raise ValueError("duplicate diagnostic observation")
            observations[key] = record
    requests = plan["relations"]["plan"]
    with server(out, args, "diagnostic") as port, journal.open("a") as stream:
        for index,(i,l,n,mode,donor,value) in enumerate(requests):
            i,l,n,donor = map(int,(i,l,n,donor))
            key = (i,l,n,mode)
            if key in observations:
                continue
            original = records[i]
            readouts = [original["next"],original["copy"]]
            patch = {"layer":l,"neuron":n,"position":len(rows[i]["ctx"])-1,"value":float(value)}
            start = time.monotonic()
            result = post(port, rows[i]["ctx"], {"patches":[patch],"readouts":readouts})
            record = {"id":i,"layer":l,"neuron":n,"mode":mode,"donor":donor,"patch":patch,
                      "next":result["next"],"logits":result["logits"],"seconds":time.monotonic()-start}
            observations[key] = record
            stream.write(json.dumps(record)+"\n")
            stream.flush()
            print(f"diagnostic {index+1}/{len(requests)}", flush=True)
    raw = []
    for record in observations.values():
        original = records[record["id"]]
        after = dict(record["logits"])
        before = dict(original["logits"])
        raw.append("\t".join(map(str, (record["id"],record["layer"],record["neuron"],record["mode"],
            original["next"],record["next"],original["next_logit"],after[original["next"]],
            before[original["copy"]],after[original["copy"]])))+"\n")
    checked(out, "diagnostic-summary", DL / "native_trace_diagnostic.dl", out / "hypotheses.dl",
            {"result":"".join(raw), "observation":"".join(f"{i}\t{l}\t{n}\t{v['gate']:.17g}\t{v['value']:.17g}\t{v['product']:.17g}\n"
                for (i,l,n),v in values.items())})


def freeze(out):
    input_gate(out)
    if (out / "frozen.json").exists() or (out / "final-observations.jsonl").exists():
        raise ValueError("freeze already exists or final data collection began")
    if not read(out / "diagnostic-summary.json")["certified"]:
        raise ValueError("complete the diagnostic and identity checks before freeze")
    rows = [r for r in read(out / "dataset.json") if r["part"] == "test"]
    artifacts = {}
    for path in sorted((out / "artifacts").glob("*/circuits.dl")):
        name = path.parent.name
        result = prediction_query(out, f"domain-{name}", path, rows)
        artifacts[name] = {"path": str(path.relative_to(out)), "sha256": sha256(path),
            "domain": [int(i) for i, in result["relations"]["domain"]],
            "predictions": result["relations"]["cdecide"], "domain_evidence": result["evidence"],
            "bytes": path.stat().st_size,
            "rule_lines": sum(":-" in line for line in path.read_text().splitlines())}
    if not set(ARMS.values()) <= set(artifacts):
        raise ValueError("missing arm composite")
    # Hash every retained input, decision, native observation, and artifact present
    # at the boundary, excluding only server logs/launch records.
    locked = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()
              and not p.name.startswith("server-")}
    sources = [Path(__file__), ROOT / "py/native_trace.py", ROOT / "py/certificate.py",
               ROOT / "py/benchmark_guarded_copy.py", ROOT / "py/benchmark_induction.py",
               ROOT / "py/benchmark_copy_generalization.py", *sorted(DL.glob("native_trace*.dl")),
               DL / "equiv.dl", DL / "guard_domain.dl"]
    write_json(out / "frozen.json", {"artifacts": artifacts, "sha256": locked,
               "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
               "note": "No final oracle reference was requested before this manifest."})


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
    frozen = validate_freeze(out)
    rows = [r for r in read(out / "dataset.json") if r["part"] == "test"]
    refs = {int(k): v for k,v in read(out / "final-references.json").items()}
    certs, verdicts, predictions = {}, [], []
    for name, artifact in frozen["artifacts"].items():
        path = out / artifact["path"]
        certs[name] = {}
        for scope, ids in (("firing", artifact["domain"]), ("full", [r["id"] for r in rows])):
            subset = [r for r in rows if r["id"] in set(ids)]
            staged = facts(subset, refs)
            staged["domain"] = "".join(f"{i}\n" for i in ids)
            result = checked(out, f"equiv-{name}-{scope}", DL / "equiv.dl", path, staged,
                             scalars={k:int for k in ("ndomain","nmiss","nuncov","nmissing","ninvalid")},
                             allow_failure=True)
            certs[name][scope] = result
        firing = certs[name]["firing"]
        verdicts.append(f"{name}\t" + "\t".join(str(firing[k]) for k in
            ("ndomain","nmiss","nuncov","nmissing","ninvalid")) + f"\t{int(firing['certified'])}\n")
        predictions.extend(f"{name}\t{i}\t{t}\n" for i,t in artifact["predictions"])
    staged = {"artifact": "".join(f"{name}\n" for name in frozen["artifacts"]),
              "sample": "".join(f"{r['id']}\t{int(r['negative'])}\n" for r in rows),
              "ref": facts(rows,refs)["ref"], "prediction": "".join(predictions), "verdict": "".join(verdicts)}
    result = checked(out, "final-summary", DL / "native_trace_report.dl", out / "hypotheses.dl", staged)
    write_json(out / "report.json", {"tag": "empirical", "scope": "pinned f32 bundle, frozen finite domains",
               "frozen_sha256": sha256(out / "frozen.json"), "summary": result,
               "artifacts": frozen["artifacts"], "certificates": certs})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("discovery","features","development","select","diagnostic","freeze","final","evaluate"))
    p.add_argument("out", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--preflight", type=Path, default=ROOT / "reference/benchmarks/native_trace_preflight")
    p.add_argument("--port", type=int, default=8391)
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.phase in ("discovery","development","final","diagnostic"):
        if not (a.binary and a.bundle):
            p.error("oracle stages need --binary and --bundle")
        if a.phase == "diagnostic":
            diagnostic(a.out, a)
        else:
            collect(a.out, a, a.phase)
    else:
        {"features":features, "select":select, "freeze":freeze, "evaluate":evaluate}[a.phase](a.out)


if __name__ == "__main__":
    main()
