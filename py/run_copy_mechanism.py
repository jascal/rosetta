"""Layer-0 donor discovery I/O. All planning, integrity and causal verdicts are Datalog."""
import argparse
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

from certificate import check, sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reference/benchmarks/qwen25_native_trace_c"
OUT = ROOT / "reference/benchmarks/qwen25_copy_mechanism"
BASELINE = ROOT / "reference/benchmarks/qwen25_native_trace_b/artifacts/behavior/circuits.dl"


def read(p):
    return json.loads(p.read_text())


def write(p, data):
    p.write_text(json.dumps(data, indent=2) + "\n")


def inputs(out):
    manifest = read(out / "discovery-inputs.json")
    for name, digest in manifest["sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"discovery input changed: {name}")
    rows = read(SOURCE / "dataset.json")
    refs = {int(i): t for i, t in read(SOURCE / "references.json").items()}
    plan = read(out / "donor-plan.json")
    evidence = Path(plan["evidence"])
    meta = read(evidence / "certificate.json")
    for name, digest in meta["sha256"].items():
        if sha256(evidence / name) != digest:
            raise ValueError(f"donor-plan evidence changed: {name}")
    # Read the authoritative query outputs, not a separately editable proposal.
    donors = {int(i): int(d) for i, d in
              (line.split("\t") for line in (evidence / "outputs/donor.csv").read_text().splitlines())}
    arms = [(int(i), k, int(d)) for i, k, d in
            (line.split("\t") for line in (evidence / "outputs/arm.csv").read_text().splitlines())]
    return rows, refs, donors, arms


def query(out, name, module, facts):
    source = ROOT / "dl" / module
    result = check(source, source, facts, {}, evidence_dir=out / "certificate-evidence" / name,
                   provenance={"source": "layer-0 discovery; instrument integrity, not model equivalence"})
    write(out / f"{name}.json", result)
    if not result["certified"]:
        raise ValueError(f"{name} failed: {result.get('error', result.get('relations'))}")
    return result


def post(port, ids, probe):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/probe/mlp",
        json.dumps({"ids": ids, "probe": probe}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    if "error" in result:
        raise ValueError(result)
    return result


@contextmanager
def server(a, stage):
    command = [str(a.binary.resolve()), "--bundle", str(a.bundle.resolve()), "--serve", str(a.port)]
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", a.port)) == 0:
            raise ValueError("port occupied; refusing to use an unattested process")
    with (a.out / f"server-{stage}.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        write(a.out / f"server-{stage}.json", {"command": command, "pid": process.pid,
              "binary_sha256": sha256(a.binary), "bundle_sha256": {
              s: sha256(str(a.bundle) + s) for s in (".fieldrun.bin", ".fieldrun.json")}})
        try:
            start = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as r:
                        if json.load(r).get("ok"):
                            break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() - start > 120:
                        raise TimeoutError("server startup")
                    time.sleep(.2)
            yield a.port
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def observation(out, i):
    with gzip.open(out / "observations" / f"{i}.json.gz", "rt") as stream:
        return json.load(stream)


def collect(a):
    rows, refs, donors, _ = inputs(a.out)
    if (a.out / "frozen-discovery.json").exists():
        raise ValueError("discovery already frozen")
    dest = a.out / "observations"
    dest.mkdir(exist_ok=True)
    with server(a, "collect") as port:
        for j, row in enumerate(rows):
            path = dest / f"{row['id']}.json.gz"
            if path.exists():
                continue
            i = row["id"]
            targets = sorted({refs[i], refs[donors.get(i, i)]})
            result = post(port, row["ctx"], {"layers": [0], "readouts": targets})
            temp = path.with_suffix(".partial")
            with gzip.open(temp, "wt") as stream:
                json.dump(result, stream)
            os.replace(temp, path)
            if (j + 1) % 10 == 0:
                print(f"observe {j+1}/{len(rows)}", flush=True)
    historical = read(SOURCE / "base.json")
    facts = {
        "row": "".join(f"{r['id']}\t{len(r['ctx'])-1}\n" for r in rows),
        "ref": "".join(f"{i}\t{t}\t{historical[str(i)]['ref_logit']:.17g}\n" for i, t in refs.items()),
        "answer": "", "observed": ""}
    for r in rows:
        i = r["id"]
        result = observation(a.out, i)
        facts["answer"] += f"{i}\t{result['next']}\t{dict(result['logits'])[refs[i]]:.17g}\n"
        facts["observed"] += "".join(f"{i}\t{o['layer']}\t{o['position']}\t{o['neuron']}\n"
                                       for o in result["observations"])
    result = query(a.out, "preflight", "copy_mechanism_preflight.dl", facts)
    print("preflight", result["certified"], flush=True)


def freeze(a):
    inputs(a.out)
    if (a.out / "frozen-discovery.json").exists() or (a.out / "interventions.jsonl").exists():
        raise ValueError("already frozen or intervened")
    preflight = read(a.out / "preflight.json")
    if not preflight["certified"]:
        raise ValueError("preflight did not pass")
    # The complete archive, including preflight query inputs, is locked.
    paths = [p for p in a.out.rglob("*") if p.is_file() and not p.name.startswith("server-")]
    sources = [Path(__file__), ROOT / "py/certificate.py", ROOT / "docs/copy-mechanism-plan.md",
               ROOT / "dl/copy_mechanism_donors.dl", ROOT / "dl/copy_mechanism_preflight.dl",
               ROOT / "dl/copy_mechanism_summary.dl", BASELINE,
               SOURCE / "dataset.json", SOURCE / "references.json", SOURCE / "base.json",
               ROOT / "reference/benchmarks/qwen25_mlp_ablation/interventions.jsonl"]
    write(a.out / "frozen-discovery.json", {
        "scope": "306 layer-0 discovery interventions; no final-test circuit or references",
        "sha256": {str(p.relative_to(a.out)): sha256(p) for p in paths},
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "model": read(a.out / "server-collect.json")})
    print("discovery frozen", flush=True)


def validate(a):
    locked = read(a.out / "frozen-discovery.json")
    for name, digest in locked["sha256"].items():
        if sha256(a.out / name) != digest:
            raise ValueError(f"frozen observation changed: {name}")
    for name, digest in locked["sources"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"frozen source changed: {name}")
    if a.binary and sha256(a.binary) != locked["model"]["binary_sha256"]:
        raise ValueError("different executable")
    if a.bundle:
        for suffix, digest in locked["model"]["bundle_sha256"].items():
            if sha256(str(a.bundle) + suffix) != digest:
                raise ValueError("different bundle")


def intervene(a):
    validate(a)
    rows, refs, donors, arms = inputs(a.out)
    by_id = {r["id"]: r for r in rows}
    obs = {i: observation(a.out, i) for i in by_id}
    # All observations are last-position, layer-0; preflight checked this.
    values = {i: {o["neuron"]: o["product"] for o in x["observations"]} for i, x in obs.items()}
    journal = a.out / "interventions.jsonl"
    cached = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            key = (r["i"], r["kind"])
            if key in cached:
                raise ValueError("duplicate intervention")
            cached[key] = r
    if set(cached) - {(i, k) for i, k, _ in arms}:
        raise ValueError("unplanned intervention")
    with server(a, "intervene") as port, journal.open("a") as stream:
        for j, (i, kind, d) in enumerate(arms):
            if (i, kind) in cached:
                continue
            patches = [{"layer": 0, "position": len(by_id[i]["ctx"])-1, "neuron": n,
                        "value": 0.0 if kind == "zero" else values[d][n]} for n in range(4864)]
            start = time.monotonic()
            r = post(port, by_id[i]["ctx"], {"patches": patches, "readouts": sorted({refs[i], refs[donors[i]]})})
            record = {"i": i, "kind": kind, "donor": d, "response": r,
                      "seconds": time.monotonic()-start}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if (j + 1) % 10 == 0:
                print(f"intervene {j+1}/{len(arms)}", flush=True)


def evaluate(a):
    validate(a)
    rows, refs, donors, arms = inputs(a.out)
    records = [json.loads(line) for line in (a.out / "interventions.jsonl").read_text().splitlines()]
    if len(records) != len({(r['i'], r['kind']) for r in records}):
        raise ValueError("duplicate intervention observations")
    facts = {"row": "".join(f"{r['id']}\t{int(r['negative'])}\n" for r in rows),
             "ref": "".join(f"{i}\t{t}\n" for i, t in refs.items()),
             "donor": "".join(f"{i}\t{d}\n" for i, d in donors.items()),
             "planned": "".join(f"{i}\t{k}\t{d}\n" for i, k, d in arms),
             "base": "", "result": "", "old_zero": "", "product": ""}
    for row in rows:
        i = row["id"]
        facts["product"] += "".join(f"{i}\t{o['neuron']}\t{o['product']:.17g}\n"
                                      for o in observation(a.out, i)["observations"])
    for i, d in donors.items():
        lg = dict(observation(a.out, i)["logits"])
        facts["base"] += f"{i}\t{lg[refs[i]]:.17g}\t{lg[refs[d]]:.17g}\n"
    for r in records:
        i, x = r["i"], r["response"]
        lg = dict(x["logits"])
        facts["result"] += f"{i}\t{r['kind']}\t{x['next']}\t{lg[refs[i]]:.17g}\t{lg[refs[donors[i]]]:.17g}\n"
    old = ROOT / "reference/benchmarks/qwen25_mlp_ablation/interventions.jsonl"
    for line in old.read_text().splitlines():
        r = json.loads(line)
        if r["arm"] == "sweep" and r["key"] == 0:
            facts["old_zero"] += f"{r['i']}\t{r['answer']}\t{r['ref_logit']:.17g}\n"
    result = query(a.out, "summary", "copy_mechanism_summary.dl", facts)
    print(json.dumps(result["relations"]["counts"], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("collect", "freeze", "intervene", "evaluate"))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--binary", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--port", type=int, default=8396)
    a = p.parse_args()
    if a.phase in ("collect", "intervene") and not (a.binary and a.bundle):
        p.error("model phases require --binary and --bundle")
    globals()[a.phase](a)


if __name__ == "__main__":
    main()
