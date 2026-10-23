"""Run-D driver: whole-MLP ablation sweep to locate the copy computation.

The treatment is the whole layer grid, fixed in advance: every layer of the
model is ablated, so no layer can be picked after the fact. The contexts, the
unpatched references and the baseline logits are run C's frozen evidence,
digest-pinned, so run D is directly comparable to run C.

Every choice -- which rows, which arms, which layers, which positions, which
value -- is a Datalog relation (dl/mlp_ablation_plan.dl). The expansion of a
registered rule into per-neuron patch rows is mechanical serialization and
happens here; the readouts and the verdicts are Datalog again
(dl/mlp_ablation_summary.dl).
"""
import argparse
import json
from pathlib import Path
import time

from benchmark_induction import DL, digest
from certificate import check, sha256
from native_trace import write_json

ROOT = Path(__file__).resolve().parents[1]

# Run C's frozen evidence, digest-pinned. Run D reuses it unchanged.
RUN_C = ROOT / "reference/benchmarks/qwen25_native_trace_c"
RUN_C_DATASET_SHA256 = "1b54ab66f056295a87a8cd4e11a0c81ba47206043ac3db5696ac33ed675f6b44"

# The registered arms.
LAYERS = tuple(range(24))          # every layer of the model
CUMUL = (1, 2, 4, 8, 12, 16, 24)   # cumulative prefixes, ending at the joint arm
WRITEBACK = (7, 15, 23)            # the run-B treated layers
STAGE2 = ()                        # filled in by the registered rule after stage 1

# The model's MLP width, from the bundle's own metadata (config[5]).
WIDTH = 4864


def read(path):
    return json.loads(Path(path).read_text())


def write(path, text):
    Path(path).write_text(text)


def inputs(a):
    """Copy run C's frozen contexts and baselines; stage the arm definitions."""
    if sha256(RUN_C / "dataset.json") != RUN_C_DATASET_SHA256:
        raise ValueError("run C's dataset.json changed; run D's inputs are not what was preregistered")
    out = a.out
    if (out / "frozen.json").exists():
        raise ValueError("study is frozen")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.json", "references.json", "base.json"):
        write(out / name, (RUN_C / name).read_text())
    donor = read(RUN_C / "plan.json")["donor"]
    write_json(out / "donor.json", donor)
    protocol = {
        "seed": 301,
        "source": "run C's frozen contexts, references and baselines (digest-pinned)",
        "source_dataset_sha256": RUN_C_DATASET_SHA256,
        "prereg_sha256": sha256(ROOT / "docs/native-trace-mlp-prereg.md"),
        "layers": list(LAYERS),
        "cumulative": list(CUMUL),
        "writeback_layers": list(WRITEBACK),
        "width": WIDTH,
        "donor": donor,
        "counts": {"n": len(read(out / "dataset.json"))},
        "note": "Reused from run C so run D is directly comparable to it; nothing is fitted here.",
    }
    write_json(out / "protocol.json", protocol)
    print(json.dumps({k: protocol[k] for k in ("counts", "prereg_sha256", "width")}, indent=2))
    return 0


def input_gate(out):
    protocol = read(out / "protocol.json")
    if protocol["prereg_sha256"] != sha256(ROOT / "docs/native-trace-mlp-prereg.md"):
        raise ValueError("the preregistration changed")
    if protocol["source_dataset_sha256"] != sha256(RUN_C / "dataset.json"):
        raise ValueError("run C's dataset changed under run D")
    for key in ("layers", "cumulative", "writeback_layers", "width"):
        if key not in protocol:
            raise ValueError(f"protocol.json is missing {key}")


def before_freeze(out):
    if (out / "frozen.json").exists():
        raise ValueError("study is frozen")
    if (out / "interventions.jsonl").exists():
        raise ValueError("interventions already exist")


def checked(out, name, verifier, candidate, facts, scalars=None, *, allow_failure=False):
    start = time.monotonic()
    result = check(verifier, candidate, facts, scalars or {},
                   evidence_dir=out / "certificate-evidence" / name,
                   provenance={"stage": name, "source": "native-trace run D"})
    result["evidence"] = str(Path(result["evidence"]).relative_to(out))
    result["wall_seconds"] = time.monotonic() - start
    write_json(out / f"{name}.json", result)
    if result.get("error") or (not result["certified"] and not allow_failure):
        raise ValueError(f"{name} query failed: {result.get('error', 'invalid inputs')}")
    return result


def arm_facts(out, stage2_layers=()):
    """The registered arm definitions, staged for the plan query."""
    protocol = read(out / "protocol.json")
    layers = list(protocol["layers"])
    cumul = list(protocol["cumulative"])
    return {
        "row": "".join(f"{r['id']}\t{int(r['negative'])}\n" for r in read(out / "dataset.json")),
        "layer": "".join(f"{l}\t{protocol['width']}\n" for l in layers),
        "neuron": "".join(f"{l}\t{n}\n" for l in layers for n in range(protocol["width"])),
        "sweep_arm": "".join(f"{l}\n" for l in layers),
        "cumul_arm": "".join(f"{k}\n" for k in cumul),
        "cumul_member": "".join(f"{k}\t{l}\n" for k in cumul for l in range(k)),
        "ident_arm": "".join(f"{l}\n" for l in protocol["writeback_layers"]),
        "donor_arm": "".join(f"{l}\n" for l in protocol["writeback_layers"]),
        "full_arm": "".join(f"{l}\n" for l in stage2_layers),
    }


def collect(a):
    """One unpatched observation per row: the MLP products of the write-back layers."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    from run_native_trace2 import post, server

    out = a.out
    input_gate(out)
    before_freeze(out)
    protocol = read(out / "protocol.json")
    writeback = list(protocol["writeback_layers"])
    width = int(protocol["width"])
    rows = read(out / "dataset.json")
    journal = out / "collection.jsonl"
    records = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            if record["id"] in records:
                raise ValueError("duplicate collection record")
            records[record["id"]] = record
    expected = {r["id"] for r in rows}
    if not set(records) <= expected:
        raise ValueError("unexpected collection ids")
    with server(out, a, "collect") as port, journal.open("a") as stream:
        for j, row in enumerate(rows):
            i = row["id"]
            if i in records:
                if records[i]["input_sha256"] != digest(row["ctx"]):
                    raise ValueError("cached context changed")
                continue
            start = time.monotonic()
            answer = post(port, row["ctx"], {"layers": list(writeback),
                                              "readouts": [row["expected"]]})
            native = answer.pop("observations")
            last = len(row["ctx"]) - 1
            record = {"id": i, "input_sha256": digest(row["ctx"]),
                      "next": answer["next"], "next_logit": answer["next_logit"],
                      "logits": answer["logits"], "seconds": time.monotonic() - start}
            vectors = {}
            for layer in writeback:
                vector = [o["product"] for o in native
                          if o["layer"] == layer and o["position"] == last]
                if len(vector) != width:
                    raise ValueError(f"row {i} layer {layer}: {len(vector)} products, expected {width}")
                vectors[str(layer)] = np.array(vector, dtype=np.float32)
            tracefile = out / "mlp-traces" / f"{i}.npz"
            tracefile.parent.mkdir(exist_ok=True)
            np.savez_compressed(tracefile, **vectors)
            record["trace_file"] = str(tracefile.relative_to(out))
            record["trace_sha256"] = sha256(tracefile)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records[i] = record
            print(f"collect: {j+1}/{len(rows)}", flush=True)
    if set(records) != expected:
        raise ValueError("incomplete collection")
    total = sum((out / "mlp-traces" / f"{i}.npz").stat().st_size for i in records)
    print(f"collect: {len(records)} rows, {total/1e6:.1f} MB of traces")


def plan(a):
    """Ask Datalog which arms, layers and positions to patch."""
    out = a.out
    input_gate(out)
    before_freeze(out)
    stage2 = tuple(int(x) for x in a.stage2.split(",")) if a.stage2 else ()
    result = checked(out, "plan", DL / "mlp_ablation_plan.dl",
                     DL / "mlp_ablation_plan.dl", arm_facts(out, stage2))
    relations = result["relations"]
    layers = {}
    for (i, arm, key, layer, mode, allpos) in relations["plan"]:
        entry = layers.setdefault((int(i), arm, int(key)),
                                  {"layers": [], "mode": mode, "allpos": allpos})
        entry["layers"].append(int(layer))
    requests = []
    for (i, arm, key) in relations["req"]:
        entry = layers[(int(i), arm, int(key))]
        requests.append({"i": int(i), "arm": arm, "key": int(key),
                         "layers": sorted(entry["layers"]), "mode": entry["mode"],
                         "allpos": bool(entry["allpos"])})
    requests.sort(key=lambda r: (r["i"], r["arm"], r["key"]))
    write_json(out / "plan.json", {
        "requests": requests,
        "nrequests": int(relations["nreq"][0][0]),
        "narms": int(relations["narms"][0][0]),
        "stage2_layers": list(stage2),
        "width": WIDTH,
    })
    print(f"plan: {relations['nreq'][0][0]} requests over {relations['narms'][0][0]} layers")


def freeze(a):
    out = a.out
    input_gate(out)
    if (out / "frozen.json").exists() or (out / "interventions.jsonl").exists():
        raise ValueError("freeze already exists or interventions began")
    if not (out / "plan.json").exists():
        raise ValueError("no plan to freeze")
    locked = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
              if p.is_file() and not p.name.startswith("server-")}
    sources = [Path(__file__), ROOT / "py/certificate.py", ROOT / "py/native_trace.py",
               ROOT / "py/benchmark_induction.py", ROOT / "py/run_native_trace2.py",
               ROOT / "docs/native-trace-mlp-prereg.md",
               DL / "mlp_ablation_plan.dl", DL / "mlp_ablation_summary.dl",
               RUN_C / "dataset.json", RUN_C / "references.json", RUN_C / "base.json"]
    write_json(out / "frozen.json", {
        "status": "ready",
        "sha256": locked,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "note": "The ablation plan was frozen before the first intervention request."})
    print(f"frozen: {len(locked)} files, {len(sources)} sources")


def validate_freeze(out):
    frozen = read(out / "frozen.json")
    for name, expected in frozen["sha256"].items():
        if sha256(out / name) != expected:
            raise ValueError(f"frozen file changed: {name}")
    for name, expected in frozen["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    return frozen


def build_patches(request, row, observations, donor_values, width):
    """Expand a registered arm into per-neuron patch rows.

    Mechanical serialization: the arm, the layers, the positions and the value
    were decided by the plan query; this only expands them over the neuron
    universe. Fail closed if the expansion does not cover exactly the neurons
    the layer has.
    """
    position = len(row["ctx"]) - 1
    positions = range(len(row["ctx"])) if request["allpos"] else [position]
    patches = []
    for layer in request["layers"]:
        for pos in positions:
            if request["mode"] == "zero":
                values = [(n, 0.0) for n in range(width)]
            elif request["mode"] == "own":
                values = list(enumerate(observations[layer][row["id"]]))
            else:  # donor
                values = list(enumerate(donor_values[row["id"]][layer]))
            if len(values) != width:
                raise ValueError(f"row {row['id']} layer {layer}: {len(values)} values, expected {width}")
            for neuron, value in values:
                patches.append({"layer": layer, "position": pos, "neuron": neuron,
                                "value": float(value)})
    expected = len(request["layers"]) * len(positions) * width
    if len(patches) != expected:
        raise ValueError(f"patch expansion produced {len(patches)} rows, expected {expected}")
    return patches


def intervene(a):
    """Execute the frozen plan, one request per planned intervention."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    from run_native_trace2 import post, server

    out = a.out
    input_gate(out)
    validate_freeze(out)
    rows = {r["id"]: r for r in read(out / "dataset.json")}
    plan_doc = read(out / "plan.json")
    references = {int(k): v for k, v in read(out / "references.json").items()}
    donor = {int(k): v for k, v in read(out / "donor.json").items()}
    protocol = read(out / "protocol.json")
    writeback = list(protocol["writeback_layers"])
    width = int(protocol["width"])
    traces = {i: dict(np.load(out / "mlp-traces" / f"{i}.npz")) for i in rows}
    for i, archive in traces.items():
        for layer in writeback:
            if str(layer) not in archive or len(archive[str(layer)]) != width:
                raise ValueError(f"row {i} layer {layer}: observation file is incomplete")
    own = {l: {i: traces[i][str(l)] for i in rows} for l in writeback}
    donor_values = {i: {l: own[l][donor[i]] for l in writeback} for i in rows}
    journal = out / "interventions.jsonl"
    cached = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            key = (record["i"], record["arm"], record["key"])
            if key in cached:
                raise ValueError("duplicate intervention")
            cached[key] = record
    planned = {(r["i"], r["arm"], r["key"]) for r in plan_doc["requests"]}
    if not set(cached) <= planned:
        raise ValueError("unexpected intervention")
    with server(out, a, "intervene") as port, journal.open("a") as stream:
        for j, request in enumerate(plan_doc["requests"]):
            i, arm, key = request["i"], request["arm"], request["key"]
            ident = (i, arm, key)
            if ident in cached:
                continue
            row = rows[i]
            patches = build_patches(request, row, own, donor_values, width)
            start = time.monotonic()
            answer = post(port, row["ctx"], {"patches": patches,
                                              "readouts": [references[i], row["expected"]]})
            logits = dict(answer["logits"])
            record = {"i": i, "arm": arm, "key": key, "answer": answer["next"],
                      "answer_logit": answer["next_logit"],
                      "ref_logit": logits[references[i]],
                      "target_logit": logits[row["expected"]],
                      "npatches": len(patches), "nlayers": len(request["layers"]),
                      "seconds": time.monotonic() - start}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            cached[ident] = record
            if (j + 1) % 250 == 0:
                print(f"intervene: {j+1}/{len(plan_doc['requests'])}", flush=True)
    if set(cached) != planned:
        raise ValueError("incomplete interventions")
    print(f"intervene: {len(cached)} requests")


def evaluate(a):
    """Ask Datalog what the ablations did, and which hypothesis they support."""
    out = a.out
    validate_freeze(out)
    rows = read(out / "dataset.json")
    plan_doc = read(out / "plan.json")
    references = {int(k): v for k, v in read(out / "references.json").items()}
    base = {int(k): v for k, v in read(out / "base.json").items()}
    donor = {int(k): v for k, v in read(out / "donor.json").items()}
    records = [json.loads(line) for line in (out / "interventions.jsonl").read_text().splitlines()]
    req_layer = {}
    for request in plan_doc["requests"]:
        req_layer[(request["i"], request["arm"], request["key"])] = request["layers"]
    facts = {
        "result": "".join(f"{r['i']}\t{r['arm']}\t{r['key']}\t{r['answer']}\t{r['answer_logit']:.17g}"
                          f"\t{r['ref_logit']:.17g}\t{r['target_logit']:.17g}\n" for r in records),
        "req": "".join(f"{r['i']}\t{r['arm']}\t{r['key']}\n" for r in plan_doc["requests"]),
        "req_layer": "".join(f"{i}\t{a}\t{k}\t{l}\n" for (i, a, k), ls in req_layer.items() for l in ls),
        "ref": "".join(f"{i}\t{t}\n" for i, t in references.items()),
        "row": "".join(f"{r['id']}\t{int(r['negative'])}\t{r['expected']}\n" for r in rows),
        "base": "".join(f"{i}\t{b['ref_logit']:.17g}\t{b['target_logit']:.17g}\n" for i, b in base.items()),
        "donor": "".join(f"{d}\t{i}\n" for i, d in donor.items()),
        "cumul_arm": "".join(f"{k}\n" for k in CUMUL),
    }
    summary = checked(out, "summary", DL / "mlp_ablation_summary.dl",
                      DL / "mlp_ablation_summary.dl", facts)
    relations = summary["relations"]
    cost = sum(r.get("seconds", 0.0) for r in records)
    write_json(out / "report.json", {
        "tag": "empirical",
        "scope": "pinned f32 bundle, one checkpoint, run C's 114 contexts, MLP-product interventions",
        "summary": summary,
        "verdict": {
            "h_b_layer_localized": bool(relations["h_b_supported"]),
            "h_b_aggregate": bool(relations["h_b_aggregate"]),
            "h_c_position_localized": bool(relations["h_c_supported"]),
            "h_a_not_the_mlp_path": bool(relations["h_a_supported"]),
            "instrument_ok": bool(relations["instrument_ok"]),
            "consistency_ok": bool(relations["consistency_ok"]),
        },
        "cost": {"interventions": len(records), "seconds": cost},
    })
    print(json.dumps({"verdict": {k: bool(v) for k, v in relations.items()
                                  if k in ("h_b_supported", "h_b_aggregate", "h_c_supported",
                                           "h_a_supported", "instrument_ok", "consistency_ok")}},
                     indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("inputs", "collect", "plan", "freeze", "intervene", "evaluate"))
    p.add_argument("out", type=Path)
    p.add_argument("--stage2", default="", help="comma-separated layers for the stage-2 arm")
    p.add_argument("--binary", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--preflight", type=Path,
                   default=ROOT / "reference/benchmarks/native_trace_preflight")
    p.add_argument("--port", type=int, default=8394)
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.phase == "inputs":
        return inputs(a)
    if a.phase in ("collect", "intervene"):
        if not (a.binary and a.bundle):
            p.error("model stages need --binary and --bundle")
    {"collect": collect, "plan": plan, "freeze": freeze,
     "intervene": intervene, "evaluate": evaluate}[a.phase](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
