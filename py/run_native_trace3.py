"""Run-C driver: is run B's trace signature the copy mechanism?

The treatment is fixed in advance -- the nine (layer, neuron) coordinates of run
B's frozen features.json -- and nothing is fitted, selected or re-ranked here.
This driver generates the fresh inputs, observes the build-time model once
(unpatched), asks Datalog which interventions to run, freezes that plan, executes
it, and asks Datalog what the effects were.

Every decision -- which patches to apply, which donor writes where, whether an
arm flipped, and which hypothesis the evidence supports -- is a Datalog relation.
Python stages facts, observes the model, and serializes the relations.
"""
import argparse
import json
from pathlib import Path
import random
import time

from benchmark_copy_generalization import generate
from benchmark_induction import DL, digest
from certificate import check, sha256
from native_trace import write_json

ROOT = Path(__file__).resolve().parents[1]

# Fresh inputs. One evaluation split is enough: run C fits nothing.
SEED = 301
LENGTHS = (8, 12, 24)
GROUPS = 2
KS = tuple(range(1, 10))
RANDOM_SEED = 205

# The treatment, in the registered layer-stratified order of
# docs/native-trace-runc-prereg.md: one neuron per layer first, so Z_3 is one
# neuron per layer and the dose-response is not confounded with depth.
REGISTER = ((7, "gate", 1898), (15, "gate", 788), (23, "gate", 412),
            (7, "value", 3), (15, "value", 680), (23, "value", 1650),
            (7, "product", 1835), (15, "product", 992), (23, "product", 1792))

# Digest-pinned to run B's frozen features.json. The driver refuses to build a
# plan if that file has changed.
RUN_B_FEATURES = ROOT / "reference/benchmarks/qwen25_native_trace_b/features.json"
RUN_B_FEATURES_SHA256 = "fe06adfa53badd035a84c735f9c98af19ca014aee63619037a8231172d6aea57"

# Every token in these datasets is excluded from run C's inputs.
HISTORICAL = (
    "reference/benchmarks/qwen25_05b/dataset.json",
    "reference/benchmarks/qwen25_05b_copy_generalization/dataset.json",
    "reference/benchmarks/qwen25_05b_guarded_seed1/dataset.json",
    "reference/benchmarks/qwen25_05b_int8_withheld_pairs/dataset.json",
    "reference/benchmarks/qwen25_05b_target_diagnostic/dataset.json",
    "reference/benchmarks/qwen25_05b_withheld_pairs/dataset.json",
)
RUN_A = ("reference/benchmarks/qwen25_native_trace/dataset.json",
         "reference/benchmarks/qwen25_native_trace/discovery.json")
RUN_B = ("reference/benchmarks/qwen25_native_trace_b/dataset.json",)


def read(path):
    return json.loads(Path(path).read_text())


def rows_of(path):
    data = read(path)
    return data["rows"] if isinstance(data, dict) and "rows" in data else data


def excluded_vocabulary():
    vocab = set()
    for path in HISTORICAL + RUN_A + RUN_B:
        for row in rows_of(ROOT / path):
            vocab.update(row.get("ctx", ()))
    return vocab


def gkey(group):
    """seed-length-index, with the class segment removed.

    A label transformation, not a decision: it is what makes a copy row and a
    control row 'the same generated group' without the host choosing a donor.
    """
    parts = group.split("-")
    return f"{parts[0]}-{parts[1]}-{parts[3]}"


def treatment():
    """The nine coordinates, verified against run B's frozen evidence."""
    if sha256(RUN_B_FEATURES) != RUN_B_FEATURES_SHA256:
        raise ValueError("run B's features.json changed; the treatment is not what was preregistered")
    features = read(RUN_B_FEATURES)
    pairs = {(f["layer"], f["neuron"]) for f in features}
    registered = {(layer, neuron) for layer, _, neuron in REGISTER}
    if pairs != registered:
        raise ValueError(f"run B's coordinates {sorted(pairs)} differ from the registered set")
    widths = {}
    for f in features:
        widths[f["layer"]] = f["width"]
    return [{"rank": rank, "layer": layer, "neuron": neuron, "channel": channel}
            for rank, (layer, channel, neuron) in enumerate(REGISTER, start=1)], widths


def random_sets(widths):
    """The registered layer-matched random control sets.

    Nested, disjoint from the treated coordinates, and with exactly the same
    per-layer counts as Z_k. Datalog re-checks all three properties; this is
    the registered draw (seed 205), not a decision.
    """
    treated = {(f["layer"], f["neuron"]) for f in treatment()[0]}
    rng = random.Random(RANDOM_SEED)
    pools = {}
    for layer, width in sorted(widths.items()):
        candidates = [n for n in range(width) if (layer, n) not in treated]
        rng.shuffle(candidates)
        pools[layer] = candidates
    counts = {}
    for k in KS:
        per_layer = {layer: 0 for layer in widths}
        for layer, _, _ in REGISTER[:k]:
            per_layer[layer] += 1
        counts[k] = per_layer
    sets = []
    for k in KS:
        for layer in sorted(widths):
            for neuron in pools[layer][:counts[k][layer]]:
                sets.append((k, layer, neuron))
    return sets


def build():
    exclude = excluded_vocabulary()
    raw = generate(seeds=(SEED,), lengths=LENGTHS, groups=GROUPS, exclude=exclude)
    rows = []
    for row in raw:
        rows.append(dict(row, part="eval", negative=row["part"] == "off_domain",
                         gkey=gkey(row["group"])))
    rows.sort(key=lambda r: r["id"])
    for i, row in enumerate(rows):
        row["id"] = i
    return rows, exclude


def check_disjoint(rows, exclude):
    problems = []
    counts = {"n": len(rows), "negative": sum(r["negative"] for r in rows)}
    if counts != {"n": 114, "negative": 18}:
        problems.append(f"split is {counts}, expected 114 rows with 18 negatives")
    vocab = set()
    for row in rows:
        vocab.update(row["ctx"])
    clash = vocab & exclude
    if clash:
        problems.append(f"reuses excluded tokens: {sorted(clash)[:8]}")
    groups = {r["group"] for r in rows}
    if len(groups) != 12:
        problems.append(f"{len(groups)} groups, expected 12")
    for row in rows:
        if row["negative"] and row["kind"] == "no_repeat" and len(row["ctx"]) < 8:
            problems.append(f"control row {row['id']} is degenerate")
    return problems, counts


def input_gate(out):
    protocol = read(out / "protocol.json")
    if protocol.get("problems"):
        raise ValueError("input generation reported problems")
    if protocol["dataset_sha256"] != sha256(out / "dataset.json"):
        raise ValueError("dataset.json does not match protocol.json")
    if protocol["prereg_sha256"] != sha256(ROOT / "docs/native-trace-runc-prereg.md"):
        raise ValueError("the preregistration changed")
    if protocol["treatment_sha256"] != RUN_B_FEATURES_SHA256:
        raise ValueError("the treatment digest in protocol.json does not match run B")


def before_freeze(out):
    if (out / "frozen.json").exists():
        raise ValueError("study is frozen")
    if (out / "interventions.jsonl").exists():
        raise ValueError("interventions already exist")


def checked(out, name, verifier, candidate, facts, scalars=None, *, allow_failure=False):
    start = time.monotonic()
    result = check(verifier, candidate, facts, scalars or {},
                   evidence_dir=out / "certificate-evidence" / name,
                   provenance={"stage": name, "source": "native-trace run C"})
    result["evidence"] = str(Path(result["evidence"]).relative_to(out))
    result["wall_seconds"] = time.monotonic() - start
    write_json(out / f"{name}.json", result)
    if result.get("error") or (not result["certified"] and not allow_failure):
        raise ValueError(f"{name} query failed: {result.get('error', 'invalid inputs')}")
    return result


# --------------------------------------------------------------------- phases
def inputs(a):
    rows, exclude = build()
    problems, counts = check_disjoint(rows, exclude)
    report = {"seed": SEED, "lengths": list(LENGTHS), "groups": GROUPS,
              "excluded_tokens": len(exclude),
              "excluded_from": list(HISTORICAL + RUN_A + RUN_B),
              "counts": counts, "problems": problems,
              "note": "Fresh inputs, generated before any run-C model request."}
    if not a.write:
        print(json.dumps(report, indent=2))
        return 1 if problems else 0
    if problems:
        for line in problems:
            print(f"refusing to write: {line}")
        return 1
    before_freeze(a.out)
    if list(a.out.glob("*-observations.jsonl")) or (a.out / "collection.jsonl").exists():
        raise ValueError("cannot reseal inputs after collection")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "dataset.json").write_text(json.dumps(rows, indent=1) + "\n")
    protocol = dict(report)
    protocol["dataset_sha256"] = sha256(a.out / "dataset.json")
    protocol["prereg_sha256"] = sha256(ROOT / "docs/native-trace-runc-prereg.md")
    protocol["treatment_sha256"] = RUN_B_FEATURES_SHA256
    protocol["treatment"] = treatment()[0]
    protocol["random_seed"] = RANDOM_SEED
    (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({k: protocol[k] for k in
                      ("seed", "counts", "excluded_tokens", "dataset_sha256")}, indent=2))
    return 0


def collect(a):
    """One unpatched observation per row: the greedy answer and the nine values."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_native_trace2 import post, server

    out = a.out
    input_gate(out)
    before_freeze(out)
    rows = read(out / "dataset.json")
    units = treatment()[0]
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
    probe = {"units": [[f["layer"], f["neuron"]] for f in units]}
    with server(out, a, "collect") as port, journal.open("a") as stream:
        for j, row in enumerate(rows):
            i = row["id"]
            if i in records:
                if records[i]["input_sha256"] != digest(row["ctx"]):
                    raise ValueError("cached context changed")
                continue
            start = time.monotonic()
            answer = post(port, row["ctx"], dict(probe, readouts=[row["expected"]]))
            native = answer.pop("observations")
            record = {"id": i, "input_sha256": digest(row["ctx"]),
                      "next": answer["next"], "next_logit": answer["next_logit"],
                      "logits": answer["logits"], "observations": native,
                      "seconds": time.monotonic() - start}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records[i] = record
            print(f"collect: {j+1}/{len(rows)}", flush=True)
    if set(records) != expected:
        raise ValueError("incomplete collection")
    write_json(out / "references.json", {str(i): r["next"] for i, r in records.items()})
    targets = {r["id"]: r["expected"] for r in rows}
    write_json(out / "base.json", {str(i): {"ref_logit": r["next_logit"],
                                             "target_logit": dict(r["logits"])[targets[i]]}
                                    for i, r in records.items()})


def plan(a):
    """Ask Datalog which patches to apply to which row. No model is touched."""
    out = a.out
    input_gate(out)
    before_freeze(out)
    rows = read(out / "dataset.json")
    units, widths = treatment()
    records = {r["id"]: r for r in map(json.loads, (out / "collection.jsonl").read_text().splitlines())}
    obs = []
    for row in rows:
        i = row["id"]
        last = len(row["ctx"]) - 1
        # The probe contract returns the LAST input position. Filter to it
        # explicitly: mixing positions would silently build one feature vector
        # out of several contexts' worth of state.
        found = {(o["layer"], o["neuron"]): o for o in records[i]["observations"]
                 if o["position"] == last}
        for f in units:
            unit = found.get((f["layer"], f["neuron"]))
            if unit is None:
                raise ValueError(f"missing unit L{f['layer']} n{f['neuron']} for row {i}")
            obs.append((i, f["layer"], f["neuron"], float(unit["product"])))
    facts = {
        "unit": "".join(f"{f['rank']}\t{f['layer']}\t{f['neuron']}\n" for f in units),
        "row": "".join(f"{r['id']}\t{int(r['negative'])}\t{r['gkey']}\t{len(r['ctx'])}\n" for r in rows),
        "obs": "".join(f"{i}\t{l}\t{n}\t{v:.17g}\n" for (i, l, n, v) in obs),
        "rset": "".join(f"{k}\t{l}\t{n}\n" for (k, l, n) in random_sets(widths)),
        "klist": "".join(f"{k}\n" for k in KS),
    }
    result = checked(out, "plan", DL / "native_trace_runc_plan.dl",
                     DL / "native_trace_runc_plan.dl", facts)
    relations = result["relations"]
    patches = {}
    for i, mode, k, layer, neuron, value in relations["patch"]:
        patches.setdefault((i, mode, k), []).append([int(layer), int(neuron), float(value)])
    requests = [{"i": int(i), "mode": mode, "k": int(k),
                  "patches": sorted(patches.get((i, mode, k), []))}
                 for (i, mode, k) in sorted(patches, key=lambda t: (int(t[0]), t[1], int(t[2])))]
    donors = {int(i): int(d) for (d, i) in relations["donor"]}
    write_json(out / "plan.json", {
        "requests": requests,
        "donor": {str(k): v for k, v in donors.items()},
        "nrequests": int(relations["nreq"][0][0]),
        "npatches": int(relations["npatch"][0][0]),
        "treatment": units,
        "random_sets": [{"k": k, "layer": l, "neuron": n} for (k, l, n) in random_sets(widths)],
    })
    print(f"plan: {relations['nreq'][0][0]} requests, {relations['npatch'][0][0]} patches")


def freeze(a):
    """Hash the inputs, the treatment, the plan and every source at the boundary."""
    out = a.out
    input_gate(out)
    if (out / "frozen.json").exists() or (out / "interventions.jsonl").exists():
        raise ValueError("freeze already exists or interventions began")
    if not (out / "plan.json").exists():
        raise ValueError("no plan to freeze")
    locked = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
              if p.is_file() and not p.name.startswith("server-")}
    sources = [Path(__file__), ROOT / "py/certificate.py", ROOT / "py/native_trace.py",
               ROOT / "py/benchmark_induction.py", ROOT / "py/benchmark_copy_generalization.py",
               ROOT / "py/run_native_trace2.py", ROOT / "docs/native-trace-runc-prereg.md",
               DL / "native_trace_runc_plan.dl", DL / "native_trace_runc_summary.dl",
               RUN_B_FEATURES]
    write_json(out / "frozen.json", {
        "status": "ready",
        "sha256": locked,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "note": "The intervention plan was frozen before the first intervention request."})
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


def intervene(a):
    """Execute the frozen plan, one request per planned intervention."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_native_trace2 import post, server

    out = a.out
    input_gate(out)
    validate_freeze(out)
    rows = {r["id"]: r for r in read(out / "dataset.json")}
    plan_doc = read(out / "plan.json")
    references = {int(k): v for k, v in read(out / "references.json").items()}
    journal = out / "interventions.jsonl"
    cached = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            key = (record["i"], record["mode"], record["k"])
            if key in cached:
                raise ValueError("duplicate intervention")
            cached[key] = record
    planned = {(r["i"], r["mode"], r["k"]) for r in plan_doc["requests"]}
    if not set(cached) <= planned:
        raise ValueError("unexpected intervention")
    with server(out, a, "intervene") as port, journal.open("a") as stream:
        for j, request in enumerate(plan_doc["requests"]):
            i, mode, k = request["i"], request["mode"], request["k"]
            key = (i, mode, k)
            if key in cached:
                continue
            row = rows[i]
            position = len(row["ctx"]) - 1
            patches = [{"layer": l, "position": position, "neuron": n, "value": v}
                       for (l, n, v) in request["patches"]]
            readouts = [references[i], row["expected"]]
            start = time.monotonic()
            answer = post(port, row["ctx"], {"patches": patches, "readouts": readouts})
            logits = dict(answer["logits"])
            record = {"i": i, "mode": mode, "k": k, "answer": answer["next"],
                      "answer_logit": answer["next_logit"],
                      "ref_logit": logits[references[i]],
                      "target_logit": logits[row["expected"]],
                      "npatches": len(patches), "seconds": time.monotonic() - start}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            cached[key] = record
            if (j + 1) % 100 == 0:
                print(f"intervene: {j+1}/{len(plan_doc['requests'])}", flush=True)
    if set(cached) != planned:
        raise ValueError("incomplete interventions")
    print(f"intervene: {len(cached)} requests")


def evaluate(a):
    """Ask Datalog what the effects were, and what they support."""
    out = a.out
    validate_freeze(out)
    rows = read(out / "dataset.json")
    plan_doc = read(out / "plan.json")
    references = {int(k): v for k, v in read(out / "references.json").items()}
    base = {int(k): v for k, v in read(out / "base.json").items()}
    records = [json.loads(line) for line in (out / "interventions.jsonl").read_text().splitlines()]
    facts = {
        "result": "".join(f"{r['i']}\t{r['mode']}\t{r['k']}\t{r['answer']}\t{r['answer_logit']:.17g}"
                          f"\t{r['ref_logit']:.17g}\t{r['target_logit']:.17g}\n" for r in records),
        "base": "".join(f"{i}\t{b['ref_logit']:.17g}\t{b['target_logit']:.17g}\n" for i, b in base.items()),
        "ref": "".join(f"{i}\t{t}\n" for i, t in references.items()),
        "row": "".join(f"{r['id']}\t{int(r['negative'])}\t{r['expected']}\n" for r in rows),
        "donor": "".join(f"{d}\t{i}\n" for i, d in plan_doc["donor"].items()),
        "request": "".join(f"{r['i']}\t{r['mode']}\t{r['k']}\n" for r in plan_doc["requests"]),
    }
    summary = checked(out, "summary", DL / "native_trace_runc_summary.dl",
                      DL / "native_trace_runc_summary.dl", facts)
    relations = summary["relations"]
    cost = sum(r.get("seconds", 0.0) for r in records)
    collection = [json.loads(line) for line in (out / "collection.jsonl").read_text().splitlines()]
    write_json(out / "report.json", {
        "tag": "empirical",
        "scope": "pinned f32 bundle, one checkpoint, 114 fresh contexts, last-position interventions",
        "summary": summary,
        "verdict": {
            "h1_off_path": bool(relations["h1_supported"]),
            "h2_redundancy": bool(relations["h2_supported"]),
            "h_sufficient": bool(relations["h_suf_supported"]),
            "relevant_not_sufficient": bool(relations["relevant_not_sufficient"]),
            "instrument_ok": bool(relations["instrument_ok"]),
        },
        "cost": {"interventions": len(records), "seconds": cost,
                 "collection_requests": len(collection),
                 "collection_seconds": sum(c.get("seconds", 0.0) for c in collection)},
    })
    print(json.dumps({"verdict": {k: bool(v) for k, v in relations.items()
                                  if k in ("h1_supported", "h2_supported", "h_suf_supported",
                                           "relevant_not_sufficient", "instrument_ok")},
                      "summary": relations["summary"]}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("inputs", "collect", "plan", "freeze", "intervene", "evaluate"))
    p.add_argument("out", type=Path)
    p.add_argument("--write", action="store_true",
                   help="for phase=inputs: write the study files; otherwise only report")
    p.add_argument("--binary", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--preflight", type=Path,
                   default=ROOT / "reference/benchmarks/native_trace_preflight")
    p.add_argument("--port", type=int, default=8393)
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
