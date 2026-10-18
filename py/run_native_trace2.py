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
import hashlib
import random
import json
from pathlib import Path

from benchmark_copy_generalization import generate
from benchmark_induction import digest, facts
from certificate import check, sha256
from native_trace import CHANNELS, DL, LAYERS, write_json

ROOT = Path(__file__).resolve().parents[1]

SEEDS = (201, 202, 203)
LENGTHS = (8, 12, 24)
GROUPS = 2
SPLIT_OF_SEED = {201: "select", 202: "sign", 203: "test"}
RANKING_SPLIT = "sign"

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
        units = {(o["layer"], o["neuron"]): o for o in records[r["id"]]["observations"]}
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("inputs", "features", "select"))
    p.add_argument("out", type=Path)
    p.add_argument("--write", action="store_true",
                   help="for phase=inputs: write the study files; otherwise only report")
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.phase == "inputs":
        return _inputs(a)
    {"features": features, "select": select}[a.phase](a.out)
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
    protocol = dict(report)
    protocol["dataset_sha256"] = sha256(a.out / "dataset.json")
    protocol["note"] = ("Generated by py/run_native_trace2.py inputs before any "
                        "run-B model request. No token is shared with the "
                        "historical datasets or with run A's frozen inputs.")
    (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({k: protocol[k] for k in
                      ("seeds", "split_of_seed", "excluded_tokens", "counts", "dataset_sha256")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
