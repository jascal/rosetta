"""I/O and artifact serialization for the preregistered native-trace experiment.

Selection, numeric trace statistics, runtime predictions, and certification live
in Datalog. This module generates inputs and stages/records the query results.
"""
import argparse
import itertools
import json
from pathlib import Path

from benchmark_copy_generalization import generate
from benchmark_induction import DL
from certificate import sha256

ROOT = DL.parent
LAYERS = (7, 15, 23)
CHANNELS = ("gate", "value", "product")
PRIOR = ROOT / "reference/benchmarks/qwen25_05b_copy_generalization/dataset.json"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def grammar():
    """Serialize the fixed, preregistered grammar, not a learned decision."""
    bases = [(0, k, m) for k in (1, 3, 6, 9) for m in (1, 2, 3)]
    bases += [(1, k, 1) for k in (1, 3, 6, 9)]
    bases += [(2, 12, 1)]
    bases += [(3, k, m) for k in (3, 6, 9) for m in (2, 3)]
    bases += [(4, 1, 1)]
    guards = [(0, 0)] + [(a, 0) for a in range(1, 11)] + list(itertools.combinations(range(1, 11), 2))
    return [(i+1, *b) for i, b in enumerate(bases)], [
        (i+1, b, a, d) for i, (b, a, d) in enumerate(
            (b, a, d) for b in range(1, len(bases)+1) for a, d in guards)]


def program(selected=(), *, candidates=None):
    """selected is exactly a Datalog output relation (rank,candidate)."""
    bases, universe = grammar()
    if candidates is not None:
        # Serialization filter over the selected identifiers, never admission.
        universe = [c for c in universe if c[0] in set(candidates)]
        bases = [b for b in bases if b[0] in {c[1] for c in universe}]
    lines = ["// Token-only artifact; no native model observations are runtime inputs.",
             ".decl base(b:number,kind:number,k:number,m:number)",
             ".decl candidate(c:number,b:number,a:number,d:number)",
             ".decl selected(rank:number,c:number)"]
    for relation, tuples in (("base", bases), ("candidate", universe), ("selected", selected)):
        lines.extend(f"{relation}({','.join(map(str, row))})." for row in tuples)
    return "\n".join(lines) + "\n" + (DL / "native_trace_runtime.dl").read_text()


def emit(out, selected):
    out.mkdir(parents=True, exist_ok=False)
    path = out / "circuits.dl"
    path.write_text(program(selected, candidates=[c for _, c in selected]))
    (out / "run.dl").write_text(
        '.decl tok(inst:number,pos:number,id:number) .input tok\n'
        '.decl domain(inst:number) .input domain\n#include "circuits.dl"\n'
        '.output cdecide\n.decl abstain(inst:number) .output abstain\n'
        'abstain(I) :- domain(I), !cdecide(I,_).\n')
    return path


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    priors = sorted((ROOT / "reference/benchmarks").glob("qwen*/dataset.json"))
    excluded = {t for path in priors for row in json.loads(path.read_text()) for t in row["ctx"]}
    rows = []
    for seed, part in ((101, "train"), (102, "validation"), (103, "test")):
        batch = generate(seeds=(seed,), lengths=(8, 12, 24), groups=2, exclude=excluded)
        mapping = {r["id"]: len(rows)+n for n, r in enumerate(batch)}
        for r in batch:
            r["negative"] = r["part"] == "off_domain"
            r["part"] = part
            r["id"] = mapping[r["id"]]
            if "base" in r:
                r["base"] = mapping[r["base"]]
        rows.extend(batch)
        excluded.update(t for r in batch for t in r["ctx"])
    discovery = [r for r in json.loads(PRIOR.read_text()) if r["part"] == "test"]
    write_json(out / "dataset.json", rows)
    write_json(out / "discovery.json", discovery)
    (out / "hypotheses.dl").write_text(program())
    write_json(out / "protocol.json", {
        "prereg_sha256": sha256(ROOT / "docs/native-trace-prereg.md"),
        "prereg_commit": "0719246", "dataset_sha256": sha256(out / "dataset.json"),
        "discovery_sha256": sha256(out / "discovery.json"),
        "historical_dataset_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in priors},
        "layers": LAYERS, "channels": CHANNELS, "budget_per_arm": 12,
        "hypotheses_sha256": sha256(out / "hypotheses.dl"),
        "nbase": len(grammar()[0]), "ncandidates": len(grammar()[1]),
        "counts": {part: sum(r["part"] == part for r in rows) for part in ("train", "validation", "test")},
        "source_sha256": {name: sha256(DL / name) for name in
            ("native_trace_runtime.dl", "native_trace_features.dl", "native_trace_select.dl")},
        "driver_sha256": sha256(__file__),
        "note": "Inputs fixed before model observations; final harness/source hashes are frozen again before test references."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare",))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    prepare(args.out)


if __name__ == "__main__":
    main()
