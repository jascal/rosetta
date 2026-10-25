"""Short-suffix development: input/oracle I/O; candidate and admission logic in Datalog."""
import argparse
import json
from pathlib import Path
import random
import time

from certificate import check, sha256
from run_copy_mechanism import server, post, read, write
from run_native_trace2 import HISTORICAL, RUN_A, _tokens

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference/benchmarks/qwen25_copy_prefix"
BASELINE = ROOT / "reference/benchmarks/qwen25_native_trace_b/artifacts/behavior/circuits.dl"
EXCLUSIONS = HISTORICAL + RUN_A + (
    "reference/benchmarks/qwen25_native_trace_b/dataset.json",
    "reference/benchmarks/qwen25_native_trace_c/dataset.json",
)
OPTIONS = [(i, k, m) for i, (k, m) in enumerate(
    ((k, m) for k in (1, 2) for m in (0, 16, 32)), 1)]
SOURCES = ("py/run_copy_prefix.py", "py/run_copy_mechanism.py", "py/run_native_trace2.py",
           "py/certificate.py", "dl/copy_prefix_runtime.dl", "dl/copy_prefix_select.dl",
           "docs/copy-prefix-development-plan.md")


def generate(excluded, seeds=(401, 402), groups=4):
    used, rows, pairs = set(excluded), [], []
    for seed, split in zip(seeds, ("train", "validation")):
        rng = random.Random(seed)
        for length in (8, 12, 24):
            for group in range(groups):
                for negative in (False, True):
                    values = rng.sample([t for t in range(200, 40000) if t not in used], length + 34)
                    used.update(values)
                    seq, replacement, decoy, noise = values[:length], values[length], values[length+1], values[length+2:]
                    meta = {"seed": seed, "split": split, "group": f"s{seed}-l{length}-g{group}-n{int(negative)}",
                            "length": length, "negative": negative}
                    if negative:
                        for kind, ctx in (("no_repeat", seq),
                            ("conflicting_sources", seq[:2]+[replacement]+seq[3:]+seq+seq[:2])):
                            rows.append(dict(meta, id=len(rows), prefix=2, kind=kind, layout="control", ctx=ctx,
                                             expected=seq[2]))
                        continue
                    for prefix in (1, 2, 3):
                        for layout in ("plain", "noise8", "noise32", "near1"):
                            gap = [] if layout == "plain" else noise[:32 if layout == "noise32" else 8]
                            if layout == "near1":
                                gap = gap[:-2] + [seq[prefix-1], decoy]
                            context = seq + gap + seq[:prefix]
                            a = len(rows)
                            rows.append(dict(meta, id=a, prefix=prefix, kind="repeat", layout=layout,
                                             ctx=context, expected=seq[prefix]))
                            edited = context[:]
                            edited[prefix] = replacement
                            b = len(rows)
                            rows.append(dict(meta, id=b, prefix=prefix, kind="intervention", layout=layout,
                                             ctx=edited, expected=replacement, base=a, positions=[prefix]))
                            pairs.append({"a": a, "b": b, "group": meta["group"], "positions": [prefix]})
    return rows, pairs


def disjoint(rows, excluded):
    splits = {}
    groups = {}
    for r in rows:
        splits.setdefault(r["split"], set()).update(r["ctx"])
        groups.setdefault(r["split"], set()).add(r["group"])
    names = sorted(splits)
    for s in names:
        if splits[s] & excluded:
            raise ValueError("historical token leakage")
        for t in names:
            if s != t and (splits[s] & splits[t] or groups[s] & groups[t]):
                raise ValueError("development splits overlap")
    return {s: {"tokens": len(splits[s]), "groups": len(groups[s]),
                 "rows": sum(r["split"] == s for r in rows)} for s in names}


def library():
    baseline = BASELINE.read_text().replace("cdecide", "bdecide")
    return (baseline + "\n.decl cp_option(c:number,k:number,minlen:number)\n" +
            "".join(f"cp_option({i},{k},{m}).\n" for i, k, m in OPTIONS) +
            (ROOT / "dl/copy_prefix_runtime.dl").read_text())


def inputs(a):
    a.out.mkdir(parents=True, exist_ok=False)
    exclude = set()
    for name in EXCLUSIONS:
        exclude.update(_tokens(name))
    rows, pairs = generate(exclude)
    stats = disjoint(rows, exclude)
    write(a.out / "dataset.json", rows)
    write(a.out / "pairs.json", pairs)
    write(a.out / "excluded-tokens.json", sorted(exclude))
    (a.out / "library.dl").write_text(library())
    (a.out / "baseline.dl").write_text(BASELINE.read_text())
    paths = [a.out / n for n in ("dataset.json", "pairs.json", "excluded-tokens.json", "library.dl", "baseline.dl")]
    write(a.out / "protocol.json", {
        "status": "development only; no final references", "seeds": [401, 402], "groups_per_length": 4,
        "options": OPTIONS, "disjoint": stats,
        "files": {p.name: sha256(p) for p in paths},
        "sources": {n: sha256(ROOT / n) for n in SOURCES},
        "exclusions": {n: sha256(ROOT / n) for n in EXCLUSIONS},
        "baseline_sha256": sha256(BASELINE),
        "pinned_model": read(ROOT / "reference/benchmarks/qwen25_native_trace_b/model.json")})
    print(json.dumps(stats, indent=2))


def gate(a):
    p = read(a.out / "protocol.json")
    for name, expected in p["files"].items():
        if sha256(a.out / name) != expected:
            raise ValueError(f"development input changed: {name}")
    for name, expected in (p["sources"] | p["exclusions"]).items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"development source changed: {name}")
    if sha256(BASELINE) != p["baseline_sha256"]:
        raise ValueError("baseline changed")
    if a.binary and sha256(a.binary) != p["pinned_model"]["binary_sha256"]:
        raise ValueError("different executable")
    if a.bundle and sha256(str(a.bundle) + ".fieldrun.bin") != p["pinned_model"]["bundle_sha256"][".fieldrun.bin"]:
        raise ValueError("different model weights")
    return read(a.out / "dataset.json")


def journal(path):
    result = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["id"] in result:
                raise ValueError("duplicate reference")
            result[r["id"]] = r
    return result


def collect(a):
    rows = gate(a)
    path = a.out / "observations.jsonl"
    cached = journal(path)
    if set(cached) - {r["id"] for r in rows}:
        raise ValueError("unplanned observation")
    with server(a, "development") as port, path.open("a") as stream:
        for j, row in enumerate(rows):
            if row["id"] in cached:
                continue
            start = time.monotonic()
            x = post(port, row["ctx"], {"readouts": [row["expected"]]})
            record = {"id": row["id"], "response": x, "seconds": time.monotonic()-start}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if (j+1) % 25 == 0:
                print(f"development {j+1}/{len(rows)}", flush=True)


def select(a):
    rows = gate(a)
    refs = journal(a.out / "observations.jsonl")
    pairs = read(a.out / "pairs.json")
    facts = {
        "tok": "".join(f"{r['id']}\t{j}\t{t}\n" for r in rows for j, t in enumerate(r["ctx"])),
        "sample": "".join(f"{r['id']}\t{r['split']}\t{r['group']}\n" for r in rows),
        "ref": "".join(f"{i}\t{r['response']['next']}\n" for i, r in refs.items()),
        "pair": "".join(f"{p['a']}\t{p['b']}\t{p['group']}\n" for p in pairs),
        "edit": "".join(f"{p['a']}\t{p['b']}\t{j}\n" for p in pairs for j in p["positions"])}
    result = check(ROOT / "dl/copy_prefix_select.dl", a.out / "library.dl", facts, {},
                   evidence_dir=a.out / "certificate-evidence/development-select",
                   provenance={"source": "pinned f32 Qwen; six-candidate development selection; not final test"})
    write(a.out / "selection.json", result)
    if not result["certified"]:
        raise ValueError(result.get("error", "invalid development data"))
    print(json.dumps({k: result["relations"][k] for k in ("score", "ngroups", "cp_selected")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("inputs", "collect", "select"))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--binary", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--port", type=int, default=8396)
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.phase == "collect" and not (a.binary and a.bundle):
        p.error("collection needs --binary and --bundle")
    globals()[a.phase](a)


if __name__ == "__main__":
    main()
