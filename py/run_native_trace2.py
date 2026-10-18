"""Run-B input generation: fresh seeds 201/202/203 as select / sign / test.

Run A's `dataset.json` was committed as an artifact with no committed generator,
so it could be hashed and replayed but not regenerated. Run B generates its
inputs from this file instead, which is the reproducibility improvement worth
having before the design is locked.

Every token used by the historical Qwen copy datasets, and every token used by
run A's own frozen inputs, is excluded. The generator additionally keeps each
seed's vocabulary disjoint from the others as it draws, so select/sign/test are
group- and token-disjoint by construction rather than by later filtering.

No model, no weights, no network: this is data generation only.
"""

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_copy_generalization import generate

ROOT = Path(__file__).resolve().parents[1]

SEEDS = (201, 202, 203)
LENGTHS = (8, 12, 24)
GROUPS = 2
SPLIT_OF_SEED = {201: "select", 202: "sign", 203: "test"}

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path, help="run-B study directory")
    p.add_argument("--write", action="store_true",
                   help="write dataset.json and protocol.json; otherwise only report")
    a = p.parse_args()

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
    protocol["dataset_sha256"] = hashlib.sha256(
        (a.out / "dataset.json").read_bytes()).hexdigest()
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