"""Attribute frozen model predictions to matching prior copy sources with Souffle."""
import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

from certificate import sha256

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "dl" / "copy_source_attribution.dl"


def read_relation(directory, name):
    path = directory / f"{name}.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [tuple(int(value) for value in row) for row in csv.reader(handle, delimiter="\t") if row]


def audit(run, out, max_suffix=16):
    dataset_path = run / "dataset.json"
    refs_path = run / "references.json"
    rows = json.loads(dataset_path.read_text())
    refs = {int(key): int(value) for key, value in json.loads(refs_path.read_text()).items()}
    if set(refs) != {int(row["id"]) for row in rows}:
        raise ValueError("reference IDs do not exactly cover the saved dataset")

    with tempfile.TemporaryDirectory(prefix="rosetta-source-attribution-") as temp:
        temp = Path(temp)
        ind, result = temp / "in", temp / "out"
        ind.mkdir()
        result.mkdir()
        (ind / "tok.facts").write_text("".join(
            f"{row['id']}\t{position}\t{token}\n"
            for row in rows for position, token in enumerate(row["ctx"])))
        (ind / "ref.facts").write_text("".join(f"{row['id']}\t{refs[row['id']]}\n" for row in rows))
        (ind / "max_suffix.facts").write_text(f"{max_suffix}\n")
        proc = subprocess.run(["souffle", str(PROGRAM), "-F", str(ind), "-D", str(result)],
                             capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip() or "Souffle attribution query failed")
        best = {instance: length for instance, length in read_relation(result, "best_specificity")}
        best_count = {instance: count for instance, count in read_relation(result, "best_source_count")}
        nearest_hit = {instance for (instance,) in read_relation(result, "nearest_source_hit")}
        longest_hit = {instance for (instance,) in read_relation(result, "longest_source_hit")}
        observations = read_relation(result, "source_observation")
        hits = read_relation(result, "source_hit")
        answer_source = {instance: (position, length, distance)
                         for instance, position, length, distance in read_relation(result, "answer_source")}

    sources_by_id = {}
    for instance, position, length, distance, successor in observations:
        sources_by_id.setdefault(instance, []).append({"position": position, "match_length": length,
                                                        "distance": distance, "successor": successor})
    hits_by_id = {}
    for instance, position, length, distance in hits:
        hits_by_id.setdefault(instance, []).append({"position": position, "match_length": length,
                                                     "distance": distance})

    records = []
    for row in rows:
        instance = int(row["id"])
        if row["part"] != "test":
            continue
        target = refs[instance]
        record = {"id": instance, "layout": row["layout"], "kind": row["kind"],
                  "answer": target, "best_suffix": best.get(instance),
                  "best_source_count": best_count.get(instance, 0),
                  "answer_matches_nearest_source": instance in nearest_hit,
                  "answer_matches_a_best_specificity_source": instance in longest_hit,
                  "answer_source_attribution": answer_source.get(instance),
                  "source_observations": sources_by_id.get(instance, []),
                  "matching_sources": hits_by_id.get(instance, [])}
        records.append(record)

    summaries = {}
    for layout in sorted({record["layout"] for record in records}):
        subset = [record for record in records if record["layout"] == layout]
        attributable = [record for record in subset if record["answer_source_attribution"] is not None]
        distance_bins, matched_lengths = {}, {}
        for record in attributable:
            _position, length, distance = record["answer_source_attribution"]
            matched_lengths[str(length)] = matched_lengths.get(str(length), 0) + 1
            bin_name = "0-15" if distance < 16 else "16-31" if distance < 32 else \
                       "32-63" if distance < 64 else "64+"
            distance_bins[bin_name] = distance_bins.get(bin_name, 0) + 1
        summaries[layout] = {
            "n": len(subset),
            "answer_matches_any_source": len(attributable),
            "answer_matches_nearest_source": sum(r["answer_matches_nearest_source"] for r in subset),
            "answer_matches_best_specificity_source": sum(
                r["answer_matches_a_best_specificity_source"] for r in subset),
            "answer_matches_no_source": len(subset) - len(attributable),
            "matched_source_suffix_lengths": matched_lengths,
            "matched_source_distance_bins": distance_bins,
        }

    bundle_hashes = None
    protocol_path = run / "protocol.json"
    if protocol_path.exists():
        provenance = json.loads(protocol_path.read_text()).get("provenance", {})
        bundle_hashes = provenance.get("bundle_sha256")
    report = {"tag": "empirical", "run": str(run), "max_suffix": max_suffix,
              "dataset_sha256": sha256(dataset_path), "references_sha256": sha256(refs_path),
              "program_sha256": sha256(PROGRAM), "bundle_sha256": bundle_hashes,
              "test_summary_by_layout": summaries, "cases": records}
    out.mkdir(parents=True, exist_ok=False)
    (out / "attribution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test_summary_by_layout": summaries,
                      "dataset_sha256": report["dataset_sha256"],
                      "references_sha256": report["references_sha256"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="frozen run directory with dataset.json and references.json")
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-suffix", type=int, default=16)
    args = parser.parse_args()
    if args.max_suffix < 1:
        parser.error("--max-suffix must be positive")
    audit(args.run.resolve(), args.out.resolve(), args.max_suffix)


if __name__ == "__main__":
    main()
