"""Record top-two fieldrun scores for a frozen copy dataset; margins are diagnostics only."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from oracle import serve_topk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="completed benchmark run directory")
    parser.add_argument("out", type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--seed", type=int, help="optional generator-seed subset")
    parser.add_argument("--positive-only", action="store_true", help="omit off-domain controls")
    args = parser.parse_args()
    run = args.run.resolve()
    rows = json.loads((run / "dataset.json").read_text())
    refs = {int(k): v for k, v in json.loads((run / "references.json").read_text()).items()}
    test = [row for row in rows if row["part"] == "test"
            and (args.seed is None or row.get("seed") == args.seed)
            and (not args.positive_only or row["kind"] in ("repeat", "intervention"))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    recorded = []
    with args.out.open("w") as output:
        for n, row in enumerate(test, 1):
            top = serve_topk(args.port, row["ctx"], k=2)
            margin = top[0][1] - top[1][1]
            record = {"id": row["id"], "group": row["group"], "kind": row["kind"],
                      "exposures": row.get("exposures"), "prefix": row.get("prefix"),
                      "length": row.get("length"), "layout": row.get("layout"),
                      "reference_top1": refs[row["id"]], "reference_match": top[0][0] == refs[row["id"]],
                      "task_expected": row["expected"], "task_correct": top[0][0] == row["expected"],
                      "top2": [[token, logit] for token, logit in top], "top1_margin": margin}
            recorded.append(record)
            output.write(json.dumps(record) + "\n")
            output.flush()
            if n % 24 == 0:
                print(f"top-two scores: {n}/{len(test)}", flush=True)
    if len(recorded) != len(test):
        raise ValueError("incomplete margin collection")
    if any(not row["reference_match"] for row in recorded):
        raise ValueError("topk top-1 differs from the frozen benchmark reference")
    groups = {}
    for correct in (True, False):
        margins = [row["top1_margin"] for row in recorded if row["task_correct"] is correct]
        groups["task_correct" if correct else "task_wrong"] = {
            "n": len(margins),
            "median_margin": statistics.median(margins) if margins else None,
            "mean_margin": statistics.mean(margins) if margins else None,
            "min_margin": min(margins) if margins else None,
            "max_margin": max(margins) if margins else None,
        }
    by_layout = {}
    for layout in sorted({row["layout"] for row in recorded if row["layout"] is not None}):
        by_layout[layout] = {}
        for correct in (True, False):
            margins = [row["top1_margin"] for row in recorded
                       if row["layout"] == layout and row["task_correct"] is correct]
            by_layout[layout]["task_correct" if correct else "task_wrong"] = {
                "n": len(margins),
                "median_margin": statistics.median(margins) if margins else None,
                "mean_margin": statistics.mean(margins) if margins else None,
            }
    summary = {"tag": "empirical", "diagnostic_only": True, "n": len(recorded),
               "reference_top1_matches": sum(row["reference_match"] for row in recorded),
               "seed_filter": args.seed, "positive_only": args.positive_only,
               "dataset_sha256": hashlib.sha256((run / "dataset.json").read_bytes()).hexdigest(),
               "bundle": args.bundle, "margin_by_task_result": groups, "margin_by_layout": by_layout}
    args.out.with_suffix(args.out.suffix + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
