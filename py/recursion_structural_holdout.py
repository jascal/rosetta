#!/usr/bin/env python3
"""Select reachability from four-node examples and certify on larger new graphs."""
import argparse
import json
from pathlib import Path

from recursion_extract import extract
from recursion_ladder import certify, write_jsonl


TRAIN = [
    ([(0, 1)], (0, 1)),
    ([(0, 1)], (1, 0)),
    ([(0, 1), (1, 2)], (0, 2)),
    ([(0, 1), (1, 2)], (2, 0)),
    ([(0, 1), (1, 2), (2, 3)], (0, 3)),
    ([(0, 1), (1, 2), (2, 3)], (3, 0)),
    ([(0, 1), (0, 2), (2, 3)], (0, 3)),
    ([(0, 1), (1, 2), (2, 0)], (0, 3)),
]
HOLDOUT = [
    ([(10, 11), (11, 12), (12, 13), (13, 14)], (10, 14)),
    ([(10, 11), (11, 12), (12, 13), (13, 14)], (14, 10)),
    ([(20, 21), (20, 22), (22, 23), (23, 24), (24, 25), (25, 22)], (20, 25)),
    ([(30, 31), (31, 32), (32, 33), (30, 33), (34, 35)], (30, 33)),
    ([(40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 40)], (40, 43)),
    ([(50, 51), (51, 52), (52, 53), (53, 54), (54, 55)], (50, 55)),
]


def reachable(edges, query):
    source, target = query
    seen, frontier = {source}, [source]
    while frontier:
        node = frontier.pop(0)
        if node == target:
            return True
        for a, b in edges:
            if a == node and b not in seen:
                seen.add(b)
                frontier.append(b)
    return False


def build_cases():
    rows = []
    for split, group in (("train", TRAIN), ("holdout", HOLDOUT)):
        for edges, query in group:
            rows.append({"id": len(rows), "split": split, "edges": [list(e) for e in edges],
                         "query": list(query), "answer": int(reachable(edges, query))})
    return rows


def run(out_dir):
    cases = build_cases()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "structural_holdout_cases.jsonl", cases)
    train_ids = {c["id"] for c in cases if c["split"] == "train"}
    refs = {c["id"]: c["answer"] for c in cases}
    search = extract(cases, refs, train_ids)
    certificate = certify(cases, refs)
    result = {"domain": {"training_graphs": "four-node", "holdout_graph_sizes": [5, 6],
                         "training_cases": len(train_ids), "holdout_cases": len(cases) - len(train_ids),
                         "holdout_features": ["new node identifiers", "cycles", "branches",
                                              "shortcut edge", "disconnected component"]},
              "candidate_search": search, "certificate": certificate}
    (out_dir / "structural_holdout_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "experiments" / "recursion_ladder")
    args = parser.parse_args()
    print(json.dumps(run(args.out_dir), indent=2))
