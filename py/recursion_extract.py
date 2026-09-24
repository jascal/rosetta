#!/usr/bin/env python3
"""Stage graph/ref facts and report Souffle's selected reachability candidate."""
import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from recursion_ladder import read_jsonl

ROOT = Path(__file__).resolve().parent.parent
PROGRAM = ROOT / "dl" / "reach_search.dl"
RULES = {0: "constant_no", 1: "constant_yes", 2: "direct_edge",
         3: "at_most_two_hops", 4: "recursive_reach"}


def extract(cases, references, train_ids):
    all_ids = {c["id"] for c in cases}
    if len(all_ids) != len(cases) or set(references) != all_ids:
        raise ValueError("one binary reference is required for each distinct case")
    train_ids = set(train_ids)
    if not train_ids or not train_ids < all_ids:
        raise ValueError("training and holdout sets must both be nonempty")
    if set(references.values()) - {0, 1}:
        raise ValueError("references must be binary")
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        facts, output = base / "facts", base / "output"
        facts.mkdir()
        output.mkdir()
        for name, rows in {
            "edge": ((c["id"], a, b) for c in cases for a, b in c["edges"]),
            "query": ((c["id"], *c["query"]) for c in cases),
            "ref": ((i, references[i]) for i in sorted(all_ids)),
            "train": ((i,) for i in sorted(train_ids)),
            "test": ((i,) for i in sorted(all_ids - train_ids)),
        }.items():
            (facts / f"{name}.facts").write_text("".join("\t".join(map(str, row)) + "\n" for row in rows))
        result = subprocess.run(["souffle", str(PROGRAM), "-F", str(facts), "-D", str(output)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)

        def rows(name):
            return [tuple(map(int, line.split("\t"))) for line in
                    (output / f"{name}.csv").read_text().splitlines() if line]

        chosen = rows("selected")
        best = rows("best_candidate")
        train_errors = {RULES[r]: n for r, n in rows("training_error_count")}
        return {"selected": RULES[chosen[0][0]] if chosen else None,
                "selected_rule_id": chosen[0][0] if chosen else None,
                "best_fit": RULES[best[0][0]] if best else None,
                "best_fit_rule_id": best[0][0] if best else None,
                "train_cases": len(train_ids), "holdout_cases": len(all_ids - train_ids),
                "training_errors_by_candidate": train_errors,
                "holdout_errors": len(rows("holdout_error")) if chosen else None,
                "best_fit_holdout_errors": len(rows("best_holdout_error")) if best else None,
                "holdout_error_ids": [i for _, i in rows("holdout_error")],
                "best_fit_holdout_error_ids": [i for i, in rows("best_holdout_error")]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--responses", type=Path, action="append",
                        help="response JSONL; repeat to combine files and iteration budgets")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = read_jsonl(args.cases)
    # Pair-preserving split: trials 0..9 train, 10..19 held out for saved
    # datasets. The rule-selection decision remains in Datalog.
    trials = sorted({c["trial"] for c in cases})
    training_trials = set(trials[:len(trials) // 2])
    train_ids = {c["id"] for c in cases if c["trial"] in training_trials}
    if args.responses:
        groups = defaultdict(list)
        for path in args.responses:
            for row in read_jsonl(path):
                key = (row["architecture"], int(row["budget"]), row.get("prompt_style", "plain"))
                groups[key].append(row)
        results = []
        for (architecture, budget, style), rows in sorted(groups.items()):
            if len(rows) != len(cases) or any(r["answer"] is None for r in rows):
                parser.error(f"{architecture} budget {budget} prompt {style} needs one binary answer per case")
            result = extract(cases, {r["id"]: r["answer"] for r in rows}, train_ids)
            results.append({"architecture": architecture, "budget": budget,
                            "prompt_style": style, **result})
        result = {"domain": {"cases": len(cases), "training_trials": sorted(training_trials),
                  "holdout_trials": sorted(set(trials) - training_trials)},
                  "architectures": results}
    else:
        result = extract(cases, {c["id"]: c["answer"] for c in cases}, train_ids)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
