"""Compare candidate source-selection policies on one frozen stress corpus."""
import argparse
import json
from pathlib import Path

import benchmark_guarded_copy as guarded
from certificate import sha256
from oracle import serve_decide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--bundle")
    parser.add_argument("--port", type=int)
    parser.add_argument("--control", action="store_true",
                        help="use authored expected labels as a deterministic copy-rule control")
    parser.add_argument("--expanded-search", action="store_true")
    args = parser.parse_args()
    dataset_path = args.dataset.resolve()
    rows = json.loads(dataset_path.read_text())
    if [row["id"] for row in rows] != list(range(len(rows))):
        raise ValueError("dataset IDs must be contiguous and ordered")
    dataset_hash = sha256(dataset_path)
    if not args.control and (not args.bundle or not args.port):
        parser.error("fieldrun runs require --bundle and --port")
    bundle_hashes = ({suffix: sha256(args.bundle + suffix) for suffix in (".fieldrun.bin", ".fieldrun.json")}
                     if not args.control else None)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def provenance(policy, cache=None):
        search = (("suffix lengths 1..16" if policy == "ranked" else "suffix lengths 1..12") +
                  ", support 1..3, horizons 16/32/64/128" if policy in ("recent", "ranked")
                  else "suffix lengths 1..12, support 1..4" if args.expanded_search
                  else "suffix lengths 3,6,9; support 2,3")
        result = {"source": "copy-control" if args.control else "fieldrun",
                  "synthetic": args.control, "bundle_sha256": bundle_hashes,
                  "dataset_sha256": dataset_hash,
                  "search": search, "policy": policy,
                  "reference_policy": "collected online after each candidate freeze" if cache is None
                  else "reused by case ID from same-checkpoint frozen reference run after policy was specified"}
        if cache is not None:
            result["reference_run_sha256"] = sha256(cache / "references.json")
        return result

    oracle = (lambda row: row["expected"]) if args.control else lambda row: serve_decide(args.port, row["ctx"])
    runs = {}
    policies = ("consensus", "plurality", "recent", "ranked")
    references = None
    reference_run = None
    for policy in policies:
        guarded.GUARDS = (guarded.RECENCY_GUARDS if policy == "recent" else
                          guarded.RANKED_GUARDS if policy == "ranked" else
                          guarded.EXPANDED_GUARDS if args.expanded_search else
                          [(i + 1, k, m) for i, (k, m) in enumerate((k, m) for k in (3, 6, 9) for m in (2, 3))])
        policy_out = out / policy
        active_oracle = oracle if references is None else lambda row, refs=references: refs[row["id"]]
        runs[policy] = guarded.benchmark(
            policy_out, rows, active_oracle, provenance(policy, reference_run),
            policy, guarded.RECENCY_HORIZON_BY_ID if policy == "recent" else
            guarded.RANKED_HORIZON_BY_ID if policy == "ranked" else None)
        if references is None:
            references = {int(k): v for k, v in json.loads((policy_out / "references.json").read_text()).items()}
            if len(references) != len(rows):
                raise ValueError("consensus run did not preserve a reference for every input row")
            reference_run = policy_out
    def artifact_summary(report, name):
        result = report["results"][name]
        return {"selected": report["selected"] if name == "selected" else None,
                "test": result["scores"]["test"],
                "off_domain": result["scores"]["off_domain"],
                "firing_domain_certificate": result["certificates"]["frozen_guard_domain"]}

    summary = {"tag": "empirical", "dataset_sha256": dataset_hash, "bundle_sha256": bundle_hashes,
               "source": "copy-control" if args.control else "fieldrun", "policies": {}}
    for policy in policies:
        summary["policies"][policy] = {"selected": artifact_summary(runs[policy], "selected"),
                                        "strict_guard": artifact_summary(runs[policy], "strict_guard")}
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
