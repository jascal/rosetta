"""Summarize paired copy-guard policy runs without querying an oracle."""
import argparse
import json
from pathlib import Path

from certificate import sha256


def artifact(report, frozen_path, dataset, name):
    result = report["results"][name]
    domain = sorted(json.loads(frozen_path.read_text())["artifacts"][name]["domain"])
    certificate = result["certificates"]["frozen_guard_domain"]
    wrong = {domain[int(row[0])] for row in certificate["mismatches"]}
    by_layout = {}
    for row in dataset:
        if row["part"] != "test":
            continue
        layout = row["layout"]
        item = by_layout.setdefault(layout, {"n": 0, "fired": 0, "correct": 0, "wrong": 0, "abstained": 0})
        item["n"] += 1
        if row["id"] in domain:
            item["fired"] += 1
            if row["id"] in wrong:
                item["wrong"] += 1
            else:
                item["correct"] += 1
        else:
            item["abstained"] += 1
    return {"selected": report["selected"] if name == "selected" else None,
            "test": result["scores"]["test"], "off_domain": result["scores"]["off_domain"],
            "firing_domain_certificate": certificate, "firing_by_layout": by_layout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="paired consensus/plurality output directory")
    args = parser.parse_args()
    run = args.run
    policies = [name for name in ("consensus", "plurality", "recent", "ranked")
                if (run / name / "report.json").exists()]
    if not policies:
        raise ValueError("no policy reports found")
    reports = {name: json.loads((run / name / "report.json").read_text()) for name in policies}
    suite_hashes = {reports[name]["provenance"]["dataset_sha256"] for name in policies}
    if len(suite_hashes) != 1:
        raise ValueError("paired policy reports do not share the frozen dataset hash")
    dataset = json.loads((run / "consensus/dataset.json").read_text())
    result = {"tag": "empirical", "dataset_sha256": sha256(run / "consensus/dataset.json"),
              "bundle_sha256": reports["consensus"]["provenance"].get("bundle_sha256"),
              "source": reports["consensus"]["provenance"]["source"], "policies": {}}
    for name in policies:
        report = reports[name]
        result["policies"][name] = {
            "search": report["provenance"]["search"],
            "selected": artifact(report, run / name / "frozen.json", dataset, "selected"),
            "strict_guard": artifact(report, run / name / "frozen.json", dataset, "strict_guard"),
        }
    (run / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
