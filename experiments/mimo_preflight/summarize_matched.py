"""Compare complete official-model and fieldrun outputs on the frozen suite."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("fieldrun", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    suite_hash = hashlib.sha256(raw).hexdigest()
    cases = json.loads(raw)["cases"]
    refs = [json.loads(line) for line in args.reference.read_text().splitlines()]
    fields = [json.loads(line) for line in args.fieldrun.read_text().splitlines()]
    if len(refs) != len(cases) or len(fields) != len(cases):
        raise ValueError("missing parity cases")
    by_family = defaultdict(lambda: {"n": 0, "top1_matches": 0})
    mismatches = []
    for case, ref, field in zip(cases, refs, fields):
        if case["id"] != ref["id"] or case["id"] != field["id"] or \
                ref["suite_sha256"] != suite_hash or field["suite_sha256"] != suite_hash:
            raise ValueError("case order or frozen suite hash changed")
        family = by_family[case["family"]]
        family["n"] += 1
        same = ref["top2_ids"][0] == field["top2_ids"][0]
        family["top1_matches"] += same
        if not same:
            mismatches.append({"id": case["id"], "family": case["family"],
                               "reference_top2": ref["top2_ids"], "fieldrun_top2": field["top2_ids"],
                               "reference_margin": ref["top2_logits"][0] - ref["top2_logits"][1]})
    result = {"tag": "empirical", "suite_sha256": suite_hash, "n": len(cases),
              "top1_matches": len(cases) - len(mismatches), "by_family": dict(by_family),
              "mismatches": mismatches}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"top1_matches": result["top1_matches"], "n": result["n"],
                      "mismatches": len(mismatches)}))


if __name__ == "__main__":
    main()
