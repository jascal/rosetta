"""Generate a fresh split copy-stress set with exact competing successor sources."""
import argparse
import json
import random
from pathlib import Path

from benchmark_copy_generalization import DEFAULT_PRIOR
from certificate import sha256


def generate(seed=37, length=32, groups=(2, 2, 4, 2), exclude=()):
    if length < 26 or len(groups) != 4 or min(groups) < 1:
        raise ValueError("need length >=26 and nonempty train/validation/test/off-domain groups")
    used = set(exclude)
    pool = list(range(200, 40000))
    rng = random.Random(seed)
    rows = []
    layouts = ("plain", "noise32", "near_match_suffix12")
    for part, group_count in zip(("train", "validation", "test", "off_domain"), groups):
        for group_index in range(group_count):
            available = [token for token in pool if token not in used]
            values = rng.sample(available, length + 98)
            used.update(values)
            seq = values[:length]
            replacement, decoy = values[length:length + 2]
            noise = [values[length + 2 + 32 * b:length + 2 + 32 * (b + 1)] for b in range(3)]
            prefix = length // 2
            label = f"seed{seed}-{part}-{group_index}"
            common = dict(seed=seed, length=length, prefix=prefix, part=part, group=label)
            if part == "off_domain":
                conflict = seq * 2 + seq[:prefix]
                conflict[prefix] = replacement
                controls = (("no_repeat", seq),
                            ("one_source", seq + noise[0][:8] + seq[:prefix]),
                            ("conflicting_sources", conflict))
                for kind, context in controls:
                    rows.append(dict(common, id=len(rows), kind=kind, layout="negative", exposures=0,
                                     ctx=context, expected=seq[prefix]))
                continue
            for exposures in (2, 3):
                for layout in layouts:
                    context, source_positions = [], []
                    for block in range(exposures):
                        source_positions.append(len(context) + prefix)
                        context.extend(seq)
                        gap = [] if layout == "plain" else noise[block][:32]
                        if layout == "near_match_suffix12" and block == exposures - 1:
                            gap[-13:] = [*seq[prefix - 12:prefix], decoy]
                        context.extend(gap)
                    context.extend(seq[:prefix])
                    variant = dict(common, exposures=exposures, layout=layout)
                    base = len(rows)
                    rows.append(dict(variant, id=base, kind="repeat", ctx=context, expected=seq[prefix]))
                    edited = context[:]
                    for position in source_positions:
                        edited[position] = replacement
                    rows.append(dict(variant, id=len(rows), kind="intervention", base=base,
                                     positions=source_positions, ctx=edited, expected=replacement))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--prior", type=Path, action="append", help="dataset to exclude token IDs from")
    args = parser.parse_args()
    priors = args.prior or [DEFAULT_PRIOR / "dataset.json", Path("reference/benchmarks/qwen25_05b/dataset.json")]
    excluded = {token for path in priors for row in json.loads(path.read_text()) for token in row["ctx"]}
    rows = generate(args.seed, args.length, exclude=excluded)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    manifest = {"tag": "empirical", "seed": args.seed, "length": args.length,
                "group_counts": {"train": 2, "validation": 2, "test": 4, "off_domain": 2},
                "dataset_sha256": sha256(args.out),
                "excluded_dataset_sha256": {str(p): sha256(p) for p in priors},
                "exclusion_policy": "all token IDs appearing in prior datasets are removed from generation pool"}
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "positive": sum(r["part"] != "off_domain" for r in rows),
                      "negative": sum(r["part"] == "off_domain" for r in rows),
                      "sha256": sha256(args.out), "manifest": str(manifest_path),
                      "excluded": manifest["excluded_dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
