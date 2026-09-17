"""Evaluate previously frozen copy circuits on new seeds, lengths and distractors. No selection or retraining."""
import argparse
import json
from pathlib import Path
import random
import shutil

from benchmark_guarded_copy import freeze_domain
from benchmark_induction import DL, digest, facts
from certificate import check, sha256
from oracle import run_equiv, serve_decide

DEFAULT_PRIOR = Path("reference/benchmarks/qwen25_05b_guarded_seed1")
LAYOUTS = ("plain", "noise8", "noise32", "near_match8")


def generate(seeds=(2, 3, 4), lengths=(8, 12, 24), groups=2, exclude=()):
    """Generate paired interventions with group-disjoint vocabulary and independent negative groups.

    Near matches repeat only the last TWO query tokens followed by a decoy, testing the fixed three-token guard.
    Noise blocks differ between source occurrences. All changed source positions are retained for inspection.
    """
    if not seeds or len(set(seeds)) != len(seeds) or not lengths or min(lengths) < 8 or groups < 1:
        raise ValueError("need unique seeds, lengths >=8, and positive groups")
    used = set(exclude)
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        for length in lengths:
            for part in ("test", "off_domain"):
                for group in range(groups):
                    values = rng.sample([t for t in range(200, 40000) if t not in used], length + 98)
                    used.update(values)
                    seq = values[:length]
                    replacement, decoy = values[length:length + 2]
                    noise = [values[length + 2 + 32 * b:length + 2 + 32 * (b + 1)] for b in range(3)]
                    prefix = length // 2
                    meta = dict(seed=seed, length=length, prefix=prefix, part=part,
                                group=f"seed{seed}-length{length}-{part}-{group}")
                    if part == "off_domain":
                        conflict = seq * 2 + seq[:prefix]
                        conflict[prefix] = replacement
                        controls = [("no_repeat", seq), ("one_source", seq + noise[0][:8] + seq[:prefix]),
                                    ("conflicting_sources", conflict)]
                        for kind, ctx in controls:
                            rows.append(dict(meta, id=len(rows), kind=kind, layout="negative", exposures=0,
                                             ctx=ctx, expected=seq[prefix]))
                        continue
                    for exposure in (2, 3):
                        for layout in LAYOUTS:
                            context, positions = [], []
                            for b in range(exposure):
                                positions.append(len(context) + prefix)
                                context.extend(seq)
                                gap = [] if layout == "plain" else noise[b][:32 if layout == "noise32" else 8]
                                if layout == "near_match8":
                                    gap[-3:] = [seq[prefix - 2], seq[prefix - 1], decoy]
                                context.extend(gap)
                            context.extend(seq[:prefix])
                            base = len(rows)
                            variant = dict(meta, layout=layout, exposures=exposure)
                            rows.append(dict(variant, id=base, kind="repeat", ctx=context, expected=seq[prefix]))
                            edited = context[:]
                            for p in positions:
                                edited[p] = replacement
                            rows.append(dict(variant, id=len(rows), kind="intervention", base=base, positions=positions,
                                             ctx=edited, expected=replacement))
    return rows


def summarize(counts):
    n, answered, correct, wrong, abstained = map(int, counts)
    return dict(n=n, answered=answered, correct=correct, wrong=wrong, abstained=abstained,
                coverage=answered / n if n else None, precision=correct / answered if answered else None)


def score(out, candidate, rows, refs, provenance, *, factors=None, extra_facts=None):
    staged = facts(rows, refs)
    staged["sample"] = "".join(f"{r['id']}\t{r['group']}\t{r['part']}\t{digest(r['ctx'])}\n" for r in rows)
    buckets = []
    for row in rows:
        fields = (factors or ("seed", "length", "layout", "exposures", "kind")) if row["part"] == "test" else ()
        buckets.extend((row["id"], field, row[field]) for field in fields)
        if row["part"] == "off_domain":
            buckets.append((row["id"], "negative", row["kind"]))
    staged["bucket"] = "".join(f"{i}\t{factor}\t{value}\n" for i, factor, value in buckets)
    if extra_facts:
        staged.update(extra_facts)
    audit = check(out / "strata-verifier.dl", candidate, staged, {"ninvalid": int, "nleak": int},
                  evidence_dir=out / "certificate-evidence", provenance=provenance)
    if not audit["certified"]:
        raise ValueError(f"evaluation audit failed: {audit}")
    relations = audit.pop("relations")
    strata = {}
    for factor, value, *counts in relations["breakdown"]:
        strata.setdefault(factor, {})[value] = summarize(counts)
    by_id = {r["id"]: r for r in rows}
    errors = []
    for i, reference, prediction in relations["counterexample"]:
        row = by_id[int(i)]
        errors.append({key: row[key] for key in ("id", "part", "group", "seed", "length", "layout", "kind", "exposures")} |
                      {"reference": int(reference), "prediction": int(prediction)})
    result = {"audit": audit, "scores": {part: summarize(counts) for part, *counts in relations["score"]},
              "strata": strata, "counterexamples": errors}
    if "paired_count" in relations:
        result["paired_count"] = relations["paired_count"]
    return result


def verify_frozen(out, frozen):
    for filename, expected in frozen["files"].items():
        if sha256(out / filename) != expected:
            raise ValueError(f"frozen input changed: {filename}")


def benchmark(out, rows, oracle, provenance, prior=DEFAULT_PRIOR):
    out, prior = Path(out), Path(prior)
    out.mkdir(parents=True, exist_ok=False)
    previous = json.loads((prior / "frozen.json").read_text())
    artifacts = {}
    for name, source in (("guarded", "selected"), ("raw_copy", "raw-copy")):
        prior_key = "selected" if name == "guarded" else "raw_copy"
        if sha256(prior / source / "circuits.dl") != previous["artifacts"][prior_key]["sha256"]:
            raise ValueError("prior frozen circuit changed")
        directory = out / name
        directory.mkdir()
        for filename in ("circuits.dl", "run.dl"):
            shutil.copyfile(prior / source / filename, directory / filename)
        artifacts[name] = directory / "circuits.dl"
    (out / "dataset.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "strata-verifier.dl").write_text((DL / "holdout.dl").read_text() + (DL / "holdout_strata.dl").read_text())
    protocol = {"provenance": provenance, "selection": "none; both circuits copied byte-for-byte from prior frozen artifacts",
                "prior_frozen_sha256": sha256(prior / "frozen.json"), "driver_sha256": sha256(__file__),
                "seeds": sorted({r["seed"] for r in rows}), "lengths": sorted({r["length"] for r in rows}),
                "layouts": LAYOUTS, "test_groups": len({r["group"] for r in rows if r["part"] == "test"}),
                "scope": "all preregistered test cases and each reference-independent firing domain"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    test = [r for r in rows if r["part"] == "test"]
    frozen = {"files": {}, "artifacts": {}}
    for name, candidate in artifacts.items():
        domain, audit = freeze_domain(out, candidate, test, provenance)
        audit["evidence"] = str(Path(audit["evidence"]).relative_to(out))
        frozen["artifacts"][name] = {"domain": domain, "domain_audit": audit}
    for filename in ("dataset.json", "protocol.json", "strata-verifier.dl",
                     "guarded/circuits.dl", "guarded/run.dl", "raw_copy/circuits.dl", "raw_copy/run.dl"):
        frozen["files"][filename] = sha256(out / filename)
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_hash = sha256(out / "frozen.json")
    print(f"FROZEN: {len(test)} test cases, {len(rows)-len(test)} negatives; no circuit selection", flush=True)
    refs = {}
    with (out / "oracle-observations.jsonl").open("w") as observations:
        for n, row in enumerate(rows, 1):
            reference = oracle(row)
            refs[row["id"]] = reference
            observations.write(json.dumps({"id": row["id"], "reference": reference}) + "\n")
            observations.flush()
            if n % 24 == 0:
                print(f"oracle references: {n}/{len(rows)}", flush=True)
    verify_frozen(out, frozen)
    if sha256(out / "frozen.json") != frozen_hash:
        raise ValueError("freeze manifest changed during evaluation")
    (out / "references.json").write_text(json.dumps(refs, indent=2) + "\n")
    results = {}
    for name, candidate in artifacts.items():
        evaluated = score(out, candidate, rows, refs, provenance)
        domain = set(frozen["artifacts"][name]["domain"])
        scoped = [r for r in test if r["id"] in domain]
        certs = {}
        for scope, cases in (("full_test", test), ("frozen_firing_domain", scoped)):
            certs[scope] = run_equiv(candidate, [r["ctx"] for r in cases], [refs[r["id"]] for r in cases],
                                    evidence_dir=out / "certificate-evidence",
                                    provenance={**provenance, "scope": scope, "original_instance_ids": [r["id"] for r in cases]})
        results[name] = {**evaluated, "certificates": certs, "firing_domain_size": len(scoped), "sha256": sha256(candidate)}

    def relative(value):
        if isinstance(value, dict):
            return {k: str(Path(v).relative_to(out)) if k == "evidence" else relative(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relative(v) for v in value]
        return value

    report = relative({"tag": "empirical", "provenance": provenance, "results": results,
                       "frozen_sha256": frozen_hash, "references_sha256": sha256(out / "references.json"),
                       "observations_sha256": sha256(out / "oracle-observations.jsonl")})
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Fixed copy guard: generalization test", "", f"Oracle: `{provenance['source']}`. **Empirical** measurements.",
             "Circuits were copied unchanged from the prior experiment. Dataset and firing domains were frozen before all oracle queries.", "",
             "| Circuit | Correct / all test | Wrong | Abstained | Negative answers | Exact firing-domain certificate |",
             "|---|---:|---:|---:|---:|---|"]
    for name, result in report["results"].items():
        s, neg = result["scores"]["test"], result["scores"]["off_domain"]
        lines.append(f"| {name} | {s['correct']}/{s['n']} | {s['wrong']} | {s['abstained']} | "
                     f"{neg['answered']}/{neg['n']} | {result['certificates']['frozen_firing_domain']['certified']} |")
    lines += ["", "## Guarded circuit by predeclared test condition", "",
              "| Factor | Value | Correct / total | Wrong | Abstained |", "|---|---|---:|---:|---:|"]
    for factor, strata in report["results"]["guarded"]["strata"].items():
        for value, s in strata.items():
            lines.append(f"| {factor} | {value} | {s['correct']}/{s['n']} | {s['wrong']} | {s['abstained']} |")
    lines += ["", "Exact certificates apply only to their stated finite domain, relative to recorded references. A failure is retained",
              "as a counterexample; no successful subset is certified after observing references. Strata are descriptive measurements,",
              "not newly selected rules. Correlated variants share sequence groups. See `report.json` for raw counters and evidence."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--oracle", choices=("copy-control", "constant-control", "fieldrun"), default="copy-control")
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--port", type=int)
    parser.add_argument("--bundle")
    args = parser.parse_args()
    prior_data = [args.prior / "dataset.json", Path("reference/benchmarks/qwen25_05b/dataset.json")]
    excluded = {t for path in prior_data for row in json.loads(path.read_text()) for t in row["ctx"]}
    rows = generate(exclude=excluded)
    provenance = {"source": args.oracle, "synthetic": args.oracle != "fieldrun",
                  "excluded_dataset_sha256": {str(p): sha256(p) for p in prior_data}}
    if args.oracle == "fieldrun":
        if not args.port or not args.bundle:
            parser.error("fieldrun needs --port and --bundle")
        provenance["bundle_sha256"] = {s: sha256(args.bundle + s) for s in (".fieldrun.bin", ".fieldrun.json")}
        prior_report = json.loads((args.prior / "report.json").read_text())
        if provenance["bundle_sha256"] != prior_report["provenance"]["bundle_sha256"]:
            parser.error("fixed-guard replication must use the same recorded model bundle")

    def oracle(row):
        if args.oracle == "fieldrun":
            return serve_decide(args.port, row["ctx"])
        return row["expected"] if args.oracle == "copy-control" else 0

    report = benchmark(args.out, rows, oracle, provenance, args.prior)
    print(json.dumps({k: v["scores"] for k, v in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
