"""Stress fixed raw-copy and diagnostic guard circuits on held-out MiMo contexts.

The candidate rules and firing domains are frozen before any model references.
This runner reuses the seeded distance/distractor matrix from
benchmark_copy_generalization, but uses a deliberately fixed strict guard when
development admission selected no guard.
"""
import argparse
import json
from pathlib import Path

from benchmark_copy_generalization import DEFAULT_PRIOR, DL, generate, score
from benchmark_guarded_copy import GUARDS, freeze_domain, write_guard
from benchmark_induction import write_copy
from certificate import sha256
from oracle import run_equiv, serve_decide


def run(out, rows, oracle, provenance):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "dataset.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "strata-verifier.dl").write_text((DL / "holdout.dl").read_text() +
                                             (DL / "holdout_strata.dl").read_text())
    artifacts = {
        "strict_guard": write_guard(out / "strict_guard", [GUARDS[-1]]),
        "raw_copy": write_copy(out / "raw_copy", 1),
    }
    protocol = {
        "tag": "empirical",
        "provenance": provenance,
        "dataset_sha256": sha256(out / "dataset.json"),
        "selection": "none; fixed strict diagnostic guard (suffix 9, 3 agreeing sources) and raw-copy circuit",
        "driver_sha256": sha256(__file__),
        "source_sha256": {name: sha256(DL / name) for name in
                          ("copy_core.dl", "copy_guard.dl", "guard_domain.dl", "holdout.dl", "holdout_strata.dl", "equiv.dl")},
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    test = [r for r in rows if r["part"] == "test"]
    frozen = {"protocol_sha256": sha256(out / "protocol.json"),
              "dataset_sha256": sha256(out / "dataset.json"),
              "verifier_sha256": sha256(out / "strata-verifier.dl"), "artifacts": {}}
    for name, path in artifacts.items():
        domain, audit = freeze_domain(out, path, test, provenance)
        audit["evidence"] = str(Path(audit["evidence"]).relative_to(out))
        frozen["artifacts"][name] = {"sha256": sha256(path), "domain": domain, "domain_audit": audit}
    for name, path in artifacts.items():
        frozen["artifacts"][name]["run_sha256"] = sha256(path.parent / "run.dl")
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_hash = sha256(out / "frozen.json")
    print(f"FROZEN: {len(test)} positive cases; {len(rows)-len(test)} negative controls", flush=True)

    refs = {}
    with (out / "oracle-observations.jsonl").open("w") as stream:
        for i, row in enumerate(rows, 1):
            refs[row["id"]] = oracle(row)
            stream.write(json.dumps({"id": row["id"], "reference": refs[row["id"]]}) + "\n")
            stream.flush()
            if i % 24 == 0:
                print(f"oracle references: {i}/{len(rows)}", flush=True)
    if sha256(out / "frozen.json") != frozen_hash:
        raise ValueError("frozen manifest changed during evaluation")
    if sha256(out / "protocol.json") != frozen["protocol_sha256"] or \
       sha256(out / "dataset.json") != frozen["dataset_sha256"] or \
       sha256(out / "strata-verifier.dl") != frozen["verifier_sha256"]:
        raise ValueError("a frozen input changed during evaluation")
    for name, path in artifacts.items():
        if sha256(path) != frozen["artifacts"][name]["sha256"] or \
           sha256(path.parent / "run.dl") != frozen["artifacts"][name]["run_sha256"]:
            raise ValueError(f"frozen circuit changed: {name}")
    (out / "references.json").write_text(json.dumps(refs, indent=2) + "\n")

    results = {}
    for name, path in artifacts.items():
        evaluated = score(out, path, rows, refs, provenance)
        domain = set(frozen["artifacts"][name]["domain"])
        scoped = [r for r in test if r["id"] in domain]
        certificates = {
            scope: run_equiv(path, [r["ctx"] for r in cases], [refs[r["id"]] for r in cases],
                             evidence_dir=out / "certificate-evidence",
                             provenance={**provenance, "scope": scope,
                                         "original_instance_ids": [r["id"] for r in cases]})
            for scope, cases in (("full_test", test), ("frozen_firing_domain", scoped))
        }
        results[name] = {**evaluated, "firing_domain_size": len(scoped),
                         "certificates": certificates, "sha256": sha256(path)}

    def relative(value):
        if isinstance(value, dict):
            return {k: str(Path(v).relative_to(out)) if k == "evidence" else relative(v)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [relative(v) for v in value]
        return value

    report = relative({"tag": "empirical", "provenance": provenance, "results": results,
                       "frozen_sha256": frozen_hash, "references_sha256": sha256(out / "references.json"),
                       "observations_sha256": sha256(out / "oracle-observations.jsonl")})
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# MiMo copy stress: fixed diagnostic circuits", "",
             f"Oracle source: `{provenance['source']}`. **Empirical** measurements.",
             "The strict guard was fixed at suffix length 9 and three agreeing sources. No guard was reselected after seeing test answers.", "",
             "| Circuit | Positive correct / total | Wrong | Abstained | Negative answers | Firing-domain certificate |",
             "|---|---:|---:|---:|---:|---|"]
    for name, result in report["results"].items():
        positive, negative = result["scores"]["test"], result["scores"]["off_domain"]
        lines.append(f"| {name} | {positive['correct']}/{positive['n']} | {positive['wrong']} | "
                     f"{positive['abstained']} | {negative['answered']}/{negative['n']} | "
                     f"{result['certificates']['frozen_firing_domain']['certified']} |")
    lines += ["", "## Strict guard by test stratum", "", "| Factor | Value | Correct / total | Wrong | Abstained |",
              "|---|---|---:|---:|---:|"]
    for factor, values in report["results"]["strict_guard"]["strata"].items():
        for value, result in values.items():
            lines.append(f"| {factor} | {value} | {result['correct']}/{result['n']} | "
                         f"{result['wrong']} | {result['abstained']} |")
    lines += ["", "The test suite contains correlated variants within 18 sequence groups. Certificates are limited to the recorded finite domains.",
              "See `report.json`, `frozen.json`, `references.json`, and certificate evidence for exact results."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--exclude-mi-mo", type=Path, required=True,
                        help="a matched MiMo guarded-copy dataset to keep its token IDs out of this stress set")
    args = parser.parse_args()
    qwen_datasets = [DEFAULT_PRIOR / "dataset.json", Path("reference/benchmarks/qwen25_05b/dataset.json")]
    excluded = {token for path in qwen_datasets for row in json.loads(path.read_text()) for token in row["ctx"]}
    excluded.update(token for row in json.loads(args.exclude_mi_mo.read_text()) for token in row["ctx"])
    rows = generate(exclude=excluded)
    provenance = {"source": "fieldrun", "bundle_sha256": {
        suffix: sha256(str(args.bundle) + suffix) for suffix in (".fieldrun.bin", ".fieldrun.json")},
        "generator_seeds": [2, 3, 4],
        "tokenizer_policy": "same tokenizer as paired MiMo runs; held-out token IDs excluded",
        "excluded_dataset_sha256": {str(path): sha256(path) for path in qwen_datasets} |
                                   {str(args.exclude_mi_mo): sha256(args.exclude_mi_mo)}}
    report = run(args.out, rows, lambda row: serve_decide(args.port, row["ctx"]), provenance)
    print(json.dumps({name: result["scores"]["test"] for name, result in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
