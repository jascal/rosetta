"""Apply a Base-development-selected copy guard unchanged to matched Base/RL tests."""
import argparse
import json
from pathlib import Path

from benchmark_guarded_copy import evaluate, freeze_domain, run_equiv, sha256, write_guard


def candidate_from_development(report_path):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    protocol = json.loads((report_path.parent / "protocol.json").read_text())
    relations = report["admission"]["relations"]
    candidates = {int(g): (int(k), int(m)) for g, k, m in protocol["guards"]}
    errors = {g: 0 for g in candidates}
    pair_fails = {g: 0 for g in candidates}
    coverage = {g: 0 for g in candidates}
    for g, _ in relations.get("failure", []):
        errors[int(g)] += 1
    for g, _, _ in relations.get("pair_fail", []):
        pair_fails[int(g)] += 1
    for g, n in relations.get("coverage", []):
        coverage[int(g)] = int(n)
    nonempty = [g for g in candidates if coverage[g] > 0]
    # Relaxed discovery rank, using development outputs only. This does NOT
    # replace zero-error Rosetta admission; it exposes the best near-miss.
    chosen = min(nonempty, key=lambda g: (errors[g], pair_fails[g], -coverage[g], *candidates[g]))
    return chosen, candidates[chosen], {
        "development_errors": errors[chosen],
        "development_pair_failures": pair_fails[chosen],
        "development_coverage": coverage[chosen],
        "candidate_count_with_coverage": len(nonempty),
        "rank": "fewest dev errors, fewest causal-pair failures, highest coverage, shortest suffix, lowest support",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_run", type=Path)
    parser.add_argument("rl_run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    base_run, rl_run, out = args.base_run.resolve(), args.rl_run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base_rows = json.loads((base_run / "dataset.json").read_text())
    rl_rows = json.loads((rl_run / "dataset.json").read_text())
    if base_rows != rl_rows:
        raise ValueError("Base and RL datasets differ; transfer requires identical token-ID cases")
    base_refs = {int(k): v for k, v in json.loads((base_run / "references.json").read_text()).items()}
    rl_refs = {int(k): v for k, v in json.loads((rl_run / "references.json").read_text()).items()}
    base_id, (suffix, support), selection = candidate_from_development(base_run / "report.json")
    candidate = write_guard(out / "frozen-base-candidate", [(base_id, suffix, support)])
    protocol = {
        "tag": "empirical",
        "source_candidate": "Base development only; relaxed near-miss rank, not Rosetta admission; post-hoc transfer probe",
        "base_candidate_id": base_id,
        "suffix_length": suffix,
        "minimum_agreeing_sources": support,
        "selection_metrics": selection,
        "dataset_sha256": sha256(base_run / "dataset.json"),
        "candidate_sha256": sha256(candidate),
        "models": {
            "Base": json.loads((base_run / "protocol.json").read_text())["provenance"],
            "RL": json.loads((rl_run / "protocol.json").read_text())["provenance"],
        },
    }
    domain, domain_audit = freeze_domain(out, candidate, [row for row in base_rows if row["part"] == "test"],
                                        {"stage": "freeze Base-derived rule domain before cross-model evaluation"})
    protocol["frozen_test_domain"] = domain
    protocol["domain_audit"] = domain_audit
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = {}
    test = [row for row in base_rows if row["part"] == "test"]
    for name, run, refs in (("Base", base_run, base_refs), ("RL", rl_run, rl_refs)):
        result_dir = out / name.lower()
        result_dir.mkdir()
        evaluated = evaluate(candidate, base_rows, refs, result_dir / "certificate-evidence",
                             {"stage": "frozen Base candidate transfer", "model": name,
                              "dataset_sha256": protocol["dataset_sha256"]})
        firing = [row for row in test if row["id"] in set(domain)]
        cert = run_equiv(candidate, [row["ctx"] for row in firing], [refs[row["id"]] for row in firing],
                         evidence_dir=result_dir / "certificate-evidence",
                         provenance={"stage": "previously frozen matched test; post-hoc transfer exploration", "model": name,
                                     "original_instance_ids": [row["id"] for row in firing]})
        results[name] = {"scores": evaluated["scores"], "test_certificate": cert}
    outcome = {"protocol": protocol, "results": results}
    (out / "result.json").write_text(json.dumps(outcome, indent=2) + "\n")
    lines = ["# Frozen Base-to-RL copy-rule transfer", "",
             "**Empirical, post-hoc finite-domain probe.** This rule was selected from Base development data by a relaxed near-miss ranking. It failed normal Rosetta admission and is not a certified selected circuit. Test references had already been collected and their summary inspected in the preceding experiments, so this is descriptive transfer evidence, not an independent confirmatory holdout.", "",
             f"Frozen rule: suffix length `{suffix}`, minimum agreeing sources `{support}`. Base development: {selection['development_errors']} answer mismatches, {selection['development_pair_failures']} causal-pair failures, coverage {selection['development_coverage']}.", "",
             "| Model | Test answers correct / 144 | Fired | Wrong | Abstained | Exact on frozen firing domain? |",
             "|---|---:|---:|---:|---:|---|"]
    for name, result in results.items():
        score = result["scores"]["test"]
        cert = result["test_certificate"]
        lines.append(f"| {name} | {score['correct']}/144 | {score['answered']} | {score['wrong']} | {score['abstained']} | {cert['certified']} |")
    lines += ["", "The rule and input token IDs are identical across the two evaluations. A clean certificate is scoped to the frozen cases where the rule fires; it does not establish arbitrary-context equivalence."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({name: {"test": result["scores"]["test"], "certified": result["test_certificate"]["certified"]}
                      for name, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
