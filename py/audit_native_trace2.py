"""Model-free replay audit for a run-B study directory.

Run A's audit only ever existed as an ad-hoc script in /tmp, so the claim
"everything replays model-free" was not reproducible from the repository. This
is the committed version, and it checks three things:

1. every retained certificate re-runs its recorded Datalog over exactly its
   recorded facts, reproducing the stored relations and the stored verdict —
   including the FAILED equivalence checks, which must keep failing;
2. every frozen file and frozen source still hashes to what the manifest says;
3. each arm's emitted artifact runs in Souffle alone, from nothing but its own
   circuits.dl, its run.dl wrapper, and token/domain facts.

No model, no weights, no network, no fieldrun.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from certificate import replay, sha256

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("behavior", "trace", "shuffled")


def _load(path):
    return json.loads(path.read_text())


def _norm_relation(rows):
    return sorted(tuple(str(field) for field in row) for row in rows)


def _norm_csv(directory, name):
    path = Path(directory) / "outputs" / f"{name}.csv"
    if not path.exists():
        return None
    return sorted(tuple(line.split("\t")) for line in path.read_text().splitlines() if line)


def audit_evidence(study):
    """Re-run each retained certificate and compare against its stored outputs.

    A FAILED equivalence check must reproduce as failed. Treating a failure as a
    replay error would be the easiest way to make this audit lie, so the stored
    verdict is compared explicitly and the relations are compared regardless.
    """
    results = []
    for directory in sorted(p for p in (study / "certificate-evidence").glob("*/*") if p.is_dir()):
        entry = {"evidence": str(directory.relative_to(study))}
        try:
            stored = _load(directory / "certificate.json")
            expected_certified = bool(stored.get("result", {}).get("certified"))
            result = replay(directory)
        except Exception as error:            # integrity failure is a RESULT, not a crash
            entry.update(certified=None, verdict_preserved=False,
                         all_relations_identical=False, mismatched_relations=[],
                         error=f"{type(error).__name__}: {error}")
            results.append(entry)
            continue
        relations = result.get("relations", {})
        recorded_names = {p.stem for p in (directory / "outputs").glob("*.csv")}
        mismatched = list(recorded_names ^ set(relations))
        for name, rows in relations.items():
            recorded = _norm_csv(directory, name)
            if recorded is None or _norm_relation(rows) != recorded:
                mismatched.append(name)
        entry.update(certified=bool(result["certified"]),
                     verdict_preserved=bool(result["certified"]) == expected_certified,
                     all_relations_identical=not mismatched,
                     mismatched_relations=mismatched)
        results.append(entry)
    return results


def audit_standalone(study):
    """Each artifact in Souffle alone: its program, its wrapper, and facts."""
    results = []
    frozen_path = study / "frozen.json"
    if not frozen_path.exists():
        return [{"arm":arm,"present":False,"ran":False} for arm in ARMS]
    frozen = _load(frozen_path)
    if frozen.get("status") == "degenerate":
        return []  # this registered terminal outcome has no runtime artifacts
    rows = [r for r in _load(study / "dataset.json") if r["part"] == "test"]
    ids = {str(r["id"]) for r in rows}
    for arm in ARMS:
        directory = study / "artifacts" / arm
        if not all((directory / name).exists() for name in ("circuits.dl","run.dl")):
            results.append({"arm":arm,"present":False,"ran":False})
            continue
        with tempfile.TemporaryDirectory() as scratch:
            work = Path(scratch)
            for name in ("circuits.dl", "run.dl"):
                (work / name).write_text((directory / name).read_text())
            (work / "tok.facts").write_text("".join(
                f"{r['id']}\t{pos}\t{token}\n" for r in rows for pos,token in enumerate(r["ctx"])))
            (work / "domain.facts").write_text("".join(f"{i}\n" for i in sorted(ids)))
            (work / "outputs").mkdir()
            proc = subprocess.run(["souffle","run.dl","-I",".","-F",".","-D","outputs"],
                                  cwd=work,capture_output=True,text=True)
            if proc.returncode:
                results.append({"arm":arm,"present":True,"ran":False,"stderr":proc.stderr})
                continue
            predictions = [line.split("\t") for line in (work / "outputs/cdecide.csv").read_text().splitlines()]
            expected = frozen["artifacts"][arm]["predictions"]
            abstain = set((work / "outputs/abstain.csv").read_text().split())
            results.append({"arm":arm,"present":True,"ran":True,
                "only_program_and_token_domain_facts":True,
                "predictions_match":_norm_relation(predictions)==_norm_relation(expected),
                "abstentions_match":abstain == ids-{str(i) for i,_ in expected},
                "all_domain_instances_abstain":abstain == ids})
    return results


def audit_freeze(study):
    frozen_path = study / "frozen.json"
    if not frozen_path.exists():
        return False, {"frozen": "absent"}
    frozen = _load(frozen_path)
    changed = []
    if not frozen.get("sha256") or not frozen.get("source_sha256"):
        changed.append("empty freeze manifest")
    for name, expected in frozen.get("sha256", {}).items():
        path = study / name
        if not path.is_file() or sha256(path) != expected:
            changed.append(f"file:{name}")
    for name, expected in frozen.get("source_sha256", {}).items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            changed.append(f"source:{name}")
    return not changed, {"checked_files": len(frozen.get("sha256", {})),
                          "checked_sources": len(frozen.get("source_sha256", {})),
                          "changed": changed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--write", action="store_true", help="write replay-audit.json into the study")
    a = p.parse_args()
    study = a.study.resolve()

    evidence = audit_evidence(study)
    standalone = audit_standalone(study)
    unchanged, detail = audit_freeze(study)
    missing = []
    def references(value):
        if isinstance(value, dict):
            if "evidence" in value and isinstance(value["evidence"], str):
                if not (study / value["evidence"] / "certificate.json").exists():
                    missing.append(value["evidence"])
            for item in value.values(): references(item)
        elif isinstance(value, list):
            for item in value: references(item)
    for path in study.glob("*.json"):
        if path.name != "replay-audit.json": references(_load(path))
    final_report = _load(study / "report.json") if (study / "report.json").exists() else {}
    frozen = _load(study / "frozen.json") if (study / "frozen.json").exists() else {}
    degenerate = frozen.get("status") == "degenerate" and final_report.get("status") == "degenerate"
    completed = degenerate or bool(final_report.get("certificates"))
    runtime_ok = degenerate or (len(standalone)==3 and all(
        r.get("ran") and r.get("predictions_match") and r.get("abstentions_match") for r in standalone))
    report = {"missing_evidence":missing, "completed":completed, "runtime_ok":runtime_ok,
              "evidence": evidence, "standalone": standalone,
              "frozen_hashes_unchanged": unchanged, "frozen_detail": detail}

    failed = [e for e in evidence
              if not e["verdict_preserved"] or not e["all_relations_identical"]]
    if a.write:
        (study / "replay-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "checks": len(evidence),
        "verdicts_preserved": len(evidence) - len(failed),
        "standalone_arms": sum(1 for s in standalone if s.get("ran")),
        "frozen_hashes_unchanged": unchanged,
        "mismatches": [e["evidence"] for e in failed],
        "missing_evidence":missing, "runtime_ok":runtime_ok, "completed":completed,
    }
    print(json.dumps(summary, indent=2))
    return 0 if evidence and not failed and unchanged and not missing and completed and runtime_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())