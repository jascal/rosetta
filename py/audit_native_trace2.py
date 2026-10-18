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
        mismatched = []
        for name, rows in relations.items():
            recorded = _norm_csv(directory, name)
            if recorded is not None and _norm_relation(rows) != recorded:
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
    for arm in ARMS:
        directory = study / "artifacts" / arm
        if not (directory / "circuits.dl").exists():
            results.append({"arm": arm, "present": False})
            continue
        with tempfile.TemporaryDirectory() as scratch:
            work = Path(scratch)
            for name in ("circuits.dl", "run.dl"):
                (work / name).write_text((directory / name).read_text())
            ids = range(8)
            (work / "tok.facts").write_text("".join(
                f"{i}\t0\t{100+i}\n{i}\t1\t7\n" for i in ids))
            (work / "domain.facts").write_text("".join(f"{i}\n" for i in ids))
            (work / "outputs").mkdir()
            proc = subprocess.run(
                ["souffle", "run.dl", "-I", ".", "-F", ".", "-D", "outputs"],
                cwd=work, capture_output=True, text=True)
            if proc.returncode:
                results.append({"arm": arm, "present": True, "ran": False,
                                "stderr": proc.stderr.strip()[:400]})
                continue
            abstain = (work / "outputs" / "abstain.csv").read_text().split()
            results.append({
                "arm": arm, "present": True, "ran": True,
                "only_program_and_token_domain_facts": True,
                "all_domain_instances_abstain": sorted(abstain, key=int) == [str(i) for i in ids],
            })
    return results


def audit_freeze(study):
    frozen_path = study / "frozen.json"
    if not frozen_path.exists():
        return False, {"frozen": "absent"}
    frozen = _load(frozen_path)
    changed = []
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
    report = {"evidence": evidence, "standalone": standalone,
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
    }
    print(json.dumps(summary, indent=2))
    return 0 if not failed and unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())