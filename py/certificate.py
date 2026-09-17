"""Certificate I/O: isolate Souffle, retain its verdict and replayable, hashed evidence.

No equivalence or distributional decisions live here; the verifier's certified relation is the verdict.
Replay: python3 py/certificate.py <evidence-directory>
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(verifier, candidate, facts, scalars, *, evidence_dir=None, provenance=None):
    """Stage complete inputs, run a self-contained checker, and read required output relations.

    facts maps relation names to serialized TSV. An absent/failed checker is an error, never a zero counter.
    """
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "facts").mkdir()
        (root / "outputs").mkdir()
        shutil.copyfile(verifier, root / "verify.dl")
        shutil.copyfile(candidate, root / "circuit.dl")
        for name, contents in facts.items():
            (root / "facts" / f"{name}.facts").write_text(contents)
        command = ["souffle", "verify.dl", "-F", "facts", "-D", "outputs", "-I", "."]
        proc = subprocess.run(command, cwd=root, capture_output=True, text=True)
        (root / "stderr.txt").write_text(proc.stderr)
        result = {"certified": False}
        if proc.returncode:
            result["error"] = proc.stderr.strip() or f"souffle exited {proc.returncode}"
        else:
            try:
                for name, cast in scalars.items():
                    result[name] = cast((root / "outputs" / f"{name}.csv").read_text().strip())
                verdict = (root / "outputs" / "certified.csv").read_text().strip()
                if verdict not in ("", "()"):
                    raise ValueError(f"unexpected certified relation: {verdict!r}")
                result["certified"] = verdict == "()"
                result["relations"] = {
                    p.stem: [line.split("\t") for line in p.read_text().splitlines()]
                    for p in (root / "outputs").glob("*.csv")
                }
            except (OSError, ValueError) as exc:
                result = {"certified": False, "error": f"incomplete verifier output: {exc}"}
        if evidence_dir is not None:
            Path(evidence_dir).mkdir(parents=True, exist_ok=True)
            dest = Path(tempfile.mkdtemp(prefix="check-", dir=evidence_dir))
            shutil.copytree(root, dest, dirs_exist_ok=True)
            metadata = {
                "schema_version": 1,
                "command": command,
                "souffle_version": subprocess.run(["souffle", "--version"], capture_output=True, text=True).stdout.strip(),
                "provenance": provenance or {"source": "caller-supplied references"},
                "scalars": {name: cast.__name__ for name, cast in scalars.items()},
                "result": {k: v for k, v in result.items() if k != "relations"},
                "sha256": {str(p.relative_to(dest)): sha256(p) for p in sorted(dest.rglob("*")) if p.is_file()},
            }
            (dest / "certificate.json").write_text(json.dumps(metadata, indent=2) + "\n")
            result["evidence"] = str(dest)
        return result


def replay(directory):
    """Check evidence integrity and rerun the recorded Datalog over exactly the recorded facts."""
    root = Path(directory)
    metadata = json.loads((root / "certificate.json").read_text())
    for relative, expected in metadata["sha256"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"certificate evidence changed: {relative}")
    casts = {"int": int, "float": float}
    return check(root / "verify.dl", root / "circuit.dl",
                 {p.stem: p.read_text() for p in (root / "facts").glob("*.facts")},
                 {k: casts[v] for k, v in metadata["scalars"].items()})


if __name__ == "__main__":
    import sys
    result = replay(sys.argv[1])
    print(json.dumps({k: v for k, v in result.items() if k != "relations"}, indent=2))
    sys.exit(0 if result["certified"] else 1)
