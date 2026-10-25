"""Replay retained discovery queries and check frozen files without model access."""
import argparse
import json
from pathlib import Path

from certificate import replay, sha256

ROOT = Path(__file__).resolve().parents[1]


def audit(out):
    frozen = json.loads((out / "frozen-discovery.json").read_text())
    for relative, digest in frozen["sha256"].items():
        if sha256(out / relative) != digest:
            raise ValueError(f"changed frozen discovery file: {relative}")
    for relative, digest in frozen["sources"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"changed frozen source: {relative}")
    checks = []
    for path in sorted((out / "certificate-evidence").rglob("certificate.json")):
        before = json.loads(path.read_text())
        after = replay(path.parent)
        if bool(after["certified"]) != bool(before["result"]["certified"]):
            raise ValueError(f"verdict changed: {path}")
        if bool(after.get("error")) != bool(before["result"].get("error")):
            raise ValueError(f"process outcome changed: {path}")
        if not after.get("error"):
            recorded = {p.stem: [line.split("\t") for line in p.read_text().splitlines()]
                        for p in (path.parent / "outputs").glob("*.csv")}
            if recorded != after["relations"]:
                raise ValueError(f"relations changed: {path}")
        checks.append({"evidence": str(path.parent.relative_to(out)),
                       "certified": after["certified"], "query_error": bool(after.get("error"))})
    return {"scope": "model-free replay of discovery instrument queries; no equivalence claim",
            "frozen_hashes_unchanged": True, "checks": checks}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    result = audit(a.out.resolve())
    (a.out / "replay-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['checks'])} discovery queries replayed; frozen hashes unchanged")


if __name__ == "__main__":
    main()
