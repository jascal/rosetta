"""Rebuild a train-only lexical baseline with the current emitter against already-recorded references.

This is a post-test implementation correction, NOT a fresh holdout experiment. Original frozen artifacts are retained.
"""
import argparse
import json
from pathlib import Path

from benchmark_induction import evaluate, learn_baseline
from certificate import sha256
from oracle import run_equiv


def recheck(run):
    run = Path(run)
    out = run / "baseline-correction"
    out.mkdir(exist_ok=False)
    rows = json.loads((run / "dataset.json").read_text())
    refs = {int(k): v for k, v in json.loads((run / "references.json").read_text()).items()}
    prior = json.loads((run / "report.json").read_text())
    if sha256(run / "references.json") != prior["references_sha256"]:
        raise ValueError("recorded references changed")
    provenance = {"source": "post-test baseline emitter correction; cached references", "original_run": run.name,
                  "dataset_sha256": sha256(run / "dataset.json"), "references_sha256": sha256(run / "references.json"),
                  "emitter_sha256": sha256(Path(__file__).with_name("minimize.py"))}
    # Only train facts enter the learner, even though final references are already known.
    train = [r for r in rows if r["part"] == "train"]
    candidate, nrules, learning = learn_baseline(train, {r["id"]: refs[r["id"]] for r in train}, out)
    evaluated = evaluate(candidate, rows, refs, out / "certificate-evidence", provenance)
    certs = {}
    for part in ("train", "test"):
        cases = [r for r in rows if r["part"] == part]
        certs[part] = run_equiv(candidate, [r["ctx"] for r in cases], [refs[r["id"]] for r in cases],
                                evidence_dir=out / "certificate-evidence", provenance={**provenance, "part": part})
    report = {"provenance": provenance, "baseline_rules": nrules, "baseline_learning": learning,
              **evaluated, "certificates": certs, "artifact_sha256": sha256(candidate)}

    def relative(value):
        if isinstance(value, dict):
            return {k: str(Path(v).relative_to(out)) if k == "evidence" else relative(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relative(v) for v in value]
        return value

    report = relative(report)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Lexical baseline emitter correction\n\n"
        "Post-test implementation correction using recorded references. This is not a fresh holdout experiment.\n"
        "The original frozen artifacts and guard certificates are unchanged. The emitter now binds each instance\n"
        "before computing its final token position, so mixed-length batches retain shorter contexts.\n\n"
        f"Train: {evaluated['scores']['train']['correct']}/{evaluated['scores']['train']['n']} correct; "
        f"exact certificate: {certs['train']['certified']}.\n"
        f"Test: {evaluated['scores']['test']['answered']}/{evaluated['scores']['test']['n']} answered. "
        "See `report.json` for all counts and retained evidence.\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(recheck(args.run)["scores"], indent=2))
