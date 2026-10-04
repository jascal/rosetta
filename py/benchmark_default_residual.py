"""Test a train-selected constant-output residual on frozen contexts."""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

from benchmark_induction import evaluate
from certificate import sha256
from oracle import run_equiv, serve_decide

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "dl"


def write_circuit(directory, token):
    directory.mkdir()
    (directory / "circuits.dl").write_text(
        "// Standalone constant default. Build-time mode selection is recorded separately.\n"
        ".decl default_token(out:number)\n"
        f"default_token({token}).\n"
        ".decl cdecide(inst:number,out:number)\n"
        "cdecide(I,T) :- tok(I,_,_), default_token(T).\n")
    (directory / "run.dl").write_text(
        '.decl tok(inst:number,pos:number,id:number) .input tok\n'
        '#include "circuits.dl"\n.output cdecide\n'
        '.decl abstain(inst:number) .output abstain\n'
        'abstain(I) :- tok(I,_,_), !cdecide(I,_).\n')
    return directory / "circuits.dl"


def facts_for(rows, references):
    return {
        "tok": "".join(f"{row['id']}\t{position}\t{token}\n"
                       for row in rows for position, token in enumerate(row["ctx"])),
        "ref": "".join(f"{row['id']}\t{references[row['id']]}\n"
                       for row in rows if row["id"] in references),
        "sample": "".join(f"{row['id']}\t{row['group']}\t{row['part']}\t"
                          f"{row['part']}\n" for row in rows),
    }


def score(candidate, cases, refs, evidence, provenance):
    result = evaluate(candidate, cases, refs, evidence, provenance)
    return {case: result["scores"].get(case, {}) for case in ("train", "validation", "test", "off_domain")}, result


def benchmark(dataset, out, oracle, bundle_hashes=None, synthetic=False):
    rows = json.loads(dataset.read_text())
    if [row["id"] for row in rows] != list(range(len(rows))):
        raise ValueError("dataset IDs must be contiguous and ordered")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(dataset, out / "dataset.json")
    dataset_hash = sha256(out / "dataset.json")
    development_train = [row for row in rows if row["part"] == "train"]
    refs = {row["id"]: oracle(row) for row in development_train}
    if set(refs) != {row["id"] for row in development_train}:
        raise ValueError("train oracle did not return every required reference")

    mode_dir = out / "mode-selection"
    mode_input, mode_output = mode_dir / "facts", mode_dir / "outputs"
    mode_input.mkdir(parents=True)
    mode_output.mkdir()
    shutil.copyfile(DL / "default_mode_select.dl", mode_dir / "default_mode_select.dl")
    (mode_input / "sample.facts").write_text("".join(f"{r['id']}\ttrain\n" for r in development_train))
    (mode_input / "ref.facts").write_text("".join(f"{r['id']}\t{refs[r['id']]}\n" for r in development_train))
    mode_proc = subprocess.run(["souffle", str(mode_dir / "default_mode_select.dl"),
                                "-F", str(mode_input), "-D", str(mode_output)],
                               capture_output=True, text=True)
    (mode_dir / "stdout.txt").write_text(mode_proc.stdout)
    (mode_dir / "stderr.txt").write_text(mode_proc.stderr)
    if mode_proc.returncode:
        raise RuntimeError(mode_proc.stderr.strip() or "Datalog train mode selection failed")
    modes_path = mode_output / "default_token.csv"
    with modes_path.open(newline="") as handle:
        modes = [int(row[0]) for row in csv.reader(handle, delimiter="\t") if row]
    if len(modes) != 1:
        raise ValueError(f"Datalog selected {len(modes)} default tokens; expected exactly one")
    token = modes[0]
    support_path = mode_output / "selected_support.csv"
    support_rows = list(csv.reader(support_path.open(newline=""), delimiter="\t"))
    if len(support_rows) != 1:
        raise ValueError("Datalog did not emit one selected train support count")
    support = int(support_rows[0][0])

    candidate = write_circuit(out / "candidate", token)
    protocol = {"tag": "empirical", "source": "constant-control" if synthetic else "fieldrun",
                "selection": "most frequent answer on train only; lowest token ID breaks frequency ties",
                "bundle_sha256": bundle_hashes, "dataset_sha256": dataset_hash,
                "mode_program_sha256": sha256(mode_dir / "default_mode_select.dl"),
                "driver_sha256": sha256(__file__), "training_references_sha256": sha256(mode_input / "ref.facts"),
                "runtime": "standalone Datalog; no model, Python, or fieldrun dependency"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    # Freeze the candidate and full test firing domain before querying validation,
    # test, or off-domain oracle answers.
    test = [row for row in rows if row["part"] == "test"]
    from benchmark_guarded_copy import freeze_domain
    domain, domain_audit = freeze_domain(out, candidate, test, protocol)
    expected_domain = {row["id"] for row in test}
    if set(domain) != expected_domain:
        raise ValueError(f"constant circuit did not cover the full test set: {len(domain)}/{len(test)}")
    frozen = {"dataset_sha256": dataset_hash, "candidate_sha256": sha256(candidate),
              "selected_token": token, "train_support": support, "test_domain": domain,
              "test_domain_audit": domain_audit,
              "frozen_before_validation_test_and_off_domain_references": True}
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")

    for row in rows:
        if row["part"] != "train":
            refs[row["id"]] = oracle(row)
    if set(refs) != {row["id"] for row in rows}:
        raise ValueError("oracle references do not cover the full frozen dataset")
    (out / "references.json").write_text(json.dumps(refs, indent=2) + "\n")

    evidence = out / "certificate-evidence"
    eval_results = {}
    for part in ("train", "validation", "test", "off_domain"):
        cases = [row for row in rows if row["part"] == part]
        if not cases:
            continue
        eval_provenance = {**protocol, "stage": f"{part} exact evaluation"}
        evaluated = evaluate(candidate, cases, refs, evidence, eval_provenance)
        certificate = run_equiv(candidate, [row["ctx"] for row in cases],
                                 [refs[row["id"]] for row in cases], evidence_dir=evidence,
                                 provenance={**protocol, "stage": f"{part} equivalence certificate",
                                             "original_instance_ids": [row["id"] for row in cases]})
        eval_results[part] = {"score": evaluated["scores"][part], "certificate": certificate}

    report = {"tag": "empirical", "provenance": protocol, "dataset_sha256": dataset_hash,
              "references_sha256": sha256(out / "references.json"), "selected_token": token,
              "train_support": support, "selected_token_mode_query": {
                  "train_count": list(csv.reader((mode_output / "train_support.csv").open(), delimiter="\t")),
                  "selected_count": support},
              "results": eval_results, "scope": "constant-output residual hypothesis on a finite random-token corpus"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_token": token, "train_support": support,
                      "scores": {part: row["score"] for part, row in eval_results.items()},
                      "certificates": {part: row["certificate"]["certified"]
                                       for part, row in eval_results.items()}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--bundle")
    parser.add_argument("--port", type=int)
    parser.add_argument("--control", action="store_true",
                        help="use a fixed authored output token to verify mode selection and certification")
    parser.add_argument("--control-token", type=int, default=50000)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if args.control:
        oracle = lambda _row: args.control_token
        bundle_hashes = None
    else:
        if not args.bundle or not args.port:
            parser.error("fieldrun runs require --bundle and --port")
        bundle_hashes = {suffix: sha256(args.bundle + suffix) for suffix in (".fieldrun.bin", ".fieldrun.json")}
        oracle = lambda row: serve_decide(args.port, row["ctx"])
    benchmark(dataset, args.out.resolve(), oracle, bundle_hashes, args.control)


if __name__ == "__main__":
    main()
