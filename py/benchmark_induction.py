"""Grouped train/validation/test copy benchmark. Python generates/stages data; Datalog decides.

Synthetic controls test the instrument, not an LLM. --port uses a resident fieldrun model as a build-time oracle.
No test reference is requested until admission, baseline learning, and artifact hashes have been frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from certificate import check, sha256
from minimize import emit
from oracle import run_equiv, serve_decide

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "dl"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate(seed=0, length=12, groups=(4, 2, 4), vocab=(200, 40000)):
    """Nonce sequences; entire sequences and interventions stay together. Vocabulary is disjoint across groups.

    Targets come from the generation recipe, independently of the copy detector. Positions are fixed before oracle calls.
    """
    if length < 6 or any(n < 1 for n in groups):
        raise ValueError("need length >= 6 and nonempty train/validation/test groups")
    positions = sorted({length // 4, length // 2, 3 * length // 4})
    per_group = length + len(positions)
    tokens = iter(random.Random(seed).sample(range(*vocab), (sum(groups) + groups[2]) * per_group))
    rows = []
    for part, count in zip(("train", "validation", "test", "off_domain"), (*groups, groups[2])):
        for g in range(count):
            values = [next(tokens) for _ in range(per_group)]
            seq = values[:length]
            group = f"{part}-{g}"
            if part == "off_domain":
                rows.append(dict(id=len(rows), group=group, part=part, kind="no_repeat", ctx=seq, expected=seq[0]))
                continue
            for j, position in enumerate(positions):
                ctx = seq + seq[:position]
                base = len(rows)
                rows.append(dict(id=base, group=group, part=part, kind="repeat", ctx=ctx, expected=seq[position]))
                changed = ctx[:]
                changed[position] = values[length + j]
                rows.append(dict(id=len(rows), group=group, part=part, kind="intervention", ctx=changed,
                                 expected=values[length + j], base=base, position=position))
    return rows


def facts(rows, refs):
    return {
        "tok": "".join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p, t in enumerate(r["ctx"])),
        "ref": "".join(f"{r['id']}\t{refs[r['id']]}\n" for r in rows if refs.get(r["id"]) is not None),
    }


def evaluate(candidate, rows, refs, evidence_dir, provenance):
    staged = facts(rows, refs)
    staged["sample"] = "".join(f"{r['id']}\t{r['group']}\t{r['part']}\t{digest(r['ctx'])}\n" for r in rows)
    result = check(DL / "holdout.dl", candidate, staged, {"ninvalid": int, "nleak": int},
                   evidence_dir=evidence_dir, provenance=provenance)
    if not result["certified"]:
        raise ValueError(f"invalid/leaking evaluation inputs: {result}")
    scores = {}
    for part, *counts in result.pop("relations")["score"]:
        n, answered, correct, wrong, abstained = map(int, counts)
        scores[part] = dict(n=n, answered=answered, correct=correct, wrong=wrong, abstained=abstained,
                            coverage=answered / n, precision=correct / answered if answered else None,
                            fidelity=correct / n, confident_wrong=wrong / n, abstain=abstained / n)
    return {"audit": result, "scores": scores}


def learn_baseline(rows, refs, out):
    train = [r for r in rows if r["part"] == "train"]
    staged = facts(train, refs)
    staged["ctx"] = staged.pop("tok")
    staged["domain"] = "".join(f"{r['id']}\n" for r in train)
    staged["wmax"] = f"{max(len(r['ctx']) for r in train)}\n"
    learned = check(DL / "ngram_train.dl", DL / "ngram.dl", staged, {"ntrain": int},
                    evidence_dir=out / "certificate-evidence", provenance={"stage": "train-only baseline"})
    if not learned["certified"]:
        raise ValueError(f"baseline learning failed: {learned}")
    by_id = {r["id"]: r for r in train}
    # Project Datalog's chosen suffixes into literal rule facts. No host-side selection.
    rules = {tuple(by_id[int(i)]["ctx"][-int(k):]): refs[int(i)] for i, k in learned["relations"]["minorder"]}
    directory = out / "baseline"
    directory.mkdir()
    emit(str(directory / "circuits.dl"), rules, False, {}, "train-only n-gram baseline")
    learned.pop("relations")
    return directory / "circuits.dl", len(rules), learned


def write_copy(directory, length):
    directory.mkdir()
    program = ("// Standalone copy candidate; no learned-token lookup or model fallback.\n"
               ".decl copy_length(k:number)\n" + f"copy_length({length}).\n" +
               (DL / "copy_core.dl").read_text() +
               "\n.decl cdecide(inst:number,out:number)\ncdecide(I,T) :- copy_prediction(_,I,T).\n")
    (directory / "circuits.dl").write_text(program)
    (directory / "run.dl").write_text(
        '.decl tok(inst:number,pos:number,id:number)\n.input tok\n#include "circuits.dl"\n.output cdecide\n'
        '.decl abstain(inst:number)\n.output abstain\nabstain(I) :- tok(I,_,_), !cdecide(I,_).\n')
    return directory / "circuits.dl"


def benchmark(out, rows, oracle, provenance):
    """Freeze all candidates before asking the oracle for ANY final-test or off-domain reference."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "dataset.json").write_text(json.dumps(rows, indent=2) + "\n")
    development = [r for r in rows if r["part"] in ("train", "validation")]
    refs = {r["id"]: oracle(r) for r in development}
    staged = facts(development, refs)
    staged["sample"] = "".join(f"{r['id']}\t{r['part']}\n" for r in development)
    staged["intervention"] = "".join(
        f"{r['base']}\t{r['id']}\t{r['position']}\t{r['ctx'][r['position']]}\n"
        for r in development if r["kind"] == "intervention")
    admission = check(DL / "copy_select.dl", DL / "copy_core.dl", staged, {"nselected": int},
                      evidence_dir=out / "certificate-evidence", provenance={**provenance, "stage": "train+validation only"})
    if not admission["certified"]:
        raise ValueError(f"admission input failure: {admission}")
    selected = [int(row[0]) for row in admission["relations"]["selected"]]
    admission.pop("relations")
    baseline, baseline_rules, learning = learn_baseline(rows, refs, out)
    # L=1 is a preregistered hypothesis even if admission rejects it; reporting it never changes the shipped choice.
    raw = write_copy(out / "raw-copy", selected[0] if selected else 1)
    chosen = out / "selected"
    if selected:
        candidate = write_copy(chosen, selected[0])
    else:
        import shutil
        shutil.copytree(baseline.parent, chosen)
        candidate = chosen / "circuits.dl"
    artifacts = {"baseline": baseline, "raw_copy": raw, "selected": candidate}
    frozen = {name: sha256(path) for name, path in artifacts.items()}
    (out / "frozen.json").write_text(json.dumps({"dataset_sha256": sha256(out / "dataset.json"),
        "artifact_sha256": frozen, "selected_length": selected, "selection": "zero train+validation errors and causal failures"}, indent=2) + "\n")
    for r in rows:
        if r["part"] not in ("train", "validation"):
            refs[r["id"]] = oracle(r)
    (out / "references.json").write_text(json.dumps(refs, indent=2) + "\n")
    results = {}
    test = [r for r in rows if r["part"] == "test"]  # entire preregistered repeat+intervention domain; no filtering by success
    for name, path in artifacts.items():
        if sha256(path) != frozen[name]:
            raise ValueError("candidate changed after freeze")
        evaluated = evaluate(path, rows, refs, out / "certificate-evidence", provenance)
        cert = run_equiv(path, [r["ctx"] for r in test], [refs[r["id"]] for r in test],
                         evidence_dir=out / "certificate-evidence", provenance={**provenance, "stage": "untouched test"})
        results[name] = {**evaluated, "test_certificate": cert, "artifact_sha256": frozen[name],
                         "bytes": path.stat().st_size}
    report = dict(tag="empirical", provenance=provenance, dataset_sha256=sha256(out / "dataset.json"),
                  references_sha256=sha256(out / "references.json"), baseline_rules=baseline_rules,
                  selected_length=selected, admission=admission, baseline_learning=learning, results=results,
                  scope="finite generated nonce corpus; synthetic controls do not establish LLM behavior")
    # Evidence links must survive moving the benchmark directory.
    def relativize(value):
        if isinstance(value, dict):
            return {k: str(Path(v).relative_to(out)) if k == "evidence" else relativize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relativize(v) for v in value]
        return value
    report = relativize(report)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Copy/induction holdout benchmark", "", f"Oracle: `{provenance['source']}`. **empirical** corpus measurements.",
             "Selection used train and validation only. Test references were queried after artifact hashes were frozen.",
             f"Selected copy length: `{selected}` (empty means rejected; selected artifact uses the baseline).", "",
             "| Artifact | Test correct / total | Coverage | Precision | Wrong / total | Off-domain answers | Exact test certificate |",
             "|---|---|---|---|---|---|---|"]
    for name, result in results.items():
        score = result["scores"]["test"]
        off = result["scores"]["off_domain"]
        precision = f"{score['precision']:.1%}" if score["precision"] is not None else "undefined"
        verdict = "proved on stated test domain" if result["test_certificate"]["certified"] else "not certified"
        lines.append(f"| {name} | {score['correct']}/{score['n']} | {score['coverage']:.1%} | {precision} | "
                     f"{score['wrong']}/{score['n']} | {off['answered']}/{off['n']} | {verdict} |")
    lines += ["", f"Baseline: {baseline_rules} suffix facts. Runtime sizes (including Datalog scaffolding): " +
              ", ".join(f"{k}={v['bytes']} bytes" for k, v in results.items()), "",
              "Synthetic controls certify only their authored task. Real-model certificates are relative to the recorded oracle.",
              "No generalization or faithfulness outside the recorded finite domain is proved. Off-domain here means no-repeat",
              "nonce contexts, not a broad natural-language safety evaluation. See `report.json` for all split metrics and evidence."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--oracle", choices=("copy-control", "constant-control", "fieldrun"), default="copy-control")
    parser.add_argument("--port", type=int)
    parser.add_argument("--bundle", help="bundle stem for provenance; fieldrun must already serve this bundle")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = generate(args.seed)
    provenance = {"source": args.oracle, "seed": args.seed, "synthetic": args.oracle != "fieldrun"}
    if args.oracle == "fieldrun":
        if not args.port or not args.bundle:
            parser.error("fieldrun requires --port and --bundle")
        provenance["bundle_sha256"] = {suffix: sha256(args.bundle + suffix) for suffix in (".fieldrun.json", ".fieldrun.bin")}
        def oracle(row):
            return serve_decide(args.port, row["ctx"])
    elif args.oracle == "copy-control":
        def oracle(row):
            return row["expected"]
    else:
        def oracle(row):
            return 0
    report = benchmark(args.out, rows, oracle, provenance)
    print(json.dumps({"selected_length": report["selected_length"],
                      "test": {k: v["scores"]["test"] for k, v in report["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
