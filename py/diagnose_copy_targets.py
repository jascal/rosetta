"""Controlled target substitutions to diagnose copy failures. Discovery experiment; no guard selection."""
import argparse
import json
from pathlib import Path
import random
import shutil

from benchmark_copy_generalization import score, verify_frozen
from benchmark_guarded_copy import freeze_domain
from benchmark_induction import DL
from certificate import sha256
from oracle import run_equiv, serve_decide

PRIOR = Path("reference/benchmarks/qwen25_05b_copy_generalization")
# Each punctuation target is paired with one lexical target. No class is inferred by the runtime.
TARGETS = [(11436, "punctuation_newline", "discovery"), (2579, "lexical", "discovery"),
           (34341, "punctuation_newline", "discovery"), (1602, "lexical", "discovery"),
           (317, "punctuation_newline", "new_target"), (23268, "lexical", "new_target"),
           (698, "punctuation_newline", "new_target"), (13551, "lexical", "new_target")]


def generate(exclude=(), seeds=(5, 6), lengths=(8, 12), prefixes=(4, 6), gaps=(0, 8, 32)):
    if not seeds or not lengths or not prefixes or not gaps or min(prefixes) < 3 or max(prefixes) >= min(lengths):
        raise ValueError("need nonempty factors and 3 <= prefix < every sequence length")
    if min(gaps) < 0:
        raise ValueError("gaps cannot be negative")
    used = set(exclude) | {t for t, _, _ in TARGETS}
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        for length in lengths:
            values = rng.sample([t for t in range(200, 40000) if t not in used], length + 2 * max(gaps))
            used.update(values)
            seq = values[:length]
            noise = [values[length + b * max(gaps):length + (b + 1) * max(gaps)] for b in range(2)]
            for prefix in prefixes:
                for gap in gaps:
                    template, positions = [], []
                    for b in range(2):
                        positions.append(len(template) + prefix)
                        template.extend(seq)
                        template.extend(noise[b][:gap])
                    template.extend(seq[:prefix])
                    cell_start = len(rows)
                    for j, (target, token_class, origin) in enumerate(TARGETS):
                        ctx = template[:]
                        for p in positions:
                            ctx[p] = target
                        row = dict(id=len(rows), part="test", group=f"seed{seed}-length{length}", seed=seed,
                                   length=length, prefix=prefix, gap=gap, layout=f"gap{gap}", exposures=2,
                                   kind=token_class, token_class=token_class, target=target, target_origin=origin,
                                   ctx=ctx, expected=target, positions=positions)
                        if j % 2:
                            row["base"] = cell_start + j - 1
                        rows.append(row)
    return rows


def experiment(out, rows, oracle, provenance, prior=PRIOR, pieces=None):
    out, prior = Path(out), Path(prior)
    out.mkdir(parents=True, exist_ok=False)
    prior_frozen = json.loads((prior / "frozen.json").read_text())
    if sha256(prior / "guarded/circuits.dl") != prior_frozen["files"]["guarded/circuits.dl"]:
        raise ValueError("prior frozen guard changed")
    shutil.copytree(prior / "guarded", out / "guarded")
    candidate = out / "guarded/circuits.dl"
    (out / "dataset.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "strata-verifier.dl").write_text("".join((DL / name).read_text() for name in
                                               ("holdout.dl", "holdout_strata.dl", "copy_target_diagnostic.dl")))
    protocol = {"scope": "controlled diagnosis informed by previous failures; not untouched final-test evidence",
                "provenance": provenance, "targets": TARGETS, "tokenizer_pieces": pieces or {},
                "seeds": sorted({r["seed"] for r in rows}), "lengths": sorted({r["length"] for r in rows}),
                "prefixes": sorted({r["prefix"] for r in rows}), "gaps": sorted({r["gap"] for r in rows}),
                "driver_sha256": sha256(__file__), "guard_sha256": sha256(candidate),
                "selection": "none; length and prefix vary independently; every target in every background cell"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    domain, audit = freeze_domain(out, candidate, rows, provenance)
    audit["evidence"] = str(Path(audit["evidence"]).relative_to(out))
    frozen = {"domain": domain, "domain_audit": audit, "files": {filename: sha256(out / filename) for filename in
              ("dataset.json", "protocol.json", "strata-verifier.dl", "guarded/circuits.dl", "guarded/run.dl")}}
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_hash = sha256(out / "frozen.json")
    print(f"FROZEN: {len(rows)} diagnostic cases; fixed guard, no selection", flush=True)
    refs = {}
    with (out / "oracle-observations.jsonl").open("w") as observations:
        for n, row in enumerate(rows, 1):
            refs[row["id"]] = oracle(row)
            observations.write(json.dumps({"id": row["id"], "reference": refs[row["id"]]}) + "\n")
            observations.flush()
            if n % 24 == 0:
                print(f"diagnostic references: {n}/{len(rows)}", flush=True)
    verify_frozen(out, frozen)
    if sha256(out / "frozen.json") != frozen_hash:
        raise ValueError("freeze manifest changed")
    (out / "references.json").write_text(json.dumps(refs, indent=2) + "\n")
    pairs = [r for r in rows if "base" in r]
    staged = {"pair": "".join(f"{r['base']}\t{r['id']}\n" for r in pairs),
              "changed": "".join(f"{r['base']}\t{r['id']}\t{p}\t{r['ctx'][p]}\n" for r in pairs for p in r["positions"])}
    evaluated = score(out, candidate, rows, refs, provenance,
                      factors=("seed", "length", "prefix", "gap", "target", "token_class", "target_origin"), extra_facts=staged)
    # Full stated diagnostic domain; no passing subset is selected after observing references.
    cert = run_equiv(candidate, [r["ctx"] for r in rows], [refs[r["id"]] for r in rows],
                     evidence_dir=out / "certificate-evidence", provenance={**provenance, "scope": "entire diagnostic dataset"})

    def relative(value):
        if isinstance(value, dict):
            return {k: str(Path(v).relative_to(out)) if k == "evidence" else relative(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relative(v) for v in value]
        return value

    report = relative({"tag": "empirical", "scope": protocol["scope"], "provenance": provenance, **evaluated,
                       "certificate": cert, "frozen_sha256": frozen_hash, "references_sha256": sha256(out / "references.json")})
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    s = report["scores"]["test"]
    lines = ["# Copy target diagnosis", "", protocol["scope"], "",
             f"Oracle: `{provenance['source']}`. **empirical**: {s['correct']}/{s['n']} correct; {s['wrong']} wrong.",
             f"Exact certificate on entire diagnostic domain: `{cert['certified']}`.", "",
             "| Factor | Value | Correct / total | Wrong |", "|---|---|---:|---:|"]
    for factor, strata in report["strata"].items():
        for value, counts in strata.items():
            lines.append(f"| {factor} | {value} | {counts['correct']}/{counts['n']} | {counts['wrong']} |")
    lines += ["", "## Paired punctuation-to-lexical substitutions", "",
              "| Original token | Replacement token | Outcome | Pairs |", "|---|---|---|---:|"]
    for a, b, state, count in report["paired_count"]:
        lines.append(f"| {a} | {b} | {state} | {count} |")
    lines += ["", "All pairs edit only the copied-from successors; Datalog checks the declared edits and counts outcomes.",
              f"Target labels are experimental metadata, not runtime filters. Correlated variants share {len({r['group'] for r in rows})} background groups.",
              "This diagnoses a hypothesis; it does not select or certify a replacement guard on fresh holdout data."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--oracle", choices=("copy-control", "fieldrun"), default="copy-control")
    parser.add_argument("--port", type=int)
    parser.add_argument("--bundle")
    args = parser.parse_args()
    paths = [PRIOR.parent / name / "dataset.json" for name in
             ("qwen25_05b", "qwen25_05b_guarded_seed1", "qwen25_05b_copy_generalization")]
    excluded = {t for p in paths for row in json.loads(p.read_text()) for t in row["ctx"]}
    rows = generate(excluded)
    provenance = {"source": args.oracle, "prior_dataset_sha256": {str(p): sha256(p) for p in paths}}
    pieces = {}
    if args.oracle == "fieldrun":
        if not args.port or not args.bundle:
            parser.error("fieldrun needs --port and --bundle")
        provenance["bundle_sha256"] = {s: sha256(args.bundle + s) for s in (".fieldrun.bin", ".fieldrun.json")}
        previous = json.loads((PRIOR / "report.json").read_text())
        if provenance["bundle_sha256"] != previous["provenance"]["bundle_sha256"]:
            parser.error("diagnosis requires the same recorded Qwen bundle")
        tokenizer = Path(args.bundle + ".tokenizer.json")
        provenance["tokenizer_sha256"] = sha256(tokenizer)
        inv = {i: s for s, i in json.loads(tokenizer.read_text())["model"]["vocab"].items()}
        pieces = {target: inv[target] for target, _, _ in TARGETS}

    def oracle(row):
        return serve_decide(args.port, row["ctx"]) if args.oracle == "fieldrun" else row["expected"]

    report = experiment(args.out, rows, oracle, provenance, pieces=pieces)
    print(json.dumps({"scores": report["scores"], "token_class": report["strata"]["token_class"],
                      "paired_count": report["paired_count"]}, indent=2))


if __name__ == "__main__":
    main()
