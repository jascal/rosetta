"""Fresh guarded-copy experiment. Datalog selects the guard and freezes its test domain before test oracle calls."""
import argparse
import json
from pathlib import Path
import random

from benchmark_induction import DL, digest, evaluate, facts, learn_baseline, write_copy
from certificate import check, sha256
from oracle import run_equiv, serve_decide

# Keep the established benchmark family as the default. The expanded family is
# an explicit search experiment, not a silent change to prior benchmark runs.
GUARDS = [(i + 1, k, m) for i, (k, m) in enumerate((k, m) for k in (3, 6, 9) for m in (2, 3))]
EXPANDED_GUARDS = [(i + 1, k, m) for i, (k, m) in enumerate((k, m) for k in range(1, 13) for m in range(1, 5))]
STRICT_GUARD = (6, 9, 3)  # fixed diagnostic comparison: suffix-9 / three-source rule
RECENCY_HORIZONS = (16, 32, 64, 128)
RECENCY_GUARDS = []
RECENCY_HORIZON_BY_ID = {}
for _k in range(1, 13):
    for _m in range(1, 4):
        for _horizon in RECENCY_HORIZONS:
            _gid = len(RECENCY_GUARDS) + 1
            RECENCY_GUARDS.append((_gid, _k, _m))
            RECENCY_HORIZON_BY_ID[_gid] = _horizon
STRICT_RECENCY_GUARD = (0, 9, 1)
STRICT_RECENCY_HORIZON_BY_ID = {0: 32}
RANKED_HORIZONS = RECENCY_HORIZONS
RANKED_GUARDS = []
RANKED_HORIZON_BY_ID = {}
for _limit in range(1, 17):
    for _support in range(1, 4):
        for _horizon in RANKED_HORIZONS:
            _gid = len(RANKED_GUARDS) + 1
            RANKED_GUARDS.append((_gid, _limit, _support))
            RANKED_HORIZON_BY_ID[_gid] = _horizon
STRICT_RANKED_GUARD = (0, 16, 1)
STRICT_RANKED_HORIZON_BY_ID = {0: 32}


def generate(seed=1, groups=(4, 3, 8), exclude=()):
    """Whole sequence groups, all exposure counts, and their interventions stay in one split.

    The target is seen 1/2/3 times before a final prefix of length 3/6/9. Interventions replace ALL previous
    occurrences of the target. No-repeat and disagreeing-source contexts are separate negative controls.
    """
    if len(groups) != 3 or min(groups) < 2:
        raise ValueError("need at least two groups per split")
    excluded = set(exclude)
    pool = [t for t in range(200, 40000) if t not in excluded]
    tokens = iter(random.Random(seed).sample(pool, (sum(groups) + groups[2]) * 15))
    rows = []
    for part, count in zip(("train", "validation", "test", "off_domain"), (*groups, groups[2])):
        for group in range(count):
            values = [next(tokens) for _ in range(15)]
            seq, replacements = values[:12], values[12:]
            label = f"seed{seed}-{part}-{group}"
            if part == "off_domain":
                rows.append(dict(id=len(rows), group=label, part=part, kind="no_repeat", ctx=seq, expected=seq[0]))
                conflict = seq * 3 + seq[:9]
                conflict[9] = replacements[2]
                rows.append(dict(id=len(rows), group=label, part=part, kind="conflicting_sources",
                                 ctx=conflict, expected=seq[9]))
                continue
            for exposure in (1, 2, 3):
                for j, prefix in enumerate((3, 6, 9)):
                    context = seq * exposure + seq[:prefix]
                    base = len(rows)
                    row = dict(group=label, part=part, exposures=exposure, prefix=prefix)
                    rows.append(dict(row, id=base, kind="repeat", ctx=context, expected=seq[prefix]))
                    edited = context[:]
                    positions = [12 * block + prefix for block in range(exposure)]
                    for pos in positions:
                        edited[pos] = replacements[j]
                    rows.append(dict(row, id=len(rows), kind="intervention", base=base, positions=positions,
                                     ctx=edited, expected=replacements[j]))
    return rows


def guard_program(guards, policy="consensus", horizons=None):
    if policy not in ("consensus", "plurality", "recent", "ranked"):
        raise ValueError(f"unknown copy-guard policy: {policy}")
    if not guards:
        # The selected circuit is an explicit Datalog abstention. Omitting the
        # unreachable copy rules also avoids a Souffle compiler failure when
        # the guard relation has no facts.
        return ("// Standalone abstaining circuit: no guard passed admission.\n"
                ".decl cdecide(inst:number,out:number)\n")
    policy_file = {"consensus": "copy_guard.dl", "plurality": "copy_guard_plurality.dl",
                   "recent": "copy_guard_recent.dl", "ranked": "copy_guard_ranked.dl"}[policy]
    policy_comment = {"consensus": "consensus over all matching sources",
                      "plurality": "unique plurality of matching-source successors",
                      "recent": "consensus over sources within a frozen recency horizon",
                      "ranked": "nearest source among the longest suffix matches"}[policy]
    text = (f"// Standalone {policy_comment} copy circuit; abstains without sufficient support.\n"
            ".decl guard(g:number,k:number,m:number)\n" +
            "".join(f"guard({g},{k},{m}).\n" for g, k, m in guards) +
            ".decl copy_length(k:number)\ncopy_length(K) :- guard(_,K,_).\n")
    if policy in ("recent", "ranked"):
        default_horizons = RECENCY_HORIZON_BY_ID if policy == "recent" else RANKED_HORIZON_BY_ID
        horizons = default_horizons if horizons is None else horizons
        missing = {g for g, _, _ in guards} - set(horizons)
        if missing:
            raise ValueError(f"missing frozen horizons for guard IDs: {sorted(missing)}")
        text += (".decl guard_horizon(g:number,h:number)\n" +
                 "".join(f"guard_horizon({g},{horizons[g]}).\n" for g, _, _ in guards))
    return text + (DL / "copy_core.dl").read_text() + (DL / policy_file).read_text() + \
        "\n.decl cdecide(inst:number,out:number)\ncdecide(I,T) :- guard_prediction(_,I,T).\n"


def write_guard(directory, guards, policy="consensus", horizons=None):
    directory.mkdir()
    candidate = directory / "circuits.dl"
    candidate.write_text(guard_program(guards, policy, horizons))
    (directory / "run.dl").write_text(
        '.decl tok(inst:number,pos:number,id:number) .input tok\n#include "circuits.dl"\n.output cdecide\n'
        '.decl abstain(inst:number) .output abstain\nabstain(I) :- tok(I,_,_), !cdecide(I,_).\n')
    return candidate


def emit_sparse_baseline(path, rules, _composed, _symbols, _name):
    """Serialize Datalog-selected suffixes without Souffle's wide tuple limit."""
    lines = ["// Standalone train-only n-gram baseline; all matching/routing is Datalog.",
             (DL / "ngram_sparse_runtime.dl").read_text()]
    for number, (suffix, answer) in enumerate(sorted(rules.items())):
        lines.append(f"gram({number},{len(suffix)},{answer}).")
        lines.extend(f"gram_token({number},{offset},{token})."
                     for offset, token in enumerate(reversed(suffix)))
    candidate = Path(path)
    candidate.write_text("\n".join(lines) + "\n")
    (candidate.parent / "run.dl").write_text(
        '.decl tok(inst:number,pos:number,id:number) .input tok\n#include "circuits.dl"\n.output cdecide\n'
        '.decl abstain(inst:number) .output abstain\nabstain(I) :- tok(I,_,_), !cdecide(I,_).\n')


def admit(out, rows, refs, provenance, policy="consensus", horizons=None):
    program = out / "hypotheses.dl"
    program.write_text(guard_program(GUARDS, policy, horizons))
    staged = facts(rows, refs)
    staged["sample"] = "".join(f"{r['id']}\t{r['group']}\t{r['part']}\t{digest(r['ctx'])}\n" for r in rows)
    staged["intervention"] = "".join(f"{r['base']}\t{r['id']}\n" for r in rows if r["kind"] == "intervention")
    staged["changed"] = "".join(f"{r['base']}\t{r['id']}\t{p}\t{r['ctx'][p]}\n"
                                 for r in rows if r["kind"] == "intervention" for p in r["positions"])
    result = check(DL / "copy_guard_select.dl", program, staged, {"nselected": int},
                   evidence_dir=out / "certificate-evidence", provenance={**provenance, "stage": "development admission"})
    if not result["certified"]:
        raise ValueError(f"invalid admission inputs: {result}")
    selected = [tuple(map(int, row)) for row in result["relations"]["selected"]]
    return selected, result


def freeze_domain(out, candidate, test, provenance):
    staged = {"tok": facts(test, {})["tok"], "sample": "".join(f"{r['id']}\n" for r in test)}
    result = check(DL / "guard_domain.dl", candidate, staged, {"ndomain": int},
                   evidence_dir=out / "certificate-evidence", provenance={**provenance, "stage": "domain before test references"})
    if not result["certified"]:
        raise ValueError(f"invalid firing domain: {result}")
    domain = sorted(int(row[0]) for row in result.pop("relations")["domain"])
    return domain, result


def benchmark(out, rows, oracle, provenance, policy="consensus", horizons=None):
    # Evidence writers return absolute paths; normalize here so saved links are
    # stable whether the caller supplies a relative or absolute output path.
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "dataset.json").write_text(json.dumps(rows, indent=2) + "\n")
    protocol = {"provenance": provenance, "guards": GUARDS, "dataset_sha256": sha256(out / "dataset.json"),
                 "selection": "zero errors on firing development cases; causal pairs in >=2 groups in EACH split; "
                             "maximize development coverage, then shorter suffix, then lower source support",
                "policy": policy,
                 "guard_horizons": ({g: horizons[g] for g, _, _ in GUARDS} if policy in ("recent", "ranked")
                                   and horizons is not None else {}),
                "source_sha256": {name: sha256(DL / name) for name in
                                  ("copy_core.dl", {"consensus": "copy_guard.dl",
                                   "plurality": "copy_guard_plurality.dl", "recent": "copy_guard_recent.dl",
                                   "ranked": "copy_guard_ranked.dl"}[policy],
                                   "copy_guard_select.dl", "guard_domain.dl", "equiv.dl")},
                "driver_sha256": sha256(__file__)}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    development = [r for r in rows if r["part"] in ("train", "validation")]
    refs = {}
    for n, row in enumerate(development, 1):
        refs[row["id"]] = oracle(row)
        if n % 24 == 0:
            print(f"development references: {n}/{len(development)}", flush=True)
    selected, admission = admit(out, development, refs, provenance, policy, horizons)
    baseline, nrules, learning = learn_baseline(rows, refs, out, emitter=emit_sparse_baseline)
    artifacts = {"baseline": baseline, "raw_copy": write_copy(out / "raw-copy", 1),
                 "strict_guard": write_guard(out / "strict-guard",
                                              [STRICT_RECENCY_GUARD if policy == "recent" else
                                               STRICT_RANKED_GUARD if policy == "ranked" else STRICT_GUARD],
                                              policy, STRICT_RECENCY_HORIZON_BY_ID if policy == "recent" else
                                              STRICT_RANKED_HORIZON_BY_ID if policy == "ranked" else horizons),
                 "selected": write_guard(out / "selected", selected, policy, horizons)}
    test = [r for r in rows if r["part"] == "test"]
    frozen = {"protocol_sha256": sha256(out / "protocol.json"), "dataset_sha256": sha256(out / "dataset.json"),
              "selected": selected, "artifacts": {}}
    for name, path in artifacts.items():
        domain, audit = freeze_domain(out, path, test, provenance)
        audit["evidence"] = str(Path(audit["evidence"]).relative_to(out))
        frozen["artifacts"][name] = {"sha256": sha256(path), "domain": domain, "domain_audit": audit}
    # Preserve development references even if the final oracle collection is interrupted.
    (out / "development-references.json").write_text(json.dumps(refs, indent=2) + "\n")
    frozen["development_references_sha256"] = sha256(out / "development-references.json")
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"FROZEN: selected guards {selected}; collecting untouched test/negative references", flush=True)
    final = [r for r in rows if r["part"] not in ("train", "validation")]
    for n, row in enumerate(final, 1):
        refs[row["id"]] = oracle(row)
        if n % 24 == 0:
            print(f"final references: {n}/{len(final)}", flush=True)
    (out / "references.json").write_text(json.dumps(refs, indent=2) + "\n")
    results = {}
    for name, path in artifacts.items():
        frozen_artifact = frozen["artifacts"][name]
        if sha256(path) != frozen_artifact["sha256"]:
            raise ValueError("artifact changed after freeze")
        evaluated = evaluate(path, rows, refs, out / "certificate-evidence", provenance)
        domain = set(frozen_artifact["domain"])
        scoped = [r for r in test if r["id"] in domain]
        certs = {}
        for scope, cases in (("full_test", test), ("frozen_guard_domain", scoped)):
            certs[scope] = run_equiv(path, [r["ctx"] for r in cases], [refs[r["id"]] for r in cases],
                                    evidence_dir=out / "certificate-evidence",
                                    provenance={**provenance, "stage": scope, "original_instance_ids": [r["id"] for r in cases]})
        results[name] = {**evaluated, "certificates": certs, "bytes": path.stat().st_size,
                         "guard_domain_size": len(scoped), "sha256": sha256(path)}
    report = {"tag": "empirical", "provenance": provenance, "selected": selected, "baseline_rules": nrules,
              "admission": admission, "baseline_learning": learning, "results": results,
              "references_sha256": sha256(out / "references.json"), "frozen_sha256": sha256(out / "frozen.json")}

    def relative(value):
        if isinstance(value, dict):
            return {k: str(Path(v).relative_to(out)) if k == "evidence" else relative(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relative(v) for v in value]
        return value

    report = relative(report)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Guarded copy: fresh holdout", "", f"Oracle: `{provenance['source']}`. **Empirical** measurements.",
             f"Selected guard `(id, suffix length, minimum agreeing sources)`: `{selected}`.",
             "An empty selection emits an abstaining circuit. Candidates and firing domains were frozen before test references.", "",
             "| Artifact | Correct / all test | Wrong | Abstained | Frozen firing domain | Exact on firing domain? | Exact on full test? |",
             "|---|---:|---:|---:|---:|---|---|"]
    for name, result in results.items():
        s = result["scores"]["test"]
        certs = result["certificates"]
        lines.append(f"| {name} | {s['correct']}/{s['n']} | {s['wrong']} | {s['abstained']} | "
                     f"{result['guard_domain_size']} | {certs['frozen_guard_domain']['certified']} | {certs['full_test']['certified']} |")
    lines += ["", "A clean firing-domain certificate proves equality only on that explicitly frozen finite subset; it does not",
              "prove equality on abstentions or arbitrary future inputs. Empty domains fail certification. The diagnostic strict",
              "guard is always suffix length 9 / three sources, whether or not admission accepted it. Off-domain measurements",
              "cover no-repeat and conflicting-source controls only. See `report.json` for every split and retained evidence."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--oracle", choices=("copy-control", "constant-control", "fieldrun"), default="copy-control")
    parser.add_argument("--port", type=int)
    parser.add_argument("--bundle")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expanded-search", action="store_true",
                        help="search suffix lengths 1..12 and agreeing-source support 1..4")
    parser.add_argument("--policy", choices=("consensus", "plurality", "recent", "ranked"), default="consensus",
                        help="successor aggregation over matching sources")
    parser.add_argument("--prior", type=Path, action="append",
                        help="prior dataset whose token IDs must be excluded (repeatable)")
    args = parser.parse_args()
    global GUARDS
    if args.policy == "recent":
        GUARDS = RECENCY_GUARDS
    elif args.policy == "ranked":
        GUARDS = RANKED_GUARDS
    elif args.expanded_search:
        GUARDS = EXPANDED_GUARDS
    priors = args.prior or [Path("reference/benchmarks/qwen25_05b/dataset.json")]
    prior_rows = [row for prior in priors for row in json.loads(prior.read_text())]
    rows = generate(args.seed, exclude={t for row in prior_rows for t in row["ctx"]})
    provenance = {"source": args.oracle, "seed": args.seed, "synthetic": args.oracle != "fieldrun",
                  "prior_dataset_sha256": {str(prior): sha256(prior) for prior in priors},
                  "prior_usage": "discovery only; all previous token IDs excluded",
                  "policy": args.policy,
                  "search": (("suffix lengths 1..16" if args.policy == "ranked" else "suffix lengths 1..12") +
                             ", support 1..3, horizons 16/32/64/128" if args.policy in ("recent", "ranked")
                             else "suffix lengths 1..12, agreeing-source support 1..4" if args.expanded_search
                             else "suffix lengths 3,6,9; agreeing-source support 2,3")}
    if args.oracle == "fieldrun":
        if not args.bundle or not args.port:
            parser.error("fieldrun requires --bundle and --port")
        provenance["bundle_sha256"] = {s: sha256(args.bundle + s) for s in (".fieldrun.bin", ".fieldrun.json")}

    def oracle(row):
        if args.oracle == "fieldrun":
            return serve_decide(args.port, row["ctx"])
        return row["expected"] if args.oracle == "copy-control" else 0

    report = benchmark(args.out, rows, oracle, provenance, args.policy,
                       RECENCY_HORIZON_BY_ID if args.policy == "recent" else
                       RANKED_HORIZON_BY_ID if args.policy == "ranked" else None)
    print(json.dumps({"selected": report["selected"], "test": {k: v["scores"]["test"] for k, v in report["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
