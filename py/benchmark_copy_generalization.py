"""Evaluate previously frozen copy circuits on new seeds, lengths and distractors. No selection or retraining."""
import argparse
import json
from pathlib import Path
import random
import shutil

from benchmark_guarded_copy import freeze_domain
from benchmark_induction import DL, digest, facts
from certificate import check, sha256
from oracle import run_equiv, serve_decide

DEFAULT_PRIOR = Path("reference/benchmarks/qwen25_05b_guarded_seed1")
LAYOUTS = ("plain", "noise8", "noise32", "near_match8")
SELECTION_PARTS = ("train", "validation")


def source_sequence(row):
    """The copied sequence of a positive row (ctx = (seq + gap) * exposures + seq[:prefix]); None for negatives."""
    if row.get("kind") not in ("repeat", "intervention") or not row.get("exposures"):
        return None
    if "seq" in row:
        return row["seq"]
    length = row.get("length") or (len(row["ctx"]) - row["prefix"]) // row["exposures"]  # gap-free prior rows
    return row["ctx"][:length]


def known_pairs(selection_rows):
    """(token, slot) pairs the circuit's selection data showed: every slot of every train/validation source sequence."""
    pairs = set()
    for row in selection_rows:
        seq = source_sequence(row) if row.get("part") in SELECTION_PARTS else None
        pairs.update((t, s) for s, t in enumerate(seq or ()))
    return pairs


def binding_facts(rows, known):
    """Facts for dl/binding_baseline.dl: unconsumed slots [prefix, len(seq)) of each test row and its target slot."""
    slots, targets = [], []
    for row in rows:
        seq = source_sequence(row) if row["part"] == "test" else None
        if seq is None:
            continue
        if row["kind"] == "intervention":
            seq = seq[:row["prefix"]] + [row["expected"]] + seq[row["prefix"] + 1:]
        slots.extend(f"{row['id']}\t{s}\t{seq[s]}\n" for s in range(row["prefix"], len(seq)))
        targets.append(f"{row['id']}\t{row['prefix']}\n")
    return {"slot": "".join(slots), "target_slot": "".join(targets),
            "known": "".join(f"{t}\t{s}\n" for t, s in sorted(known)),
            "gold": "".join(f"{r['id']}\t{r['expected']}\n" for r in rows if r["part"] == "test")}


def generate_withheld(known, seeds=(5, 6, 7), novel=range(7), length=12, groups=1, exclude=()):
    """Withheld-pair protocol (McCoy et al. 2026, §8): vary k, the number of NOVEL (token, slot) pairs among the
    unconsumed slots [prefix, length). Novel pairs come first, so the queried slot is novel iff k >= 1; the other
    unconsumed slots reuse a token the selection data showed in that same slot. Prefix tokens are fresh.
    """
    prefix = length // 2
    by_slot = {}
    for token, slot in known:
        by_slot.setdefault(slot, []).append(token)
    novel = list(novel)
    if any(not 0 <= k <= length - prefix for k in novel):
        raise ValueError("k must lie in [0, length - prefix]")
    if any(len(by_slot.get(s, ())) < 2 for s in range(prefix, length)):
        raise ValueError("selection data lacks known pairs for the unconsumed slots")
    used = set(exclude) | {t for t, _ in known}
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        for k in novel:
            for part in ("test", "off_domain"):
                for group in range(groups):
                    values = rng.sample([t for t in range(200, 40000) if t not in used], length + 18)
                    used.update(values)
                    fresh, replacement, noise = values[:length], values[length], values[length + 2:length + 18]
                    seq = fresh[:prefix + k]
                    for s in range(prefix + k, length):
                        seq.append(rng.choice([t for t in sorted(by_slot[s]) if t not in seq]))
                    meta = dict(seed=seed, length=length, prefix=prefix, part=part, design_k=k,
                                group=f"withheld-seed{seed}-k{k}-{part}-{group}")
                    if part == "off_domain":
                        rows.append(dict(meta, id=len(rows), kind="no_repeat", layout="negative", exposures=0,
                                         ctx=seq, expected=seq[prefix]))
                        continue
                    for exposure in (2, 3):
                        for layout in ("plain", "noise8"):
                            context, positions = [], []
                            for b in range(exposure):
                                positions.append(len(context) + prefix)
                                context.extend(seq)
                                context.extend([] if layout == "plain" else noise[8 * (b % 2):8 * (b % 2) + 8])
                            context.extend(seq[:prefix])
                            base = len(rows)
                            variant = dict(meta, layout=layout, exposures=exposure, seq=seq)
                            rows.append(dict(variant, id=base, kind="repeat", ctx=context, expected=seq[prefix]))
                            edited = context[:]
                            for p in positions:
                                edited[p] = replacement
                            rows.append(dict(variant, id=len(rows), kind="intervention", base=base,
                                             positions=positions, ctx=edited, expected=replacement))
    return rows


def generate(seeds=(2, 3, 4), lengths=(8, 12, 24), groups=2, exclude=()):
    """Generate paired interventions with group-disjoint vocabulary and independent negative groups.

    Near matches repeat only the last TWO query tokens followed by a decoy, testing the fixed three-token guard.
    Noise blocks differ between source occurrences. All changed source positions are retained for inspection.
    """
    if not seeds or len(set(seeds)) != len(seeds) or not lengths or min(lengths) < 8 or groups < 1:
        raise ValueError("need unique seeds, lengths >=8, and positive groups")
    used = set(exclude)
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        for length in lengths:
            for part in ("test", "off_domain"):
                for group in range(groups):
                    values = rng.sample([t for t in range(200, 40000) if t not in used], length + 98)
                    used.update(values)
                    seq = values[:length]
                    replacement, decoy = values[length:length + 2]
                    noise = [values[length + 2 + 32 * b:length + 2 + 32 * (b + 1)] for b in range(3)]
                    prefix = length // 2
                    meta = dict(seed=seed, length=length, prefix=prefix, part=part,
                                group=f"seed{seed}-length{length}-{part}-{group}")
                    if part == "off_domain":
                        conflict = seq * 2 + seq[:prefix]
                        conflict[prefix] = replacement
                        controls = [("no_repeat", seq), ("one_source", seq + noise[0][:8] + seq[:prefix]),
                                    ("conflicting_sources", conflict)]
                        for kind, ctx in controls:
                            rows.append(dict(meta, id=len(rows), kind=kind, layout="negative", exposures=0,
                                             ctx=ctx, expected=seq[prefix]))
                        continue
                    for exposure in (2, 3):
                        for layout in LAYOUTS:
                            context, positions = [], []
                            for b in range(exposure):
                                positions.append(len(context) + prefix)
                                context.extend(seq)
                                gap = [] if layout == "plain" else noise[b][:32 if layout == "noise32" else 8]
                                if layout == "near_match8":
                                    gap[-3:] = [seq[prefix - 2], seq[prefix - 1], decoy]
                                context.extend(gap)
                            context.extend(seq[:prefix])
                            base = len(rows)
                            variant = dict(meta, layout=layout, exposures=exposure)
                            rows.append(dict(variant, id=base, kind="repeat", ctx=context, expected=seq[prefix]))
                            edited = context[:]
                            for p in positions:
                                edited[p] = replacement
                            rows.append(dict(variant, id=len(rows), kind="intervention", base=base, positions=positions,
                                             ctx=edited, expected=replacement))
    return rows


def summarize(counts):
    n, answered, correct, wrong, abstained = map(int, counts)
    return dict(n=n, answered=answered, correct=correct, wrong=wrong, abstained=abstained,
                coverage=answered / n if n else None, precision=correct / answered if answered else None)


def score(out, candidate, rows, refs, provenance, *, factors=None, extra_facts=None, known=None):
    staged = facts(rows, refs)
    if known is not None:
        staged.update(binding_facts(rows, known))
    staged["sample"] = "".join(f"{r['id']}\t{r['group']}\t{r['part']}\t{digest(r['ctx'])}\n" for r in rows)
    buckets = []
    for row in rows:
        fields = (factors or ("seed", "length", "layout", "exposures", "kind")) if row["part"] == "test" else ()
        buckets.extend((row["id"], field, row[field]) for field in fields)
        if row["part"] == "off_domain":
            buckets.append((row["id"], "negative", row["kind"]))
    staged["bucket"] = "".join(f"{i}\t{factor}\t{value}\n" for i, factor, value in buckets)
    if extra_facts:
        staged.update(extra_facts)
    audit = check(out / "strata-verifier.dl", candidate, staged, {"ninvalid": int, "nleak": int},
                  evidence_dir=out / "certificate-evidence", provenance=provenance)
    if not audit["certified"]:
        raise ValueError(f"evaluation audit failed: {audit}")
    relations = audit.pop("relations")
    strata = {}
    for factor, value, *counts in relations["breakdown"]:
        strata.setdefault(factor, {})[value] = summarize(counts)
    by_id = {r["id"]: r for r in rows}
    errors = []
    for i, reference, prediction in relations["counterexample"]:
        row = by_id[int(i)]
        errors.append({key: row[key] for key in ("id", "part", "group", "seed", "length", "layout", "kind", "exposures")} |
                      {"reference": int(reference), "prediction": int(prediction)})
    result = {"audit": audit, "scores": {part: summarize(counts) for part, *counts in relations["score"]},
              "strata": strata, "counterexamples": errors}
    if "paired_count" in relations:
        result["paired_count"] = relations["paired_count"]
    if "binding_baseline" in relations:
        result["binding_baseline"] = binding_summary(relations["binding_baseline"])
    return result


def binding_summary(relation):
    """Per-k rows plus a total; chance is the expected number of binding-blind hits on the gold token."""
    rows = sorted(({"k": int(k), "n": int(n), "model_gold": int(g), "answered": int(a), "agree": int(c),
                    "chance": float(e)} for k, n, g, a, c, e in relation), key=lambda r: r["k"])
    total = {key: sum(r[key] for r in rows) for key in ("n", "model_gold", "answered", "agree", "chance")}
    return {"by_k": rows, "total": total}


def strata_verifier():
    return "".join((DL / name).read_text() for name in ("holdout.dl", "holdout_strata.dl", "binding_baseline.dl"))


def binding_lines(summary, title):
    lines = ["", f"## {title}", "",
             "k = novel (token, slot) pairs among the unconsumed slots, relative to the guard's selection data. Chance is the",
             "expected gold-token hits of the strong binding-blind baseline (1 if the queried pair is known, else 1/k).", "",
             "| k | Cases | Model = gold | Chance hits | Guard answered | Guard = model |", "|---:|---:|---:|---:|---:|---:|"]
    for r in summary["by_k"] + [dict(summary["total"], k="all")]:
        lines.append(f"| {r['k']} | {r['n']} | {r['model_gold']} | {r['chance']:.1f} | {r['answered']} | {r['agree']} |")
    return lines


def verify_frozen(out, frozen):
    for filename, expected in frozen["files"].items():
        if sha256(out / filename) != expected:
            raise ValueError(f"frozen input changed: {filename}")


def benchmark(out, rows, oracle, provenance, prior=DEFAULT_PRIOR, protocol_name="fixed-guard-generalization"):
    out, prior = Path(out).resolve(), Path(prior)  # evidence paths are absolute; relative_to needs an absolute out
    known = known_pairs(json.loads((prior / "dataset.json").read_text()))
    out.mkdir(parents=True, exist_ok=False)
    previous = json.loads((prior / "frozen.json").read_text())
    artifacts = {}
    for name, source in (("guarded", "selected"), ("raw_copy", "raw-copy")):
        prior_key = "selected" if name == "guarded" else "raw_copy"
        if sha256(prior / source / "circuits.dl") != previous["artifacts"][prior_key]["sha256"]:
            raise ValueError("prior frozen circuit changed")
        directory = out / name
        directory.mkdir()
        for filename in ("circuits.dl", "run.dl"):
            shutil.copyfile(prior / source / filename, directory / filename)
        artifacts[name] = directory / "circuits.dl"
    (out / "dataset.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "strata-verifier.dl").write_text(strata_verifier())
    protocol = {"name": protocol_name, "provenance": provenance, "known_pairs": len(known), "selection": "none; both circuits copied byte-for-byte from prior frozen artifacts",
                "prior_frozen_sha256": sha256(prior / "frozen.json"), "driver_sha256": sha256(__file__),
                "seeds": sorted({r["seed"] for r in rows}), "lengths": sorted({r["length"] for r in rows}),
                "layouts": LAYOUTS, "test_groups": len({r["group"] for r in rows if r["part"] == "test"}),
                "scope": "all preregistered test cases and each reference-independent firing domain"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    test = [r for r in rows if r["part"] == "test"]
    frozen = {"files": {}, "artifacts": {}}
    for name, candidate in artifacts.items():
        domain, audit = freeze_domain(out, candidate, test, provenance)
        audit["evidence"] = str(Path(audit["evidence"]).relative_to(out))
        frozen["artifacts"][name] = {"domain": domain, "domain_audit": audit}
    for filename in ("dataset.json", "protocol.json", "strata-verifier.dl",
                     "guarded/circuits.dl", "guarded/run.dl", "raw_copy/circuits.dl", "raw_copy/run.dl"):
        frozen["files"][filename] = sha256(out / filename)
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_hash = sha256(out / "frozen.json")
    print(f"FROZEN: {len(test)} test cases, {len(rows)-len(test)} negatives; no circuit selection", flush=True)
    refs = {}
    with (out / "oracle-observations.jsonl").open("w") as observations:
        for n, row in enumerate(rows, 1):
            reference = oracle(row)
            refs[row["id"]] = reference
            observations.write(json.dumps({"id": row["id"], "reference": reference}) + "\n")
            observations.flush()
            if n % 24 == 0:
                print(f"oracle references: {n}/{len(rows)}", flush=True)
    verify_frozen(out, frozen)
    if sha256(out / "frozen.json") != frozen_hash:
        raise ValueError("freeze manifest changed during evaluation")
    (out / "references.json").write_text(json.dumps(refs, indent=2) + "\n")
    results = {}
    for name, candidate in artifacts.items():
        evaluated = score(out, candidate, rows, refs, provenance, known=known)
        domain = set(frozen["artifacts"][name]["domain"])
        scoped = [r for r in test if r["id"] in domain]
        certs = {}
        for scope, cases in (("full_test", test), ("frozen_firing_domain", scoped)):
            certs[scope] = run_equiv(candidate, [r["ctx"] for r in cases], [refs[r["id"]] for r in cases],
                                    evidence_dir=out / "certificate-evidence",
                                    provenance={**provenance, "scope": scope, "original_instance_ids": [r["id"] for r in cases]})
        results[name] = {**evaluated, "certificates": certs, "firing_domain_size": len(scoped), "sha256": sha256(candidate)}

    def relative(value):
        if isinstance(value, dict):
            return {k: str(Path(v).relative_to(out)) if k == "evidence" else relative(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relative(v) for v in value]
        return value

    report = relative({"tag": "empirical", "provenance": provenance, "results": results,
                       "frozen_sha256": frozen_hash, "references_sha256": sha256(out / "references.json"),
                       "observations_sha256": sha256(out / "oracle-observations.jsonl")})
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Fixed copy guard: generalization test", "", f"Oracle: `{provenance['source']}`. **Empirical** measurements.",
             "Circuits were copied unchanged from the prior experiment. Dataset and firing domains were frozen before all oracle queries.", "",
             "| Circuit | Correct / all test | Wrong | Abstained | Negative answers | Exact firing-domain certificate |",
             "|---|---:|---:|---:|---:|---|"]
    for name, result in report["results"].items():
        s, neg = result["scores"]["test"], result["scores"]["off_domain"]
        lines.append(f"| {name} | {s['correct']}/{s['n']} | {s['wrong']} | {s['abstained']} | "
                     f"{neg['answered']}/{neg['n']} | {result['certificates']['frozen_firing_domain']['certified']} |")
    lines += ["", "## Guarded circuit by predeclared test condition", "",
              "| Factor | Value | Correct / total | Wrong | Abstained |", "|---|---|---:|---:|---:|"]
    for factor, strata in report["results"]["guarded"]["strata"].items():
        for value, s in strata.items():
            lines.append(f"| {factor} | {value} | {s['correct']}/{s['n']} | {s['wrong']} | {s['abstained']} |")
    lines += binding_lines(report["results"]["guarded"]["binding_baseline"], "Withheld pairs and the binding-blind baseline")
    lines += ["", "Exact certificates apply only to their stated finite domain, relative to recorded references. A failure is retained",
              "as a counterexample; no successful subset is certified after observing references. Strata are descriptive measurements,",
              "not newly selected rules. Correlated variants share sequence groups. See `report.json` for raw counters and evidence."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def rescore(run, out, prior=DEFAULT_PRIOR):
    """Binding baseline for a COMPLETED run from its recorded dataset and references. No oracle queries.

    The run directory is only read; frozen inputs are re-verified first and the result goes to a new directory.
    """
    run, out, prior = Path(run), Path(out), Path(prior)
    verify_frozen(run, json.loads((run / "frozen.json").read_text()))
    rows = json.loads((run / "dataset.json").read_text())
    refs = {int(k): v for k, v in json.loads((run / "references.json").read_text()).items()}
    out.mkdir(parents=True, exist_ok=False)
    (out / "strata-verifier.dl").write_text(strata_verifier())
    known = known_pairs(json.loads((prior / "dataset.json").read_text()))
    provenance = {"source": "recorded-references", "run": str(run), "frozen_sha256": sha256(run / "frozen.json"),
                  "references_sha256": sha256(run / "references.json"), "driver_sha256": sha256(__file__)}
    report = {"tag": "empirical", "provenance": provenance, "known_pairs": len(known), "results": {}}
    for name in ("guarded", "raw_copy"):
        evaluated = score(out, run / name / "circuits.dl", rows, refs, provenance, known=known)
        report["results"][name] = {"binding_baseline": evaluated["binding_baseline"], "scores": evaluated["scores"]}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    lines = ["# Binding-blind baseline for a recorded run", "", f"Run: `{run}` (references recorded; no new oracle queries).",
             "**Empirical.**"]
    lines += binding_lines(report["results"]["guarded"]["binding_baseline"], "Guarded circuit")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--oracle", choices=("copy-control", "constant-control", "fieldrun"), default="copy-control")
    parser.add_argument("--protocol", choices=("fixed", "withheld-pairs"), default="fixed")
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--port", type=int)
    parser.add_argument("--bundle")
    parser.add_argument("--allow-bundle-mismatch", action="store_true",
                        help="withheld-pairs only: test the guard on a bundle other than its selection bundle (recorded)")
    parser.add_argument("--rescore", type=Path, help="compute the binding baseline for a completed run; no oracle")
    args = parser.parse_args()
    if args.rescore:
        report = rescore(args.rescore, args.out, args.prior)
        print(json.dumps({k: v["binding_baseline"]["total"] for k, v in report["results"].items()}, indent=2))
        return
    prior_data = [args.prior / "dataset.json", Path("reference/benchmarks/qwen25_05b/dataset.json")]
    excluded = {t for path in prior_data for row in json.loads(path.read_text()) for t in row["ctx"]}
    if args.protocol == "fixed":
        rows = generate(exclude=excluded)
    else:
        rows = generate_withheld(known_pairs(json.loads((args.prior / "dataset.json").read_text())), exclude=excluded)
    provenance = {"source": args.oracle, "synthetic": args.oracle != "fieldrun", "protocol": args.protocol,
                  "excluded_dataset_sha256": {str(p): sha256(p) for p in prior_data}}
    if args.oracle == "fieldrun":
        if not args.port or not args.bundle:
            parser.error("fieldrun needs --port and --bundle")
        provenance["bundle_sha256"] = {s: sha256(args.bundle + s) for s in (".fieldrun.bin", ".fieldrun.json")}
        prior_report = json.loads((args.prior / "report.json").read_text())
        same = provenance["bundle_sha256"] == prior_report["provenance"]["bundle_sha256"]
        provenance["same_bundle_as_selection"] = same
        if not same and not (args.protocol == "withheld-pairs" and args.allow_bundle_mismatch):
            parser.error("fixed-guard replication must use the same recorded model bundle")

    def oracle(row):
        if args.oracle == "fieldrun":
            return serve_decide(args.port, row["ctx"])
        return row["expected"] if args.oracle == "copy-control" else 0

    name = "fixed-guard-generalization" if args.protocol == "fixed" else "withheld-pairs"
    report = benchmark(args.out, rows, oracle, provenance, args.prior, protocol_name=name)
    print(json.dumps({k: v["scores"] for k, v in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
