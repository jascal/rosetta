"""Compiled TPR behind n-gram and idiom layers (pre-registered: docs/compiled-tpr-prereg.md).

    python3 py/benchmark_compiled_tpr.py generate OUT --tokenizer .../gpt2.tokenizer.json   # datasets, stdlib only
    # pil: experiments/compile_tpr.py trains the certificate-aware TPR and exports weighted facts to OUT/<task>/tpr
    fieldrun --bundle .../gpt2 --serve 8189
    python3 py/benchmark_compiled_tpr.py run OUT --port 8189 --bundle .../gpt2

Layers (router order): train-only n-gram (dev-filtered) -> idiom (SVO copy-subject with a dev-selected verb
whitelist; COPY the frozen guarded copy circuit, unchanged) -> compiled TPR (Datalog role parse, weighted score facts,
dev-selected margin threshold) -> abstain. Every runtime layer is Soufflé-only; certificates are dl/equiv.dl.
"""
import argparse
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile

from benchmark_induction import DL, learn_baseline
from certificate import check, sha256
from oracle import serve_decide

ROOT = Path(__file__).resolve().parents[1]
COPY_IDIOM = ROOT / "reference/benchmarks/qwen25_05b_guarded_seed1/selected/circuits.dl"
OCCUPATIONS = """doctor lawyer teacher nurse pilot farmer baker chef judge poet painter singer dancer actor writer
author editor banker soldier sailor driver student scientist engineer artist captain priest waiter guard coach player
officer agent tourist manager clerk tailor butcher miner hunter spy king queen prince""".split()
VERBS = """helped called thanked liked hated pushed followed visited praised blamed hired warned chased loved watched
kissed attacked admired avoided""".split()
NOUNS = """apple table river horse window garden pencil mirror candle bottle rabbit forest island castle bridge jacket
carpet basket kitchen engine ladder pepper rubber lion trouble kindness coffee chair house money water music paper stone
bread cloud train plane ship tree flower school church phone glass metal wood fire snow rain wind salt sugar milk cheese
butter egg fish bird dog cat cow pig sheep mouse snake bear wolf fox deer duck eagle shark whale frog bee rose grass sand
rock gold silver iron steel coal oil glove shirt shoe hat coat ring clock lamp bed door wall roof floor road street park
farm field hill lake sea ocean beach desert valley cave star moon sun planet book letter map flag box bag cup plate knife
spoon bowl pot pan key coin ticket card camera radio piano guitar drum ball kite toy doll game bike truck boat car bus""".split()
SEEDS = dict(SVO=31, COPY=32, split=33)
N_CONTEXTS = 12000
SCALE = 2 ** 16
COPY_RMAX = 36


def vocab(tokenizer):
    return json.loads(Path(tokenizer).read_text())["model"]["vocab"]


def single(v, words, cap):
    return [v["Ġ" + w] for w in words if "Ġ" + w in v][:cap]


# ---------------------------------------------------------------- datasets

def generate_svo(v, rng):
    occ, verbs = single(v, OCCUPATIONS, 40), single(v, [w for w in VERBS if w.endswith("ed")], 16)
    THE, the, dot, was, by = v["The"], v["Ġthe"], v["."], v["Ġwas"], v["Ġby"]
    combos = [(s, vb, o) for s in occ for o in occ if s != o for vb in verbs]
    rng.shuffle(combos)
    rows = [dict(ctx=[THE, s, vb, the, o, dot, THE, o, was, vb, by, the], subject=s, verb=vb, object=o)
            for s, vb, o in combos[:N_CONTEXTS]]
    meta = dict(fillers=occ + verbs, n_role=3, template={"0": THE, "3": the, "5": dot, "6": THE, "8": was,
                                                          "10": by, "11": the})
    return rows, meta


def generate_copy(v, rng):
    pool = single(v, NOUNS, 120)
    rows = []
    for layout in ("plain", "noise8", "near_match8"):
        for _ in range(N_CONTEXTS // 3):
            vals = rng.sample(pool, 8 + 1 + 16)
            seq, decoy, noise = vals[:8], vals[8], [vals[9:17], vals[17:25]]
            ctx = []
            for b in range(2):
                ctx += seq
                gap = [] if layout == "plain" else list(noise[b])
                if layout == "near_match8":
                    gap[-3:] = [seq[2], seq[3], decoy]
                ctx += gap
            ctx += seq[:4]
            rows.append(dict(ctx=ctx, layout=layout, target=seq[4]))
    rng.shuffle(rows)
    return rows, dict(fillers=pool, n_role=COPY_RMAX)


def cmd_generate(args):
    global N_CONTEXTS
    if args.smoke:
        N_CONTEXTS = 600
        for k in SEEDS:
            SEEDS[k] += 1000
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    v = vocab(args.tokenizer)
    for task, gen in (("SVO", generate_svo), ("COPY", generate_copy)):
        rows, meta = gen(v, random.Random(SEEDS[task]))
        order = list(range(len(rows)))
        random.Random(SEEDS["split"]).shuffle(order)
        n_tr, n_dev = int(0.6 * len(rows)), int(0.1 * len(rows))
        for rank, i in enumerate(order):
            rows[i]["part"] = "train" if rank < n_tr else ("dev" if rank < n_tr + n_dev else "test")
        for i, row in enumerate(rows):
            row["id"], row["group"] = i, f"{task}-{i}"
        (out / task).mkdir()
        (out / task / "dataset.json").write_text(json.dumps(dict(task=task, meta=meta, rows=rows)) + "\n")
        print(f"{task}: {len(rows)} contexts, {len(meta['fillers'])} fillers")
    (out / "protocol.json").write_text(json.dumps(dict(prereg="docs/compiled-tpr-prereg.md", seeds=SEEDS,
                                                       tokenizer_sha256=sha256(args.tokenizer),
                                                       driver_sha256=sha256(__file__)), indent=2) + "\n")


# ---------------------------------------------------------------- the compiled TPR (Soufflé-only runtime)

PARSE = {
    "SVO": """// Role parse from tok facts: fixed template, subject/verb/object at positions 1/2/4.
.decl tp_len(inst:number,k:number)
tp_len(I,K) :- tok(I,_,_), K = count : { tok(I,_,_) }.
.decl tp_bad(inst:number)
tp_bad(I) :- tp_len(I,K), K != 12.
tp_bad(I) :- tp_template(P,T), tok(I,P,X), X != T.
tp_bad(I) :- tok(I,4,A), tok(I,7,B), A != B.
tp_bad(I) :- tok(I,2,A), tok(I,9,B), A != B.
.decl tp_pair(inst:number,f:number,r:number)
tp_pair(I,F,0) :- tp_len(I,_), !tp_bad(I), tok(I,1,X), tp_filler(X,F).
tp_pair(I,F,1) :- tp_len(I,_), !tp_bad(I), tok(I,2,X), tp_filler(X,F).
tp_pair(I,F,2) :- tp_len(I,_), !tp_bad(I), tok(I,4,X), tp_filler(X,F).
.decl tp_full(inst:number)
tp_full(I) :- tp_pair(I,_,0), tp_pair(I,_,1), tp_pair(I,_,2).
""",
    "COPY": """// Role parse from tok facts: role = distance from the end; every token must be a pool filler.
.decl tp_len(inst:number,k:number)
tp_len(I,K) :- tok(I,_,_), K = count : { tok(I,_,_) }.
.decl tp_bad(inst:number)
tp_bad(I) :- tok(I,_,X), !tp_filler(X,_).
tp_bad(I) :- tp_len(I,K), K > %d.
.decl tp_pair(inst:number,f:number,r:number)
tp_pair(I,F,R) :- tp_len(I,K), !tp_bad(I), tok(I,P,X), tp_filler(X,F), R = K - 1 - P.
.decl tp_full(inst:number)
tp_full(I) :- tp_len(I,_), !tp_bad(I).
""" % COPY_RMAX,
}

SCORE = """// Weighted score: total(v) = bias(v) + sum over parsed pairs of w(v,f,r); argmax over the candidate set, margin.
.decl tp_total(inst:number,v:number,s:number)
tp_total(I,V,S) :- tp_full(I), tp_bias(V,B), S0 = sum X : { tp_pair(I,F,R), tp_w(V,F,R,X) }, S = S0 + B.
.decl tp_best(inst:number,m:number)
tp_best(I,M) :- tp_full(I), M = max S : { tp_total(I,_,S) }.
.decl tp_top(inst:number,v:number)
tp_top(I,V) :- tp_best(I,M), V = min V2 : { tp_total(I,V2,M) }.
.decl tp_second(inst:number,m:number)
tp_second(I,M2) :- tp_top(I,V), M2 = max S : { tp_total(I,V2,S), V2 != V }.
.decl tp_margin(inst:number,g:number)
tp_margin(I,G) :- tp_best(I,M), tp_second(I,M2), G = M - M2.
.decl tp_decide(inst:number,out:number)
tp_decide(I,V) :- tp_top(I,V), tp_margin(I,G), tp_theta(T), G >= T.
"""


def tpr_program(task, theta, meta):
    """The compiled TPR as one Soufflé program. Weights are INPUT relations (tp_w, tp_bias, tp_filler) supplied as
    fact files: Soufflé-only at runtime, no Python/weights/unembedding; inline facts would make the program huge."""
    L = ["// rosetta · compiled TPR (pil experiments/compile_tpr.py), fixed point scale 2^16.",
         ".decl tp_filler(id:number,f:number) .input tp_filler",
         ".decl tp_w(v:number,f:number,r:number,x:number) .input tp_w",
         ".decl tp_bias(v:number,b:number) .input tp_bias",
         ".decl tp_theta(t:number)", f"tp_theta({theta}).", ".decl tp_template(pos:number,id:number)"]
    L += [f"tp_template({p},{t})." for p, t in meta.get("template", {}).items()]
    return "\n".join(L) + "\n" + PARSE[task] + SCORE


def tpr_facts(tpr_dir, prefix=""):
    """Weight facts for the TPR layer, keyed by the (possibly namespaced) relation names."""
    d = Path(tpr_dir)
    return {f"{prefix}tp_w": (d / "w.facts").read_text(), f"{prefix}tp_bias": (d / "bias.facts").read_text(),
            f"{prefix}tp_filler": (d / "filler.facts").read_text()}


def tok_facts(rows):
    return "".join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p, t in enumerate(r["ctx"]))


def run_equiv_with(candidate, rows, refs, extra, *, evidence_dir, provenance):
    """oracle.run_equiv with extra input facts (the TPR weights): same dl/equiv.dl verdict, same evidence."""
    staged = {"domain": "".join(f"{r['id']}\n" for r in rows), "tok": tok_facts(rows),
              "ref": "".join(f"{r['id']}\t{refs[r['id']]}\n" for r in rows), **extra}
    result = check(DL / "equiv.dl", candidate, staged,
                   {name: int for name in ("ndomain", "ncover", "nmiss", "nuncov", "nmissing", "ninvalid")},
                   evidence_dir=evidence_dir, provenance=provenance)
    relations = result.pop("relations", {})
    result["mismatches"] = [tuple(map(int, row)) for row in relations.get("mismatch", [])]
    return result


def freeze_with(out, candidate, rows, extra, provenance):
    """benchmark_guarded_copy.freeze_domain with extra input facts: dl/guard_domain.dl over tok facts only."""
    staged = {"tok": tok_facts(rows), "sample": "".join(f"{r['id']}\n" for r in rows), **extra}
    result = check(DL / "guard_domain.dl", candidate, staged, {"ndomain": int},
                   evidence_dir=out / "certificate-evidence",
                   provenance={**provenance, "stage": "domain before test references"})
    if not result["certified"]:
        raise ValueError(f"invalid firing domain: {result}")
    domain = sorted(int(row[0]) for row in result.pop("relations")["domain"])
    return domain, result


def namespace(text, prefix):
    """Prefix every relation the text declares (except tok) so layers can share one program."""
    names = sorted(set(re.findall(r"\.decl\s+(\w+)", text)) - {"tok"}, key=len, reverse=True)
    for name in names:
        text = re.sub(rf"\b{name}\b", prefix + name, text)
    return text


def svo_idiom(whitelist):
    return ("// SVO copy-subject idiom: output the token at position 1 iff the verb (position 2) is whitelisted.\n"
            ".decl verb_ok(v:number)\n" + "".join(f"verb_ok({x}).\n" for x in sorted(whitelist)) +
            ".decl cdecide(inst:number,out:number)\n"
            "cdecide(I,S) :- tok(I,1,S), tok(I,2,V), verb_ok(V).\n")


def router(ngram_dl, idiom_dl, tpr_dl):
    """Layer order n-gram -> idiom -> TPR; every layer is namespaced; the router itself is three rules."""
    tp = tpr_dl.replace("tp_decide", "cdecide")
    return ("// rosetta · layered circuit: n-gram -> idiom -> compiled TPR -> abstain. Soufflé-only.\n"
            + namespace(ngram_dl, "ng_") + "\n" + namespace(idiom_dl, "id_") + "\n" + namespace(tp, "tc_") +
            "\n.decl ng_any(inst:number)\nng_any(I) :- ng_cdecide(I,_).\n"
            ".decl id_any(inst:number)\nid_any(I) :- id_cdecide(I,_).\n"
            ".decl layer(inst:number,l:number,out:number)\n"
            "layer(I,1,T) :- ng_cdecide(I,T).\n"
            "layer(I,2,T) :- id_cdecide(I,T), !ng_any(I).\n"
            "layer(I,3,T) :- tc_cdecide(I,T), !ng_any(I), !id_any(I).\n"
            ".decl cdecide(inst:number,out:number)\ncdecide(I,T) :- layer(I,_,T).\n")


def souffle_relation(program, rows, relation, extra_facts=None):
    """Run a standalone program over tok facts of `rows` and return one output relation as tuples."""
    with tempfile.TemporaryDirectory(prefix="compiled-tpr-") as raw:
        d = Path(raw)
        (d / "circuit.dl").write_text(program)
        (d / "run.dl").write_text('.decl tok(inst:number,pos:number,id:number)\n.input tok\n#include "circuit.dl"\n'
                                  f".output {relation}\n")
        (d / "tok.facts").write_text(tok_facts(rows))
        for name, text in (extra_facts or {}).items():
            (d / f"{name}.facts").write_text(text)
        subprocess.run(["souffle", "run.dl", "-F", ".", "-D", ".", "-I", "."], cwd=d, check=True,
                       capture_output=True, text=True)
        return [tuple(map(int, line.split("\t"))) for line in (d / f"{relation}.csv").read_text().splitlines()]


# ---------------------------------------------------------------- guards (dev only)

def ngram_dev_filter(rules_path, train, dev, refs):
    """Keep n-gram rules that fire on >= 1 dev context and are right on all of them (longest-match routing)."""
    text = Path(rules_path).read_text()
    facts_re = re.compile(r"^gram(\d+)\(([-\d,]+)\)\.$", re.M)
    rules = {}
    for n, args in facts_re.findall(text):
        vals = list(map(int, args.split(",")))
        rules[tuple(vals[:-1])] = vals[-1]
    by_rule = {}
    for row in dev:
        hit = next((suf for k in sorted({len(s) for s in rules}, reverse=True)
                    if (suf := tuple(row["ctx"][-k:])) in rules), None)
        if hit is not None:
            by_rule.setdefault(hit, []).append(rules[hit] == refs[row["id"]])
    return {s: rules[s] for s, ok in by_rule.items() if all(ok)}, len(rules)


def svo_whitelist(dev, refs):
    stats = {}
    for row in dev:
        stats.setdefault(row["verb"], []).append(refs[row["id"]] == row["subject"])
    return {vb for vb, ok in stats.items() if len(ok) >= 3 and all(ok)}


THETA_DL = """// theta = 1 + the largest compiled-TPR margin among dev ERRORS (fire iff margin >= theta); 0 if no errors.
.decl ref(inst:number,out:number) .input ref
.decl err_margin(g:number)
err_margin(G) :- tp_top(I,V), ref(I,R), V != R, tp_margin(I,G).
.decl theta(t:number) .output theta
theta(T) :- M = max G : { err_margin(G) }, T = M + 1.
theta(0) :- !err_margin(_).
.decl certified() .output certified
certified() :- theta(_).
"""


def select_theta(task, tpr_dir, meta, dev, refs, out):
    staged = {"tok": tok_facts(dev), "ref": "".join(f"{r['id']}\t{refs[r['id']]}\n" for r in dev),
              **tpr_facts(tpr_dir)}
    verifier = out / "theta.dl"
    verifier.write_text('.decl tok(inst:number,pos:number,id:number) .input tok\n#include "circuit.dl"\n' + THETA_DL)
    candidate = out / "tpr_unguarded.dl"
    candidate.write_text(tpr_program(task, 0, meta))  # unguarded: margins are >= 0; THETA_DL reads tp_margin
    result = check(verifier, candidate, staged, {}, evidence_dir=out / "certificate-evidence",
                   provenance={"stage": "theta selection on dev"})
    return int(result["relations"]["theta"][0][0])


# ---------------------------------------------------------------- run

def cmd_run(args):
    out = Path(args.out).resolve()
    provenance = {"source": "fieldrun", "bundle_sha256": {s: sha256(args.bundle + s)
                                                         for s in (".fieldrun.bin", ".fieldrun.json")}}
    summary = dict(tag="empirical", tag_note="aggregate numbers are empirical; each certified/nmiss is a dl/equiv.dl query result", prereg="docs/compiled-tpr-prereg.md", provenance=provenance, tasks={})
    for task in ("SVO", "COPY"):
        tdir = out / task
        data = json.loads((tdir / "dataset.json").read_text())
        rows, meta = data["rows"], data["meta"]
        tpr_dir = tdir / "tpr"
        tmeta = json.loads((tpr_dir / "meta.json").read_text())
        part = lambda p: [r for r in rows if r["part"] == p]  # noqa: E731
        train, dev = part("train"), part("dev")
        test_all = part("test")
        seen = set(map(tuple, tmeta["train_pairs_seen"]))
        test = [r for r in test_all if all(tuple(p) in seen for p in pairs_of(task, r, meta))]
        refs = {}
        for r in train + dev:
            refs[r["id"]] = serve_decide(args.port, r["ctx"])
        # ---- layers and guards: train + dev only
        ngram_all, n_rules, _ = learn_baseline(train, refs, tdir)
        kept, _ = ngram_dev_filter(ngram_all, train, dev, refs)
        from minimize import emit
        (tdir / "ngram").mkdir()
        emit(str(tdir / "ngram" / "circuits.dl"), kept, False, {}, "dev-filtered n-gram")
        ngram_dl = (tdir / "ngram" / "circuits.dl").read_text()
        if task == "SVO":
            whitelist = svo_whitelist(dev, refs)
            idiom_dl = svo_idiom(whitelist)
        else:
            whitelist = None
            idiom_dl = COPY_IDIOM.read_text()
        theta = select_theta(task, tpr_dir, meta, dev, refs, tdir)
        tpr_dl = tpr_program(task, theta, meta)
        extra = tpr_facts(tpr_dir, prefix="tc_")
        (tdir / "circuits.dl").write_text(router(ngram_dl, idiom_dl, tpr_dl))
        candidate = tdir / "circuits.dl"
        # ---- parse parity: Datalog role parse vs the pairs pil trained on (train contexts)
        parsed = souffle_relation(tpr_dl, train, "tp_pair", tpr_facts(tpr_dir))
        dl_pairs = {(i, f, r) for i, f, r in parsed}
        py_pairs = {(row["id"], *p) for row in train for p in pairs_of(task, row, meta)}
        if dl_pairs != py_pairs:
            raise ValueError(f"{task}: Datalog/Python parse parity fails on train ({len(dl_pairs ^ py_pairs)} pairs)")
        # ---- freeze test firing domains BEFORE any test reference
        layers = souffle_relation((tdir / "circuits.dl").read_text(), test, "layer", extra)
        domain, audit = freeze_with(tdir, candidate, test, extra, provenance)
        frozen = dict(domain=domain, layers=layers, circuit_sha256=sha256(candidate),
                      dataset_sha256=sha256(tdir / "dataset.json"))
        (tdir / "frozen.json").write_text(json.dumps(frozen) + "\n")
        frozen_hash = sha256(tdir / "frozen.json")
        for r in test:
            refs[r["id"]] = serve_decide(args.port, r["ctx"])
        if sha256(tdir / "frozen.json") != frozen_hash or sha256(candidate) != frozen["circuit_sha256"]:
            raise ValueError("frozen inputs changed during reference collection")
        (tdir / "references.json").write_text(json.dumps(refs) + "\n")
        by_id = {r["id"]: r for r in test}
        layer_of = {i: (lay, t) for i, lay, t in layers}
        certs = {}
        for lname, lid in (("ngram", 1), ("idiom", 2), ("tpr", 3), ("composite", None)):
            ids = sorted(i for i, (lay, _) in layer_of.items() if lid is None or lay == lid)
            if not ids:
                certs[lname] = dict(ndomain=0, certified=None, nmiss=0)
                continue
            res = run_equiv_with(candidate, [by_id[i] for i in ids], refs, extra,
                                 evidence_dir=tdir / "certificate-evidence",
                                 provenance={**provenance, "layer": lname, "instance_ids": ids})
            certs[lname] = dict(ndomain=res["ndomain"], nmiss=res["nmiss"], nuncov=res["nuncov"],
                                certified=res["certified"],
                                mismatches=[list(m) for m in res["mismatches"]])
        # ---- residual metric
        def category(r):
            ref = refs[r["id"]]
            if task == "SVO":
                return "R_obj" if ref == r["object"] else ("R_other" if ref not in r["ctx"] else "R_in_sentence")
            return "R_ctx" if ref in r["ctx"] else "R_out"
        residual = [r for r in test if not (r["id"] in layer_of and layer_of[r["id"]][0] in (1, 2)
                                            and layer_of[r["id"]][1] == refs[r["id"]])]
        tpr_clean = certs["tpr"]["certified"] is True
        split = {}
        for r in residual:
            c = category(r)
            s = split.setdefault(c, dict(n=0, tpr_decides=0, tpr_agrees=0))
            s["n"] += 1
            if r["id"] in layer_of and layer_of[r["id"]][0] == 3:
                s["tpr_decides"] += 1
                s["tpr_agrees"] += layer_of[r["id"]][1] == refs[r["id"]]
        add = sum(s["tpr_decides"] for s in split.values()) if tpr_clean else 0
        q1 = tpr_clean and add >= 10 and add >= 0.05 * len(residual)
        summary["tasks"][task] = dict(
            n_test=len(test), test_dropped_unseen_pair=len(test_all) - len(test),
            ngram_rules_learned=n_rules, ngram_rules_kept=len(kept),
            idiom_whitelist=sorted(whitelist) if whitelist is not None else "frozen copy guard",
            theta=theta, candidates=len(tmeta["candidates"]), parse_parity=dl_pairs == py_pairs,
            layer_counts={n: sum(1 for lay, _ in layer_of.values() if lay == k)
                          for n, k in (("ngram", 1), ("idiom", 2), ("tpr", 3))},
            abstain=len(test) - len(layer_of), certificates=certs, residual_n=len(residual),
            residual_split=split, additional_certified=add,
            additional_certified_fraction=add / len(residual) if residual else None,
            Q1_passes=bool(q1), frozen_sha256=frozen_hash, domain_audit_evidence=str(audit["evidence"]))
        print(json.dumps({task: {k: v for k, v in summary["tasks"][task].items() if k != "certificates"}},
                         indent=1, default=str), flush=True)
    passes = [t["Q1_passes"] for t in summary["tasks"].values()]
    summary["Q1_overall"] = "yes" if all(passes) else ("partial" if any(passes) else "no")
    (out / "report.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("Q1 overall:", summary["Q1_overall"])


def pairs_of(task, row, meta):
    """Python twin of the Datalog role parse (pairs as (filler, role)); fillers by token id unless as_index."""
    fidx = {t: i for i, t in enumerate(meta["fillers"])}
    if task == "SVO":
        toks = [(row["ctx"][1], 0), (row["ctx"][2], 1), (row["ctx"][4], 2)]
    else:
        n = len(row["ctx"])
        toks = [(t, n - 1 - p) for p, t in enumerate(row["ctx"])]
    return [(fidx[t], r) for t, r in toks]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("out")
    g.add_argument("--tokenizer", required=True, help="fieldrun gpt2.tokenizer.json")
    g.add_argument("--smoke", action="store_true", help="bug check: 600 contexts, seeds + 1000; NOT results")
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--bundle", required=True)
    args = ap.parse_args()
    {"generate": cmd_generate, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
