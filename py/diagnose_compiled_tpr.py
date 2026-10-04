"""POST-HOC diagnostic for docs/compiled-tpr-prereg.md (not a pre-registered result).

The pre-registered router lets the idiom pre-empt the compiled TPR. This asks, with no re-selection: on the residual
contexts (where the deployed n-gram + idiom layers do not decide GPT-2's answer), what does the compiled TPR layer
alone decide, with its own dev-selected theta? Decisions are Soufflé query results; numbers are `empirical`.

    python3 py/diagnose_compiled_tpr.py reference/benchmarks/gpt2_compiled_tpr .../gpt2.tokenizer.json
"""
import json
from pathlib import Path
import sys

from benchmark_compiled_tpr import pairs_of, souffle_relation, tpr_facts, tpr_program


def main():
    out = Path(sys.argv[1])
    report = json.loads((out / "report.json").read_text())
    vocab = {i: t for t, i in json.loads(Path(sys.argv[2]).read_text())
             ["model"]["vocab"].items()}
    diag = {}
    for task, rep in report["tasks"].items():
        tdir = out / task
        data = json.loads((tdir / "dataset.json").read_text())
        rows, meta = data["rows"], data["meta"]
        refs = {int(k): v for k, v in json.loads((tdir / "references.json").read_text()).items()}
        frozen = json.loads((tdir / "frozen.json").read_text())
        layer_of = {i: (lay, t) for i, lay, t in frozen["layers"]}
        tmeta = json.loads((tdir / "tpr" / "meta.json").read_text())
        seen = set(map(tuple, tmeta["train_pairs_seen"]))
        test = [r for r in rows if r["part"] == "test" and all(tuple(p) in seen for p in pairs_of(task, r, meta))]
        # the pre-registered residual: contexts the deployed n-gram (1) + idiom (2) layers do not decide correctly
        residual = [r for r in test if not (r["id"] in layer_of and layer_of[r["id"]][0] in (1, 2)
                                            and layer_of[r["id"]][1] == refs[r["id"]])]
        program = tpr_program(task, rep["theta"], meta)
        tp = {i: t for i, t in souffle_relation(program, residual, "tp_decide", tpr_facts(tdir / "tpr"))}
        unguarded = {i: t for i, t in souffle_relation(tpr_program(task, 0, meta), residual, "tp_decide",
                                                       tpr_facts(tdir / "tpr"))}
        cats = {}
        for r in residual:
            ref = refs[r["id"]]
            if task == "SVO":
                c = "R_obj" if ref == r["object"] else ("R_other" if ref not in r["ctx"] else "R_in_sentence")
            else:
                c = ("R_ctx" if ref in r["ctx"] else "R_out") + "/" + r["layout"]
            s = cats.setdefault(c, dict(n=0, guarded_fires=0, guarded_agrees=0, unguarded_agrees=0))
            s["n"] += 1
            s["unguarded_agrees"] += unguarded.get(r["id"]) == ref
            if r["id"] in tp:
                s["guarded_fires"] += 1
                s["guarded_agrees"] += tp[r["id"]] == ref
        mism = [dict(id=i, ref=vocab.get(rf), tpr=vocab.get(c)) for i, rf, c in rep["certificates"]["tpr"].get(
            "mismatches") or []]
        diag[task] = dict(residual=len(residual), by_category=cats, tpr_mismatches_decoded=mism)
    (out / "posthoc_diagnostic.json").write_text(json.dumps(dict(post_hoc=True, tag="empirical", tasks=diag),
                                                           indent=2) + "\n")
    print(json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()
