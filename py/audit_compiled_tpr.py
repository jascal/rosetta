"""POST-RUN audit for docs/compiled-tpr-prereg.md: recomputes, from the stored artifacts, what the frozen run did not
record. It re-selects nothing: datasets, circuits, frozen domains, θ and references are all read back from disk.

    gunzip -k reference/benchmarks/gpt2_compiled_tpr/COPY/tpr/w.facts.gz
    .venv/bin/python py/audit_compiled_tpr.py reference/benchmarks/gpt2_compiled_tpr

1. Full certificate mismatch lists (the run stored the first 50), re-derived by dl/equiv.dl on the frozen domains.
2. Parse parity over the FULL train split (the run checked train[:500] and did not gate); fails on any mismatch.
3. θ re-selected with the unguarded program at tp_theta(0) (the run used -2**60): must equal the stored θ.
4. HF GPT-2 vs fieldrun disagreements (pre-reg §1), per split. HF is what pil trained on; certificates are fieldrun.
5. Subject-copy rate per split (the outcome's idiom accuracy figure).
"""
import json
from pathlib import Path
import sys
import tempfile

from benchmark_compiled_tpr import (pairs_of, run_equiv_with, select_theta, souffle_relation, tpr_facts,
                                    tpr_program)


def hf_decisions(rows):
    """Hugging Face GPT-2 greedy next token per context (same forward as pil experiments/compile_tpr.py)."""
    import torch
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    out = {}
    with torch.no_grad():
        for s in range(0, len(rows), 256):
            chunk = rows[s:s + 256]
            length = max(len(x["ctx"]) for x in chunk)
            ids = torch.zeros(len(chunk), length, dtype=torch.long)
            att = torch.zeros(len(chunk), length, dtype=torch.long)
            for i, x in enumerate(chunk):
                ids[i, :len(x["ctx"])] = torch.tensor(x["ctx"])
                att[i, :len(x["ctx"])] = 1
            h = model.transformer(input_ids=ids, attention_mask=att).last_hidden_state
            u = h[torch.arange(len(chunk)), att.sum(1) - 1]
            for x, d in zip(chunk, model.lm_head(u).argmax(-1).tolist()):
                out[x["id"]] = d
    return out


def main():
    out = Path(sys.argv[1]).resolve()
    report = json.loads((out / "report.json").read_text())
    audit = dict(post_run_audit=True, reselects_nothing=True, tasks={})
    for task, rep in report["tasks"].items():
        tdir = out / task
        data = json.loads((tdir / "dataset.json").read_text())
        rows, meta = data["rows"], data["meta"]
        refs = {int(k): v for k, v in json.loads((tdir / "references.json").read_text()).items()}
        frozen = json.loads((tdir / "frozen.json").read_text())
        tpr_dir = tdir / "tpr"
        if not (tpr_dir / "w.facts").exists():
            raise SystemExit(f"{tpr_dir}/w.facts missing: gunzip -k {tpr_dir}/w.facts.gz first")
        part = {p: [r for r in rows if r["part"] == p] for p in ("train", "dev", "test")}
        by_id = {r["id"]: r for r in rows}
        a = {}
        # 1. full mismatch lists on the frozen per-layer domains
        candidate = tdir / "circuits.dl"
        extra = tpr_facts(tpr_dir, prefix="tc_")
        layer_of = {i: lay for i, lay, _ in frozen["layers"]}
        a["certificates"] = {}
        with tempfile.TemporaryDirectory(prefix="tpr-audit-") as ev:
            for lname, lid in (("ngram", 1), ("idiom", 2), ("tpr", 3), ("composite", None)):
                ids = sorted(i for i, lay in layer_of.items() if lid is None or lay == lid)
                if not ids:
                    continue
                res = run_equiv_with(candidate, [by_id[i] for i in ids], refs, extra, evidence_dir=Path(ev),
                                     provenance={"stage": "post-run audit", "layer": lname})
                stored = rep["certificates"][lname]
                if (res["ndomain"], res["nmiss"], res["certified"]) != (stored["ndomain"], stored["nmiss"],
                                                                        stored["certified"]):
                    raise ValueError(f"{task}/{lname}: re-derived verdict differs from report.json")
                a["certificates"][lname] = dict(
                    ndomain=res["ndomain"], nmiss=res["nmiss"], certified=res["certified"],
                    mismatches=[dict(id=i, ref=r, out=o, layout=by_id[i].get("layout"))
                                for i, r, o in sorted(res["mismatches"])])
        # 2. parse parity, full train, gated
        parsed = souffle_relation(tpr_program(task, 0, meta), part["train"], "tp_pair", tpr_facts(tpr_dir))
        dl_pairs = set(parsed)
        py_pairs = {(row["id"], *p) for row in part["train"] for p in pairs_of(task, row, meta)}
        if dl_pairs != py_pairs:
            raise ValueError(f"{task}: parse parity fails on full train ({len(dl_pairs ^ py_pairs)} pairs differ)")
        a["parse_parity_full_train"] = dict(contexts=len(part["train"]), pairs=len(py_pairs), equal=True)
        # 3. θ with tp_theta(0) in the unguarded program
        with tempfile.TemporaryDirectory(prefix="tpr-theta-") as scratch:
            theta = select_theta(task, tpr_dir, meta, part["dev"], refs, Path(scratch))
        if theta != rep["theta"]:
            raise ValueError(f"{task}: theta {theta} != stored {rep['theta']}")
        a["theta_reselected_at_tp_theta_0"] = theta
        # 4. HF vs fieldrun, per split
        hf = hf_decisions(rows)
        a["hf_vs_fieldrun"] = {p: dict(n=len(rs), disagree=sum(hf[r["id"]] != refs[r["id"]] for r in rs),
                                       disagree_ids=sorted(r["id"] for r in rs if hf[r["id"]] != refs[r["id"]]))
                               for p, rs in part.items()}
        # 5. subject-copy rate (SVO only)
        if task == "SVO":
            a["subject_copy_rate"] = {p: dict(n=len(rs), subject=sum(refs[r["id"]] == r["subject"] for r in rs))
                                      for p, rs in part.items()}
        audit["tasks"][task] = a
        print(task, json.dumps({k: v for k, v in a.items() if k != "certificates"}, default=str)[:600], flush=True)
    (out / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")


if __name__ == "__main__":
    main()
