"""Protocol tests with authored mechanisms; never study observations."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from benchmark_guarded_copy import guard_program
from certificate import check
from native_trace import DL, emit, grammar, program
from native_trace import write_json
from certificate import sha256, replay
from run_native_trace import select, freeze, validate_freeze, evaluate

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")


def run_runtime(out, contexts, selected):
    emit(out, selected)
    (out / "tok.facts").write_text("".join(f"{i}\t{p}\t{t}\n"
        for i, ctx in enumerate(contexts) for p, t in enumerate(ctx)))
    (out / "domain.facts").write_text("".join(f"{i}\n" for i in range(len(contexts))))
    subprocess.run(["souffle", "run.dl", "-F", ".", "-D", "."], cwd=out,
                   capture_output=True, text=True, check=True)
    return {(int(i), int(t)) for i,t in (line.split() for line in (out / "cdecide.csv").read_text().splitlines())}


def test_standalone_router_resolves_disagreement_and_covers_empty_abstention(tmp_path):
    bases, candidates = grammar()
    assert len(bases) == 24 and len(candidates) == 1344
    # No token labels, references, trace facts or code outside this directory.
    # Nearest 1-token source copies 9; longest match copies 7.
    contexts = [[1,2,3,7,8,3,9,1,2,3], [], [1,2,3]]
    nearest = next(c[0] for c in candidates if c[1:] == (13,0,0))
    longest = next(c[0] for c in candidates if c[1:] == (17,0,0))
    assert run_runtime(tmp_path / "nearest", contexts, [(1,nearest),(2,longest)]) == {(0,9)}
    assert run_runtime(tmp_path / "longest", contexts, [(1,longest),(2,nearest)]) == {(0,7)}
    assert (tmp_path / "nearest/abstain.csv").read_text().splitlines() == ["1", "2"]
    assert run_runtime(tmp_path / "empty", contexts, []) == set()
    assert (tmp_path / "empty/abstain.csv").read_text().splitlines() == ["0", "1", "2"]


def test_native_trace_features_use_class_means_not_unit_scale(tmp_path):
    # Unit 0 has huge, irrelevant scale; unit 1 exactly discriminates correctness.
    candidate = tmp_path / "guard.dl"
    candidate.write_text('.decl cdecide(inst:number,out:number)\ncdecide(I,T) :- tok(I,0,T).\n')
    staged = {"tok": "0\t0\t1\n1\t0\t1\n2\t0\t1\n3\t0\t1\n",
              "ref": "0\t1\n1\t1\n2\t2\n3\t2\n", "sample": "0\n1\n2\n3\n",
              "unit": "0\n1\n", "observation": "".join(
                  f"{i}\t{n}\t{v}\n" for n, vals in enumerate(((100,-100,100,-100),(2,4,-2,-4)))
                  for i,v in enumerate(vals))}
    result = check(DL / "native_trace_features.dl", candidate, staged, {})
    assert result["certified"], result
    chosen = result["relations"]["chosen"]
    assert len(chosen) == 1 and chosen[0][0] == "1"
    assert float(chosen[0][2]) == 0 and chosen[0][3] == "1"
    assert float(chosen[0][1]) == pytest.approx(3.6)
    # Missing observations fail closed rather than selectively dropping a bad row.
    staged["observation"] = staged["observation"].replace("3\t1\t-4\n", "")
    assert not check(DL / "native_trace_features.dl", candidate, staged, {})["certified"]


def selection_fixture(tmp_path):
    # Thirteen distractor rules tie a transferable rule on train. The 12-slot
    # search budget misses the good rule under behavior-only and shuffled traces.
    candidate = tmp_path / "hypotheses.dl"
    candidate.write_text('.decl candidate(c:number,b:number,a:number,d:number)\n' +
        "".join(f"candidate({c},0,0,0).\n" for c in range(1,15)) +
        '.decl nt_candidate(c:number,i:number,t:number)\n'
        'nt_candidate(C,I,T) :- candidate(C,_,_,_), C<14, tok(I,0,T), tok(I,1,2).\n'
        'nt_candidate(14,I,T) :- tok(I,0,T), tok(I,1,1).\n')
    facts = {k: "" for k in ("tok", "ref", "sample", "pair", "trace", "permutation")}
    for part_idx, part in enumerate(("train", "validation")):
        for marker in (1,2):
            for g in range(4):
                for edit in (0,1):
                    i = part_idx*16 + (marker-1)*8 + g*2 + edit
                    t = 100 + i
                    facts["tok"] += f"{i}\t0\t{t}\n{i}\t1\t{marker}\n"
                    reference = t if part == "train" or marker == 1 else t+1000
                    facts["ref"] += f"{i}\t{reference}\n"
                    facts["sample"] += f"{i}\t{part}-{marker}-{g}\t{part}\tinput-{i}\n"
                    if edit:
                        facts["pair"] += f"{i-1}\t{i}\n"
                    if part == "train":
                        facts["trace"] += f"{i}\t0\t{1 if marker==1 else -1}\n"
                        facts["permutation"] += f"{i}\t{(i+8)%16}\n"
    facts["feature"] = "0\t0\t1\n"
    return candidate, facts


def test_trace_prior_and_shuffle_discriminate_under_equal_budget(tmp_path):
    candidate, facts = selection_fixture(tmp_path)
    result = check(DL / "native_trace_select.dl", candidate, facts, {})
    assert result["certified"], result
    proposed = result["relations"]["proposed"]
    assert [sum(int(a)==arm for a,_,_ in proposed) for arm in range(3)] == [12,12,12]
    assert result["relations"]["admitted"] == [["1", "1", "14"]]
    assert all(int(c) != 14 for a,_,c in proposed if a in ("0", "2"))


@pytest.mark.parametrize("defect", ["missing_reference", "leaked_test", "bad_permutation", "missing_trace"])
def test_selection_integrity_rejects_incomplete_or_leaked_data(tmp_path, defect):
    candidate, facts = selection_fixture(tmp_path)
    if defect == "missing_reference":
        facts["ref"] = facts["ref"].replace("0\t100\n", "")
    elif defect == "leaked_test":
        facts["sample"] = facts["sample"].replace("\tvalidation\t", "\ttest\t")
    elif defect == "bad_permutation":
        facts["permutation"] = facts["permutation"].replace("0\t8\n", "0\t9\n")
    else:
        facts["trace"] = facts["trace"].replace("0\t0\t1\n", "")
    result = check(DL / "native_trace_select.dl", candidate, facts, {})
    assert not result["certified"], result
    assert not result.get("error"), result  # semantic rejection, not a broken program


def test_frozen_study_inputs_are_group_and_token_disjoint():
    root = ROOT / "reference/benchmarks/qwen25_native_trace"
    rows = json.loads((root / "dataset.json").read_text())
    discovery = json.loads((root / "discovery.json").read_text())
    vocabs = [{t for r in discovery for t in r["ctx"]}]
    groups = [{r["group"] for r in discovery}]
    for part in ("train", "validation", "test"):
        batch = [r for r in rows if r["part"] == part]
        assert len(batch) == 114 and sum(r["negative"] for r in batch) == 18
        vocabs.append({t for r in batch for t in r["ctx"]})
        groups.append({r["group"] for r in batch})
    for values in (vocabs, groups):
        for i, a in enumerate(values):
            for b in values[i+1:]:
                assert not a & b


def control_run(out):
    from benchmark_copy_generalization import generate
    out.mkdir()
    rows = []
    used = set()
    for seed,part in ((901,"train"),(902,"validation"),(903,"test")):
        batch = generate(seeds=(seed,), lengths=(8,), groups=2, exclude=used)
        mapping = {r["id"]:len(rows)+n for n,r in enumerate(batch)}
        for row in batch:
            row["negative"] = row["part"] == "off_domain"
            row["part"] = part
            row["id"] = mapping[row["id"]]
            if "base" in row:
                row["base"] = mapping[row["base"]]
            used.update(row["ctx"])
        rows.extend(batch)
    write_json(out / "dataset.json", rows)
    write_json(out / "discovery.json", [])
    (out / "hypotheses.dl").write_text(program())
    write_json(out / "protocol.json", {"dataset_sha256":sha256(out / "dataset.json"),
        "discovery_sha256":sha256(out / "discovery.json"), "hypotheses_sha256":sha256(out / "hypotheses.dl"),
        "prereg_sha256":sha256(ROOT / "docs/native-trace-prereg.md")})
    development = [r for r in rows if r["part"] != "test"]
    write_json(out / "development-references.json", {str(r["id"]): -1 if r["negative"] else r["expected"] for r in development})
    (out / "development-observations.jsonl").write_text("".join(json.dumps({"id":r["id"],"observations":[]})+"\n" for r in development))
    write_json(out / "features.json", [])
    write_json(out / "diagnostic-summary.json", {"certified":True,"fixture":True})
    select(out)
    freeze(out)
    return rows


@pytest.mark.parametrize("corrupt", [False,True])
def test_full_pipeline_preserves_final_counterexamples_and_abstentions(tmp_path, corrupt):
    out = tmp_path / "control"
    rows = control_run(out)
    frozen = validate_freeze(out)
    assert not (out / "final-references.json").exists()
    domain = frozen["artifacts"]["trace"]["domain"]
    assert domain  # do not pass vacuously through empty selection
    refs = {str(r["id"]): -1 if r["negative"] else r["expected"] for r in rows if r["part"] == "test"}
    if corrupt:
        refs[str(domain[0])] = -999
    write_json(out / "final-references.json", refs)
    evaluate(out)
    result = json.loads((out / "report.json").read_text())
    cert = result["certificates"]["trace"]["firing"]
    assert cert["certified"] is (not corrupt)
    assert cert["nmiss"] == int(corrupt)
    assert replay(out / cert["evidence"])["certified"] is (not corrupt)
    assert not result["certificates"]["trace"]["full"]["certified"]  # negatives abstain
    assert not result["summary"]["relations"]["improved"]  # identical arms are not improvement
    if corrupt:
        assert any(int(i)==domain[0] and int(ref)==-999 for p,i,ref,_ in result["summary"]["relations"]["counterexample"] if p=="trace")
    candidate = out / frozen["artifacts"]["trace"]["path"]
    candidate.write_text(candidate.read_text()+"\n// modification after freeze\n")
    with pytest.raises(ValueError,match="frozen file changed"):
        validate_freeze(out)


def test_diagnostic_plan_patches_opposite_class_and_detects_broken_identity(tmp_path):
    empty = tmp_path / "empty.dl"
    empty.write_text("")
    plan = check(DL / "native_trace_diagnostic_plan.dl", empty,
        {"label":"0\t0\n1\t1\n", "chosen_unit":"7\t1\t4\n",
         "product":"0\t7\t1\t-2\n1\t7\t1\t3\n"}, {})
    assert plan["certified"], plan
    assert ["0","7","1","swap","1","3"] in plan["relations"]["plan"]
    assert ["1","7","2","control_zero","1","0"] in plan["relations"]["plan"]
    good = {"result":"0\t7\t1\tidentity\t8\t8\t1.5\t1.5\t2\t2\n", "observation":"0\t7\t1\t-1\t-2\t0.5\n"}
    result = check(DL / "native_trace_diagnostic.dl", empty, good, {})
    assert result["certified"], result
    assert result["relations"]["regime"] == [["7","1","--","1"]]
    good["result"] = good["result"].replace("1.5\t1.5", "1.5\t1.6")
    result = check(DL / "native_trace_diagnostic.dl", empty, good, {})
    assert not result["certified"] and not result.get("error"), result
