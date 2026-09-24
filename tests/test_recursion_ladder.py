"""Depth/perturbation generator and recursive Datalog certificate."""
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py"))
from recursion_ladder import certify, evaluate, exhaustive_cases, make_cases, prompt, symbolic_iterations  # noqa: E402
from recursion_extract import extract  # noqa: E402
from recursion_structural_holdout import build_cases  # noqa: E402


def _reachable(case):
    source, target = case["query"]
    frontier = deque([(source, 0)])
    seen = {source}
    while frontier:
        node, hops = frontier.popleft()
        if node == target:
            return hops
        for a, b in case["edges"]:
            if a == node and b not in seen:
                seen.add(b)
                frontier.append((b, hops + 1))
    return None


def test_depth_ladder_and_edge_perturbations():
    for topology in ("chains", "branched"):
        cases = make_cases(seed=31, depths=(1, 2, 3, 4, 6), trials=5, topology=topology)
        for positive, negative in zip(cases[::2], cases[1::2]):
            assert _reachable(positive) == positive["depth"]
            assert _reachable(negative) is None
            assert len(positive["edges"]) == len(negative["edges"])
            assert len(set(map(tuple, positive["edges"])) ^ set(map(tuple, negative["edges"]))) == 2
            assert positive["query"] == negative["query"]
        assert any(1 < c["cut_hop"] < c["depth"] for c in cases if c["positive"])


@pytest.mark.skipif(not shutil.which("souffle"), reason="requires souffle")
def test_recursive_certificate_distinguishes_model_failure():
    cases = make_cases(seed=31, depths=(1, 2, 3, 4, 6), trials=3)
    truth = certify(cases, {c["id"]: c["answer"] for c in cases})
    assert truth["certified"] and truth["ncover"] == len(cases)
    assert truth["nmiss"] == truth["nuncov"] == 0
    # A model that always says no has 50% aggregate accuracy but never follows
    # the perturbed edge; the Datalog certificate must reject it.
    rows = [{"id": c["id"], "architecture": "constant_no", "budget": 1, "answer": 0}
            for c in cases]
    measured = evaluate(cases, rows)[0]
    assert not measured["model_certificate"]["certified"]
    assert measured["model_certificate"]["nmiss"] == len(cases) // 2
    assert all(d["perturbation_follow"] == 0 for d in measured["depths"].values())


@pytest.mark.skipif(not shutil.which("souffle"), reason="requires souffle")
def test_iteration_budget_recovers_each_depth():
    cases = make_cases(seed=4, depths=(1, 2, 3, 4), trials=2)
    measured = evaluate(cases, symbolic_iterations(cases, (1, 2, 3, 4)))
    for row in measured:
        for depth, metrics in row["depths"].items():
            should_pass = int(depth) <= row["budget"]
            assert metrics["perturbation_follow"] == float(should_pass)
            if metrics["interior_edge_pairs"]:
                assert metrics["interior_edge_follow"] == float(should_pass)
        assert row["model_certificate"]["certified"] == (row["budget"] == 4)


@pytest.mark.skipif(not shutil.which("souffle"), reason="requires souffle")
def test_runtime_circuit_needs_only_graph_and_query(tmp_path):
    cases = make_cases(seed=6, depths=(1, 3), trials=1)
    facts, out = tmp_path / "facts", tmp_path / "out"
    facts.mkdir()
    out.mkdir()
    with open(facts / "edge.facts", "w") as f:
        for case in cases:
            for a, b in case["edges"]:
                f.write(f"{case['id']}\t{a}\t{b}\n")
    with open(facts / "query.facts", "w") as f:
        for case in cases:
            f.write(f"{case['id']}\t{case['query'][0]}\t{case['query'][1]}\n")
    dl = Path(__file__).resolve().parent.parent / "dl" / "reach_circuit.dl"
    subprocess.run(["souffle", str(dl), "-F", str(facts), "-D", str(out)], check=True)
    got = {int(i): int(v) for i, v in (line.split("\t") for line in
                                     (out / "cdecide.csv").read_text().splitlines())}
    assert got == {case["id"]: case["answer"] for case in cases}


@pytest.mark.skipif(not shutil.which("souffle"), reason="requires souffle")
def test_exhaustive_three_node_graph_domain():
    cases = exhaustive_cases(3)
    assert len(cases) == 384  # 2^(3*2) directed graphs × 3*2 ordered queries
    verdict = certify(cases, {c["id"]: c["answer"] for c in cases})
    assert verdict["certified"] and verdict["ncover"] == 384


def test_partial_model_budget_cannot_be_certified():
    cases = make_cases(depths=(1, 2), trials=2)
    rows = [{"id": c["id"], "architecture": "partial", "budget": 2, "answer": c["answer"]}
            for c in cases[:-1]]
    with pytest.raises(ValueError, match="omitted cases"):
        evaluate(cases, rows)


@pytest.mark.skipif(not shutil.which("souffle"), reason="requires souffle")
def test_prompt_styles_keep_independent_certificates():
    cases = make_cases(depths=(1,), trials=2)
    assert "90 -> 91" in prompt(cases[0], "fewshot")
    rows = []
    for c in cases:
        rows.append({"id": c["id"], "architecture": "sample", "budget": 1,
                     "prompt_style": "plain", "answer": c["answer"]})
        rows.append({"id": c["id"], "architecture": "sample", "budget": 1,
                     "prompt_style": "fewshot", "answer": 0})
    verdicts = {r["prompt_style"]: r for r in evaluate(cases, rows)}
    assert verdicts["plain"]["model_certificate"]["certified"]
    assert not verdicts["fewshot"]["model_certificate"]["certified"]


@pytest.mark.skipif(not shutil.which("souffle"), reason="requires souffle")
def test_datalog_selects_recursion_on_disjoint_graph_trials():
    cases = make_cases(seed=19, depths=(1, 2, 3, 4, 5), trials=10, topology="branched")
    train_ids = {c["id"] for c in cases if c["trial"] < 5}
    result = extract(cases, {c["id"]: c["answer"] for c in cases}, train_ids)
    assert result["selected"] == "recursive_reach"
    assert result["training_errors_by_candidate"]["at_most_two_hops"] > 0
    assert result["holdout_errors"] == 0
    assert result["best_fit"] == "recursive_reach"
    assert result["best_fit_holdout_errors"] == 0
    constant = extract(cases, {c["id"]: 0 for c in cases}, train_ids)
    assert constant["selected"] == "constant_no"
    assert constant["holdout_errors"] == 0
    always_yes = extract(cases, {c["id"]: 1 for c in cases}, train_ids)
    assert always_yes["selected"] == "constant_yes"
    assert always_yes["holdout_errors"] == 0


@pytest.mark.skipif(not shutil.which("souffle"), reason="requires souffle")
def test_best_fit_rule_is_reported_without_claiming_exact_selection():
    cases = make_cases(seed=23, depths=(1, 2, 3, 4), trials=6, topology="branched")
    train_ids = {c["id"] for c in cases if c["trial"] < 3}
    refs = {c["id"]: c["answer"] for c in cases}
    flipped = next(c["id"] for c in cases if c["id"] in train_ids and c["positive"])
    refs[flipped] = 0
    result = extract(cases, refs, train_ids)
    assert result["selected"] is None
    assert result["best_fit"] == "recursive_reach"
    assert result["training_errors_by_candidate"]["recursive_reach"] == 1
    assert result["best_fit_holdout_errors"] == 0


@pytest.mark.skipif(not shutil.which("souffle"), reason="requires souffle")
def test_recursive_rule_selection_generalizes_to_new_graph_sizes_and_structures():
    cases = build_cases()
    train = [row for row in cases if row["split"] == "train"]
    truth = {row["id"]: row["answer"] for row in cases}
    selected = extract(cases, truth, {row["id"] for row in train})
    assert selected["selected"] == "recursive_reach"
    assert selected["holdout_errors"] == 0
    certificate = certify(cases, truth)
    assert certificate["certified"] and certificate["nmiss"] == certificate["nuncov"] == 0
