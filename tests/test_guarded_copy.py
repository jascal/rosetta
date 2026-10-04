"""Consensus guards, causal admission, and reference-independent finite certificate domains."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from benchmark_guarded_copy import GUARDS, admit, benchmark, emit_sparse_baseline, freeze_domain, generate, write_guard  # noqa: E402
from certificate import replay  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")


def test_fresh_groups_and_all_source_interventions():
    excluded = set(range(200, 500))
    rows = generate(groups=(2, 2, 2), exclude=excluded)
    vocab = {}
    for row in rows:
        assert not set(row["ctx"]) & excluded
        vocab.setdefault(row["group"], set()).update(row["ctx"])
        if row["kind"] == "intervention":
            base = rows[row["base"]]
            differences = [p for p, (a, b) in enumerate(zip(base["ctx"], row["ctx"])) if a != b]
            assert differences == row["positions"]
            assert len(differences) == row["exposures"]
            assert row["group"] == base["group"] and row["part"] == base["part"]
            assert row["ctx"][-row["prefix"]:] == base["ctx"][-row["prefix"]:]
    for a in vocab:
        for b in vocab:
            if a != b:
                assert not vocab[a] & vocab[b]


def test_standalone_consensus_guard_and_abstentions(tmp_path):
    source = write_guard(tmp_path / "source", [GUARDS[0]])
    directory = tmp_path / "runtime"
    directory.mkdir()
    for name in ("run.dl", "circuits.dl"):
        shutil.copyfile(source.parent / name, directory / name)
    contexts = [[1, 2, 3, 7, 1, 2, 3, 7, 1, 2, 3],  # two agreeing successors -> 7
                [1, 2, 3, 7, 1, 2, 3, 8, 1, 2, 3],  # conflicting successors -> abstain
                [1, 2, 3, 7, 1, 2, 3],              # one source -> abstain
                [1, 2, 3, 7, 1, 2, 3, 7, 9, 2, 3],  # only two-token suffix matches -> abstain
                [1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 3]]  # change both sources -> 9
    (directory / "tok.facts").write_text("".join(f"{i}\t{p}\t{t}\n"
                                               for i, ctx in enumerate(contexts) for p, t in enumerate(ctx)))
    subprocess.run(["souffle", "run.dl", "-F", ".", "-D", "."], cwd=directory, check=True, capture_output=True)
    assert set((directory / "cdecide.csv").read_text().splitlines()) == {"0\t7", "4\t9"}
    assert set((directory / "abstain.csv").read_text().splitlines()) == {"1", "2", "3"}


def test_plurality_guard_resolves_one_conflicting_source_and_abstains_on_ties(tmp_path):
    contexts = [
        [1, 2, 3, 7, 1, 2, 3, 7, 1, 2, 3, 8, 1, 2, 3],  # 7 has two votes; 8 has one
        [1, 2, 3, 7, 1, 2, 3, 8, 1, 2, 3],              # tie: abstain
        [1, 2, 3, 7, 9, 1, 2, 3],                        # one source: insufficient support
    ]
    for policy, expected in (("plurality", {"0\t7"}), ("consensus", set())):
        directory = tmp_path / policy
        write_guard(directory, [(1, 3, 2)], policy=policy)
        (directory / "tok.facts").write_text("".join(
            f"{i}\t{p}\t{token}\n" for i, ctx in enumerate(contexts) for p, token in enumerate(ctx)))
        subprocess.run(["souffle", "run.dl", "-F", ".", "-D", "."], cwd=directory,
                       check=True, capture_output=True)
        assert set((directory / "cdecide.csv").read_text().splitlines()) == expected
        abstain = set((directory / "abstain.csv").read_text().splitlines())
        assert abstain == ({"1", "2"} if policy == "plurality" else {"0", "1", "2"})


def test_bounded_recency_ignores_stale_conflicting_source(tmp_path):
    context = [1, 2, 3, 8, 9, 9, 9, 9, 1, 2, 3, 7, 1, 2, 3, 7, 1, 2, 3]
    outputs = {}
    for policy, horizons in (("consensus", None), ("recent", {1: 12})):
        directory = tmp_path / policy
        write_guard(directory, [(1, 3, 2)], policy=policy, horizons=horizons)
        (directory / "tok.facts").write_text("".join(
            f"0\t{position}\t{token}\n" for position, token in enumerate(context)))
        subprocess.run(["souffle", "run.dl", "-F", ".", "-D", "."], cwd=directory,
                       check=True, capture_output=True)
        outputs[policy] = set((directory / "cdecide.csv").read_text().splitlines())
    assert outputs == {"consensus": set(), "recent": {"0\t7"}}


def test_ranked_copy_prefers_specificity_then_nearest_source(tmp_path):
    # Case 0: the nearer occurrence matches only three tokens; the older
    # occurrence's four-token match wins despite its more distant position.
    # Case 1: both sources match four tokens, so the nearest one wins.
    contexts = [
        [1, 2, 3, 4, 7, 2, 3, 4, 8, 1, 2, 3, 4],
        [1, 2, 3, 4, 7, 9, 9, 1, 2, 3, 4, 8, 1, 2, 3, 4],
    ]
    directory = tmp_path / "ranked"
    write_guard(directory, [(1, 4, 1)], policy="ranked", horizons={1: 32})
    (directory / "tok.facts").write_text("".join(
        f"{instance}\t{position}\t{token}\n"
        for instance, context in enumerate(contexts)
        for position, token in enumerate(context)))
    subprocess.run(["souffle", "run.dl", "-F", ".", "-D", "."], cwd=directory,
                   check=True, capture_output=True)
    assert set((directory / "cdecide.csv").read_text().splitlines()) == {"0\t7", "1\t8"}


@pytest.mark.parametrize("corrupt_final", [False, True])
def test_firing_domain_frozen_before_test_and_cannot_hide_errors(tmp_path, corrupt_final):
    rows = generate(groups=(2, 2, 2))
    out = tmp_path / "run"

    def oracle(row):
        if row["part"] in ("test", "off_domain"):
            frozen = json.loads((out / "frozen.json").read_text())
            assert frozen["selected"] == [[1, 3, 2]]
            domain = frozen["artifacts"]["selected"]["domain"]
            assert len(domain) == 24
            # All firing cases remain obligations, including intentionally wrong test references.
            assert set(domain) == {r["id"] for r in rows if r["part"] == "test" and r["exposures"] >= 2}
            if corrupt_final and row["part"] == "test":
                return -1
        return row["expected"]

    report = benchmark(out, rows, oracle, {"source": "test-control"})
    chosen = report["results"]["selected"]
    assert chosen["guard_domain_size"] == 24
    assert chosen["scores"]["test"]["wrong"] == (24 if corrupt_final else 0)
    assert not chosen["certificates"]["full_test"]["certified"]  # 12 deliberate abstentions
    cert = chosen["certificates"]["frozen_guard_domain"]
    assert cert["certified"] is (not corrupt_final)
    assert replay(out / cert["evidence"])["certified"] is (not corrupt_final)
    assert chosen["scores"]["off_domain"]["answered"] == 0


def test_constant_oracle_rejects_every_guard_and_empty_proof(tmp_path):
    report = benchmark(tmp_path / "constant", generate(groups=(2, 2, 2)), lambda row: 0, {"source": "constant"})
    assert report["selected"] == []
    chosen = report["results"]["selected"]
    assert chosen["scores"]["test"]["answered"] == 0
    assert chosen["certificates"]["frozen_guard_domain"]["ndomain"] == 0
    assert not chosen["certificates"]["frozen_guard_domain"]["certified"]


def test_empty_guard_freezes_full_sized_domain_and_abstains(tmp_path):
    rows = [row for row in generate() if row["part"] == "test"]
    candidate = write_guard(tmp_path / "empty", [])
    out = tmp_path / "audit"
    domain, audit = freeze_domain(out, candidate, rows, {})
    assert domain == []
    assert audit["certified"] and audit["ndomain"] == 0
    abstained = (Path(audit["evidence"]) / "outputs" / "abstained.csv").read_text().splitlines()
    assert len(abstained) == len(rows)


def test_sparse_baseline_matches_longest_wide_suffix(tmp_path):
    candidate = tmp_path / "circuits.dl"
    emit_sparse_baseline(candidate, {(1,): 8, tuple(range(12)): 9}, False, {}, "test")
    (tmp_path / "tok.facts").write_text("".join(
        f"{i}\t{p}\t{token}\n"
        for i, ctx in enumerate((list(range(12)), [5, 1], [7]))
        for p, token in enumerate(ctx)))
    subprocess.run(["souffle", "run.dl", "-F", ".", "-D", "."],
                   cwd=tmp_path, check=True, capture_output=True)
    assert (tmp_path / "cdecide.csv").read_text().splitlines() == ["0\t9", "1\t8"]
    assert (tmp_path / "abstain.csv").read_text().splitlines() == ["2"]


@pytest.mark.parametrize("defect", ["unrecorded-edit", "missing-reference", "leaked-group", "test-reference"])
def test_admission_rejects_invalid_evidence(tmp_path, defect):
    rows = [r for r in generate(groups=(2, 2, 2)) if r["part"] in ("train", "validation")]
    refs = {r["id"]: r["expected"] for r in rows}
    if defect == "unrecorded-edit":
        edited = next(r for r in rows if r["kind"] == "intervention")
        edited["ctx"][0] = 199999
    elif defect == "missing-reference":
        del refs[rows[0]["id"]]
    elif defect == "leaked-group":
        next(r for r in rows if r["part"] == "validation")["group"] = rows[0]["group"]
    else:
        rows[0]["part"] = "test"
    with pytest.raises(ValueError, match="invalid admission"):
        admit(tmp_path, rows, refs, {})


def test_admission_requires_two_independent_causal_groups_per_split(tmp_path):
    rows = [r for r in generate(groups=(2, 2, 2)) if r["part"] in ("train", "validation")]
    for row in rows:
        if row["part"] == "validation":
            row["group"] = "single-validation-group"
    selected, audit = admit(tmp_path, rows, {r["id"]: r["expected"] for r in rows}, {})
    assert audit["certified"] and selected == []
