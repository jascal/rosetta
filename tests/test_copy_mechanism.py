"""Discriminating instrument controls; these synthetic rows are not LLM evidence."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from certificate import check

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")


def run(module, facts):
    p = ROOT / "dl" / module
    return check(p, p, facts, {})


def test_donors_hold_final_token_and_change_continuation():
    facts = {"row": "0\t0\n1\t0\n2\t0\n3\t1\n",
             "tok": "0\t0\t7\n1\t0\t7\n2\t0\t8\n3\t0\t7\n",
             "ref": "0\t10\n1\t11\n2\t11\n3\t10\n"}
    x = run("copy_mechanism_donors.dl", facts)
    assert x["certified"]
    assert sorted(x["relations"]["donor"]) == [["0", "1"], ["1", "0"], ["3", "1"]]
    assert x["relations"]["no_donor"] == [["2"]]
    facts["ref"] += "1\t99\n"
    assert not run("copy_mechanism_donors.dl", facts)["certified"]


def test_summary_distinguishes_transfer_disruption_and_identity_failure():
    facts = {"row": "0\t0\n1\t0\n", "ref": "0\t10\n1\t11\n",
             "donor": "0\t1\n", "planned": "0\tidentity\t0\n0\tzero\t0\n0\texchange\t1\n",
             "base": "0\t5\t2\n", "old_zero": "0\t12\t1\n", "product": "0\t0\t2\n1\t0\t3\n",
             "result": "0\tidentity\t10\t5\t2\n0\tzero\t12\t1\t0\n0\texchange\t11\t2\t5\n"}
    x = run("copy_mechanism_summary.dl", facts)
    assert x["certified"]
    assert ["0", "exchange"] in x["relations"]["transferred"]
    assert ["0", "zero"] in x["relations"]["other"]
    assert ["0", "identity"] in x["relations"]["preserved"]
    # Matching argmax alone must not let a nonidentity write-back pass.
    bad = dict(facts, result=facts["result"].replace("identity\t10\t5", "identity\t10\t4"))
    assert not run("copy_mechanism_summary.dl", bad)["certified"]
    missing = dict(facts, result=facts["result"].replace("0\texchange\t11\t2\t5\n", ""))
    assert not run("copy_mechanism_summary.dl", missing)["certified"]


def test_preflight_rejects_wrong_answers_and_incomplete_coordinate_universe():
    facts = {"row": "0\t2\n", "ref": "0\t10\t5\n", "answer": "0\t10\t5\n",
             "observed": "".join(f"0\t0\t2\t{n}\n" for n in range(4864))}
    assert run("copy_mechanism_preflight.dl", facts)["certified"]
    assert not run("copy_mechanism_preflight.dl", dict(facts, answer="0\t11\t5\n"))["certified"]
    assert not run("copy_mechanism_preflight.dl", dict(facts, observed=facts["observed"].replace("0\t0\t2\t0\n", "")))["certified"]
