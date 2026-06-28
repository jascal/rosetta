"""rosetta · the reference-model certificates must stay clean — the Datalog verdict is the test.

These are not unit tests of Python; they assert that dl/equiv.dl certifies the reference circuit against whole.dl. If a
rewrite-rule or emitter change ever breaks faithfulness, the certificate goes red here. Needs `souffle` on PATH."""
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "py"))
from oracle import certify  # noqa: E402

REF = os.path.join(HERE, "reference", "threx")
WHOLE = os.path.join(REF, "whole.dl")
CIRCUIT = os.path.join(REF, "circuit.dl")
BEARINGS = [21, 22, 23, 24, 25]

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle on PATH")


def test_threx_composed_circuit_certified():
    """The composed circuit is provably equivalent to the model over all 25 bearing pairs (Datalog certificate)."""
    instances = [[0, 20, bi, bj, 19, 19, 7] for bi in BEARINGS for bj in BEARINGS]
    r = certify(CIRCUIT, WHOLE, instances)
    assert "error" not in r, r.get("error")
    assert r["ncover"] == 25, r
    assert r["nmiss"] == 0, f"infidelity: {r['mismatches']}"
    assert r["nuncov"] == 0, f"gaps: {r['uncovered']}"


def test_threx_full_program_certified(tmp_path):
    """The complete minimized program (composed + minimal-suffix cover) certifies nmiss=0 ∧ nuncov=0 over the train domain.
    Runs against a temp copy so minimize.py's crisp emit doesn't clobber the committed canonical reference/threx artifacts
    (the T-distributional circuits.dl + symbols, written by py/temperature.py)."""
    import subprocess
    work = tmp_path / "threx"
    work.mkdir()
    for f in ("whole.dl", "corpus.json", "lexicon.json"):
        shutil.copyfile(os.path.join(REF, f), work / f)
    r = subprocess.run([sys.executable, os.path.join(HERE, "py", "minimize.py"), "60", "8", str(work)],
                       capture_output=True, text=True)
    assert "threx train CERTIFIED (Datalog): True" in r.stdout, r.stdout + r.stderr
    assert "nmiss=0  nuncov=0" in r.stdout, r.stdout


def test_induction_detector_logic():
    """dl/induction.dl predicts the copy after the most recent earlier occurrence of the current suffix."""
    from oracle import run_induction
    # [A B C A B]: last bigram 'A B' recurred at pos 0-1, the token after it was C → induction predicts C
    assert run_induction([[5, 6, 7, 5, 6]], [7], 2)["n_hit"] == 1     # model follows the copy
    assert run_induction([[5, 6, 7, 5, 6]], [99], 2)["n_miss"] == 1   # induction fires, model differs → flagged


def test_equiv_catches_a_wrong_circuit():
    """Sanity: equiv.dl must FAIL a circuit that disagrees with the model (the certificate can't be vacuous)."""
    # corrupt the ∿ marker (id 20 → id 18, a valid token) so the circuit's frame guard fails and it fires nothing,
    # while the model still answers → equiv.dl must report a gap (uncovered), not a clean certificate.
    instances = [[0, 18, 21, 21, 19, 19, 7]]  # pos 1 is `ne`, not `∿`; circuit's frame guard fails → uncovered
    r = certify(CIRCUIT, WHOLE, instances)
    assert "error" not in r, r.get("error")
    assert r["nuncov"] >= 1 or r["nmiss"] >= 1, f"equiv.dl wrongly certified a non-firing/mismatched circuit: {r}"
