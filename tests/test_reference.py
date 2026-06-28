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


def test_equiv_catches_a_wrong_circuit():
    """Sanity: equiv.dl must FAIL a circuit that disagrees with the model (the certificate can't be vacuous)."""
    # corrupt the ∿ marker (id 20 → id 18, a valid token) so the circuit's frame guard fails and it fires nothing,
    # while the model still answers → equiv.dl must report a gap (uncovered), not a clean certificate.
    instances = [[0, 18, 21, 21, 19, 19, 7]]  # pos 1 is `ne`, not `∿`; circuit's frame guard fails → uncovered
    r = certify(CIRCUIT, WHOLE, instances)
    assert "error" not in r, r.get("error")
    assert r["nuncov"] >= 1 or r["nmiss"] >= 1, f"equiv.dl wrongly certified a non-firing/mismatched circuit: {r}"
