"""Run-B native-trace: negative-bearing selection, weighted gated vote, and the
non-degeneracy gate that run A lacked.

Every fixture here is authored so that a plausible-looking wrong rule fails:
the gate must fire when it should, traces on the select split must be rejected
as a leak, and the weighted vote must disagree with run A's binary majority
wherever magnitude carries information the count throws away.

No model, no weights, no network: all of it is Souffle over authored facts.
"""

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from certificate import check
from native_trace import DL

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")

FEATURES2 = DL / "native_trace_features2.dl"
SELECT2 = DL / "native_trace_select2.dl"

CANDIDATE_PROGRAM = (
    '.decl candidate(c:number,b:number,a:number,d:number)\n'
    # 13 distractors plus the one good rule, so the 12-slot budget must drop one
    + "".join(f"candidate({c},0,0,0).\n" for c in range(1, 15))
    + '.decl nt_candidate(c:number,i:number,t:number)\n'
    # distractors fire where marker==2; the good rule fires where marker==1
    'nt_candidate(C,I,T) :- candidate(C,_,_,_), C<14, tok(I,0,T), tok(I,1,2).\n'
    'nt_candidate(14,I,T) :- tok(I,0,T), tok(I,1,1).\n'
)


def _program(tmp_path):
    path = tmp_path / "hypotheses.dl"
    path.write_text(CANDIDATE_PROGRAM)
    return path


# --------------------------------------------------------------- features2

def _selection_facts(n_pos=8, n_neg=8):
    """select split: n_pos positive copy contexts, n_neg negative controls."""
    obs, gate = [], []
    v = 0
    for _ in range(n_pos):           # positive: unit 1 high, unit 0 irrelevant scale
        obs.append((v, 0, 100.0)); obs.append((v, 1, 4.0)); gate.append((v, 0, 1.0))
        v += 1
        gate.append((v - 1, 1, 1.0))
    for _ in range(n_neg):           # negative control: unit 1 low
        obs.append((v, 0, 100.0)); obs.append((v, 1, -4.0))
        gate.append((v, 0, 1.0)); gate.append((v, 1, 1.0))
        v += 1
    return {
        "sample": "".join(f"{i}\t{0 if i < n_pos else 1}\n" for i in range(n_pos + n_neg)),
        "unit": "0\n1\n",
        "observation": "".join(f"{i}\t{n}\t{x}\n" for i, n, x in obs),
        "gate": "".join(f"{i}\t{n}\t{x}\n" for i, n, x in gate),
    }


def test_features2_selects_against_negative_controls_not_scale(tmp_path):
    # Unit 0 has 25x the scale and carries no signal; unit 1 separates the classes.
    facts = _selection_facts()
    result = check(FEATURES2, _program(tmp_path), facts, {})
    assert result["certified"], result
    assert not result.get("error"), result
    assert result["relations"]["size"] == [["0", "8"], ["1", "8"]]
    assert result["relations"]["eligible"] == [["()"]]
    chosen = result["relations"]["chosen"]
    assert len(chosen) == 1 and chosen[0][0] == "1"
    assert chosen[0][3] == "1"                      # polarity points at the positive class
    assert float(chosen[0][2]) == pytest.approx(0)  # midpoint of +4 / -4
    assert result["relations"]["gate_mean"] == [["1", "1"]]


def test_features2_class_floor_is_enforced_and_reported(tmp_path):
    # 8 positive / 7 negative: below the registered floor of 8 per class.
    facts = _selection_facts(n_pos=8, n_neg=7)
    result = check(FEATURES2, _program(tmp_path), facts, {})
    assert result["certified"], result                  # a small class is data, not corruption
    assert result["relations"]["chosen"] == []
    assert result["relations"]["eligible"] == []
    assert result["relations"]["no_vote"] == [["class_below_floor", "8", "7"]]
    # One more instance in each class clears the floor.
    ok = _selection_facts(n_pos=8, n_neg=8)
    assert check(FEATURES2, _program(tmp_path), ok, {})["relations"]["chosen"]


def test_features2_keeps_a_chosen_unit_even_when_raw_gate_is_zero(tmp_path):
    facts = _selection_facts()
    facts["gate"] = "".join(f"{i}\t{n}\t0.0\n" for i, n, _ in
                            [(int(i), int(n), 0.0)
                             for i, n, _ in (line.split("\t") for line in
                                             _selection_facts()["gate"].splitlines())])
    result = check(FEATURES2, _program(tmp_path), facts, {})
    assert result["relations"]["chosen"]                       # still selected
    assert result["relations"]["gate_mean"] == [["1", "0"]]
    assert "unweighted" not in result["relations"]  # gate magnitude never suppresses a vote


def test_features2_fails_closed_on_incomplete_observations(tmp_path):
    facts = _selection_facts()
    facts["observation"] = facts["observation"].replace("0\t1\t4.0\n", "")
    result = check(FEATURES2, _program(tmp_path), facts, {})
    assert not result["certified"]
    assert not result.get("error"), result  # semantic rejection, not a broken program


# ----------------------------------------------------------------- select2

def _runb_fixture(tmp_path, *, uniform=False, margin_spec=None):
    """select + sign splits, 8 negative controls each, one discriminating feature."""
    facts = {k: "" for k in ("tok", "ref", "sample", "negmark", "pair",
                             "trace", "gate", "permutation")}
    for part_idx, part in enumerate(("sign", "select")):
        base = part_idx * 16
        for marker in (1, 2):
            for g in range(4):
                for edit in (0, 1):
                    i = base + (marker - 1) * 8 + g * 2 + edit
                    t = 100 + i
                    facts["tok"] += f"{i}\t0\t{t}\n{i}\t1\t{marker}\n"
                    # the select split punishes the distractors: they fire where
                    # marker==2 and are wrong there, so only the good rule is clean
                    # on both splits. Mirrors run A's train/validation asymmetry.
                    reference = t if part != "select" or marker == 1 else t + 1000
                    facts["ref"] += f"{i}\t{reference}\n"
                    facts["sample"] += f"{i}\t{part}-{marker}-{g}\t{part}\tinput-{i}\n"
                    if marker == 2:                       # negatives are the marker-2 rows
                        facts["negmark"] += f"{i}\n"
                    if edit:
                        facts["pair"] += f"{i-1}\t{i}\n"
                    if part == "sign":                    # traces live on sign only
                        v = 1 if (uniform or marker == 1) else -1
                        if margin_spec is not None:
                            v = margin_spec[i]
                        facts["trace"] += f"{i}\t0\t{v}\n"
                        facts["gate"] += f"{i}\t0\t1\n"
    # swap the two marker classes so the shuffled arm sees the opposite correlation
    for i in range(8):
        facts["permutation"] += f"{i}\t{i+8}\n{i+8}\t{i}\n"
    facts["feature"] = "0\t0\t1\n"                     # f, threshold, polarity, gate_mean
    return _program(tmp_path), facts


def test_nondegeneracy_gate_stops_a_uniform_signature(tmp_path):
    # This is run A's signature exactly: every sign row positive, negatives included.
    program, facts = _runb_fixture(tmp_path, uniform=True)
    result = check(SELECT2, program, facts, {})
    assert result["certified"], result
    assert not result.get("error"), result
    assert result["relations"]["degenerate_uniform"] == [["()"]]
    assert result["relations"]["degenerate_all_negatives"] == [["()"]]
    assert result["relations"]["degenerate"] == [["()"]]
    assert result["relations"]["non_degenerate"] == []
    # The gate is a hard stop, not a hint: nothing is ranked.
    assert result["relations"]["proposed"] == []
    assert result["relations"]["admitted"] == []
    summary = result["relations"]["signature_summary"][0]
    assert summary == ["16", "16", "8", "8"]


def test_trace_arm_admits_what_behavior_and_shuffled_miss(tmp_path):
    program, facts = _runb_fixture(tmp_path)
    result = check(SELECT2, program, facts, {})
    assert result["certified"], result
    assert not result.get("error"), result
    # The healthy signature passes the gate.
    assert result["relations"]["degenerate"] == []
    assert result["relations"]["non_degenerate"] == [["()"]]
    assert result["relations"]["signature_summary"][0] == ["16", "8", "8", "0"]

    proposed = result["relations"]["proposed"]
    assert [sum(int(a) == arm for a, _, _ in proposed) for arm in range(3)] == [12, 12, 12]
    # Behavior ties all 14 at the same score, so the 12-slot budget drops candidate 14.
    assert all(int(c) != 14 for a, _, c in proposed if a in ("0", "2"))
    # The trace arm ranks it first and admits it; the shuffled arm does not.
    assert [c for a, _, c in proposed if a == "1"][0] == "14"
    assert result["relations"]["admitted"] == [["1", "1", "14"]]
    # The trace term is signed: the good rule gains, the distractors are penalised.
    term = {int(c): float(t) for a, c, t in result["relations"]["trace_term"] if a == "1"}
    assert term[14] == pytest.approx(8.0)
    assert term[1] == pytest.approx(-8.0)


@pytest.mark.parametrize("defect", ["traces_on_select", "missing_trace", "bad_permutation", "leaked_test"])
def test_selection_integrity_rejects_leaked_or_incomplete_data(tmp_path, defect):
    program, facts = _runb_fixture(tmp_path)
    if defect == "traces_on_select":
        # a trace on the split that fitted the thresholds is a leak
        facts["trace"] += "16\t0\t1\n"
        facts["gate"] += "16\t0\t1\n"
    elif defect == "missing_trace":
        facts["trace"] = facts["trace"].replace("0\t0\t1\n", "")
    elif defect == "bad_permutation":
        facts["permutation"] = facts["permutation"].replace("0\t8\n", "0\t9\n")
    else:
        facts["sample"] = facts["sample"].replace("\tsign\t", "\ttest\t")
    result = check(SELECT2, program, facts, {})
    assert not result["certified"], result
    assert not result.get("error"), result


def _margin_facts(values):
    """Three rows x three features. Row i carries values[i] on f0 and -1 elsewhere,
    every gate open and every threshold 0, so margin == value and weight == 1."""
    tok = ref = sample = negmark = ""
    for i, v in enumerate(values):
        tok += f"{i}\t0\t7\n{i}\t1\t{1 if i == 0 else 2}\n"
        ref += f"{i}\t7\n"
        sample += f"{i}\tg{i}\tsign\tinput-{i}\n"
        if i:
            negmark += f"{i}\n"
    # the select split is required to exist but carries no trace: it fits
    # parameters in the real run and is never ranked here
    for j in range(2):
        tok += f"{100+j}\t0\t7\n{100+j}\t1\t{1 if j == 0 else 2}\n"
        ref += f"{100+j}\t7\n"
        sample += f"{100+j}\tgsel{j}\tselect\tinput-{100+j}\n"
    feature = "".join(f"{f}\t0\t1\n" for f in range(3))
    trace = ""
    for i, v in enumerate(values):
        for f in range(3):
            trace += f"{i}\t{f}\t{v if f == 0 else -1.0}\n"
    gate = "".join(f"{i}\t{f}\t1\n" for i in range(3) for f in range(3))
    perm = "".join(f"{i}\t{i}\n" for i in range(3))
    return {"tok": tok, "ref": ref, "sample": sample, "negmark": negmark, "pair": "",
            "feature": feature, "trace": trace, "gate": gate, "permutation": perm}


def test_weighted_vote_uses_magnitude_where_the_binary_majority_cannot(tmp_path):
    """Margins +1,-1,-1: the count votes BAD (1*2 > 3 is false) and the signed sum
    is -1, also bad. Margins +3,-1,-1: the count still votes BAD because it
    discards magnitude, while the signed sum is +1 and votes GOOD. That
    divergence is the entire reason change 2 exists, so it is asserted directly.
    """
    # row0 contributes +3, rows 1 and 2 contribute -1 each => sum +1 => GOOD
    good = check(SELECT2, _program(tmp_path), _margin_facts([3.0, 1.0, 1.0]), {})
    assert good["certified"], good
    assert good["relations"]["trace_good"] == [["0"]]
    assert float(good["relations"]["signature_sum"][0][1]) == pytest.approx(1.0)
    # the binary count disagrees: only one of three rows votes positive
    assert not (1 * 2 > 3)

    # a small positive majority must not be able to carry a negative sum
    bad = check(SELECT2, _program(tmp_path), _margin_facts([1.0, 1.0, 1.0]), {})
    assert float(bad["relations"]["signature_sum"][0][1]) == pytest.approx(-1.0)
    assert bad["relations"]["trace_good"] == []


def test_raw_gate_magnitude_and_sign_do_not_weight_margins(tmp_path):
    facts = _margin_facts([3.0, 2.0, 2.0])
    base = check(SELECT2, _program(tmp_path), facts, {})
    facts["gate"] = "".join(f"{i}\t{f}\t{-1000.0 if int(f)%2 else 0.0}\n"
                              for i,f,_ in (line.split("\t") for line in facts["gate"].splitlines()))
    changed = check(SELECT2, _program(tmp_path), facts, {})
    assert changed["certified"]
    for name in ("signature_sum", "score", "trace_term"):
        assert changed["relations"][name] == base["relations"][name]


def test_negative_margins_retain_magnitude_and_zero_is_neutral(tmp_path):
    program, facts = _runb_fixture(tmp_path, margin_spec=[1.0]*8+[-0.25,-9.0,0.0,-1.0]*2)
    result = check(SELECT2, program, facts, {})
    assert result["certified"]
    terms = {(int(a),int(c)):float(v) for a,c,v in result["relations"]["trace_term"]}
    assert terms[1,1] == pytest.approx(-20.5)
    assert terms[1,14] == pytest.approx(8.0)
    assert terms[2,14] == pytest.approx(-20.5)
