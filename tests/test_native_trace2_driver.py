"""Driver phases end to end over a controlled synthetic study directory.

These exercise the run-B phases against authored native evidence, so the staging — split labels, negmark, the gate channel, and
the select/sign split discipline — is checked rather than hoped for.

No model: every "observation" here is authored. The study directory carries its
own dataset and protocol so the contexts can be shaped for a candidate program
that actually fires; the committed inputs are covered separately in
test_native_trace2_inputs.py.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from run_native_trace2 import RANKING_SPLIT, features, select
from native_trace import grammar, program

LAYERS = (7, 15, 23)
CHANNELS = ("gate", "value", "product")
N_NEURONS = 4
PER_SPLIT = 24          # 12 positive + 12 negative, above the floor of 8 per class


def _hypotheses(out):
    """13 distractors firing on marker 2, one good rule firing on marker 1."""
    path = out / "hypotheses.dl"
    bases, universe = grammar()
    path.write_text(
        # program() already includes native_trace_runtime.dl, which declares
        # nt_candidate; distractors fire on marker 2, candidate 14 on marker 1
        program(candidates=[u[0] for u in universe][:14])
        + 'nt_candidate(C,I,T) :- candidate(C,_,_,_), C<14, tok(I,0,T), tok(I,1,2).\n'
          'nt_candidate(14,I,T) :- tok(I,0,T), tok(I,1,1).\n')
    return path


def _seal_protocol(out):
    """Pin dataset, hypotheses and preregistration the way the real inputs phase does."""
    protocol = json.loads((out / "protocol.json").read_text())
    protocol["hypotheses_sha256"] = hashlib.sha256((out / "hypotheses.dl").read_bytes()).hexdigest()
    protocol["prereg_sha256"] = hashlib.sha256(
        (ROOT / "docs/native-trace-prereg2.md").read_bytes()).hexdigest()
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def _rows():
    """Contexts whose position 1 carries a marker, 1 for positive and 2 for negative."""
    rows, i = [], 0
    for part in ("select", RANKING_SPLIT):
        for marker in (1, 2):
            for g in range(6):
                for edit in (0, 1):
                    t = 500 + i
                    # position 0 holds the expected token (the candidate predicts
                    # tok(I,0,_)); position 1 carries the class marker
                    ctx = [t, marker, 7000 + i, 8000 + i]
                    row = dict(id=i, ctx=ctx, expected=t, group=f"{part}-{marker}-{g}",
                               part=part, seed=201 if part == "select" else 202,
                               negative=(marker == 2), kind="repeat",
                               layout="plain" if not row_is_negative(marker) else "negative",
                               exposures=0, length=len(ctx), prefix=2)
                    if edit:
                        row["base"] = i - 1
                        row["kind"] = "intervention"
                        ctx[2] = 6000 + i
                    rows.append(row)
                    i += 1
    return rows


def row_is_negative(marker):
    return marker == 2


def _study(tmp_path):
    out = tmp_path / "study"
    out.mkdir()
    rows = _rows()
    (out / "dataset.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out / "protocol.json").write_text(json.dumps({
        "seeds": [201, 202, 203], "split_of_seed": {"201": "select", "202": "sign", "203": "test"},
        "excluded_tokens": 0, "problems": [], "counts": {},
        "dataset_sha256": hashlib.sha256((out / "dataset.json").read_bytes()).hexdigest(),
        "note": "synthetic fixture",}, indent=2) + "\n")
    _hypotheses(out)
    _seal_protocol(out)

    traces = out / "select-traces"
    traces.mkdir()
    for r in rows:
        if r["part"] != "select":
            continue
        arrays = {}
        for layer in LAYERS:
            for channel in CHANNELS:
                v = np.full(N_NEURONS, 100.0 if channel == "value" else 1.0, dtype=np.float32)
                # neuron 1 separates positive copy contexts from negative controls
                v[1] = -4.0 if r["negative"] else 4.0
                arrays[f"{layer}_{channel}"] = v
        np.savez_compressed(traces / f"{r['id']}.npz", **arrays)

    signing = [r for r in rows if r["part"] == RANKING_SPLIT]
    sign_refs, sel_refs = {}, {}
    for r in rows:
        t = r["expected"]
        # the select split punishes the distractors, as run A's did
        (sel_refs if r["part"] == "select" else sign_refs)[r["id"]] = (
            t + 1000 if r["negative"] and r["part"] == "select" else t)
    (out / "select-references.json").write_text(json.dumps({str(k): v for k, v in sel_refs.items()}))
    (out / f"{RANKING_SPLIT}-references.json").write_text(
        json.dumps({str(k): v for k, v in sign_refs.items()}))

    def _write_signing(product_of):
        """The signing observations must mirror what the features were fitted on:
        neuron 1 carries the same +/-4 signal in every channel, so each fitted
        feature's margin actually varies per row."""
        with (out / f"{RANKING_SPLIT}-observations.jsonl").open("w") as stream:
            for r in signing:
                signal = product_of(r)
                obs = [{"layer": layer, "position": 3, "neuron": 1,
                        "gate": signal, "value": signal, "product": signal}
                       for layer in LAYERS]
                stream.write(json.dumps({"id": r["id"], "next": r["expected"],
                                         "observations": obs, "seconds":0.25}) + "\n")

    _write_signing(lambda r: -4.0 if r["negative"] else 4.0)
    return out, rows, signing


def test_features_phase_selects_on_negative_controls_and_carries_gate_mean(tmp_path):
    out, _, _ = _study(tmp_path)
    features(out)
    chosen = json.loads((out / "features.json").read_text())
    assert len(chosen) == 9
    assert {(f["layer"], f["channel"]) for f in chosen} == {
        (l, c) for l in LAYERS for c in CHANNELS}
    # neuron 1 is the discriminator in every channel, polarity toward the positives
    assert {f["neuron"] for f in chosen} == {1}
    assert {f["polarity"] for f in chosen} == {1}
    assert {f["threshold"] for f in chosen} == {0.0}
    # gate means are retained as descriptive metadata, never vote weights
    products = [f for f in chosen if f["channel"] == "product"]
    assert {f["gate_channel"] for f in products} == {"gate"}
    assert {f["gate_mean"] for f in products} == {4.0}
    assert {f["gate_channel"] for f in chosen} == {"gate"}


def test_select_phase_passes_the_gate_and_runs_all_three_arms(tmp_path):
    out, _, _ = _study(tmp_path)
    features(out)
    select(out)
    rel = json.loads((out / "selection.json").read_text())["relations"]
    assert rel["degenerate"] == []
    assert rel["non_degenerate"] == [["()"]]
    # signature_summary columns are (sign_rows, good, negative_rows, negative_good)
    rows_n, good, neg, neg_good = rel["signature_summary"][0]
    assert (int(rows_n), int(good), int(neg), int(neg_good)) == (
        PER_SPLIT, PER_SPLIT // 2, PER_SPLIT // 2, 0)
    proposed = rel["proposed"]
    assert [sum(int(a) == arm for a, _, _ in proposed) for arm in range(3)] == [12, 12, 12]
    # the trace arm admits the rule the trace evidence points at
    assert ["1", "1", "14"] in rel["admitted"]
    # behavior-only ties all 14 at the same score, so the 12-slot budget drops it
    assert all(int(c) != 14 for a, _, c in proposed if a == "0")
    # The degeneracy signature from run A is elementwise equality between the
    # trace and shuffled score vectors. That must NOT hold here. Whether the
    # shuffled arm happens to surface candidate 14 on a 24-row fixture is a
    # property of the shuffle sample, not of the design, so it is not asserted.
    scores = {}
    for a, c, s_ in rel["score"]:
        scores.setdefault(a, {})[c] = s_
    assert set(scores["1"]) == set(scores["2"])
    assert scores["1"] != scores["2"]


def test_select_phase_reports_the_gate_instead_of_ranking_a_uniform_signature(tmp_path):
    # Make the chosen unit's trace identical on every signing row: run A's failure
    # mode. The driver must report degeneracy and rank nothing.
    out, rows, signing = _study(tmp_path)
    features(out)
    with (out / f"{RANKING_SPLIT}-observations.jsonl").open("w") as stream:
        for r in signing:
            obs = [{"layer": layer, "position": 3, "neuron": 1,
                    "gate": 4.0, "value": 4.0, "product": 4.0} for layer in LAYERS]
            stream.write(json.dumps({"id": r["id"], "next": r["expected"],
                                     "observations": obs, "seconds":0.25}) + "\n")
    select(out)
    rel = json.loads((out / "selection.json").read_text())["relations"]
    assert rel["degenerate"] == [["()"]]
    assert rel["proposed"] == []
    assert rel["admitted"] == []
    rows_n, good, neg, neg_good = rel["signature_summary"][0]
    assert (int(rows_n), int(good), int(neg), int(neg_good)) == (
        PER_SPLIT, PER_SPLIT, PER_SPLIT // 2, PER_SPLIT // 2)


def test_input_gate_refuses_a_tampered_dataset(tmp_path):
    out, _, _ = _study(tmp_path)
    rows = json.loads((out / "dataset.json").read_text())
    rows[0]["ctx"][0] = 999999
    (out / "dataset.json").write_text(json.dumps(rows, indent=1) + "\n")
    with pytest.raises(ValueError, match="protocol.json"):
        features(out)


def test_select_phase_refuses_when_a_staged_row_has_no_reference(tmp_path):
    out, _, _ = _study(tmp_path)
    features(out)
    # drop half the sign references: select2 must reject rather than score 0
    path = out / f"{RANKING_SPLIT}-references.json"
    refs = json.loads(path.read_text())
    path.write_text(json.dumps({k: v for k, v in list(refs.items())[:4]}))
    with pytest.raises(ValueError, match="missing references"):
        select(out)