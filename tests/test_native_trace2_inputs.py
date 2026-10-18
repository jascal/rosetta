"""Run-B inputs: generated from committed code, disjoint from run A by construction.

Run A's inputs were committed as an artifact with no generator, so they could be
hashed and replayed but not regenerated. These tests are the run-B equivalent of
that guarantee, plus the property run A lacked and this experiment depends on:
the SELECT split must carry negative controls, or change 1 is not implemented.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from run_native_trace2 import (
    RUN_A, SPLIT_OF_SEED, SEEDS, build, check_disjoint, excluded_vocabulary,
)

STUDY = ROOT / "reference/benchmarks/qwen25_native_trace_b"


def _rows():
    return json.loads((STUDY / "dataset.json").read_text())


def _protocol():
    return json.loads((STUDY / "protocol.json").read_text())


def test_committed_inputs_regenerate_bit_for_bit():
    """The committed dataset must be exactly what the generator produces today."""
    rows, _ = build()
    assert rows == _rows()


def test_select_split_carries_negative_controls():
    # The whole point of change 1: the population the features are fitted on must
    # contain the class they are meant to discriminate.
    rows = _rows()
    select = [r for r in rows if r["part"] == "select"]
    assert len(select) == 114
    assert sum(r["negative"] for r in select) == 18
    for part in ("sign", "test"):
        batch = [r for r in rows if r["part"] == part]
        assert len(batch) == 114 and sum(r["negative"] for r in batch) == 18


def test_splits_are_group_and_token_disjoint():
    problems, _ = check_disjoint(*build())
    assert problems == []


def test_no_token_is_shared_with_run_a_or_history():
    # Sharing a token with run A would make the two runs comparable on the same
    # embeddings, which the preregistration forbids.
    forbidden = excluded_vocabulary()
    used = {t for r in _rows() for t in r["ctx"]}
    assert not used & forbidden


def test_run_a_frozen_inputs_are_untouched():
    for rel in RUN_A:
        path = ROOT / rel
        assert path.exists(), rel
    # run A's own manifest still hashes to what its report records
    frozen = ROOT / "reference/benchmarks/qwen25_native_trace/frozen.json"
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    report = json.loads((ROOT / "reference/benchmarks/qwen25_native_trace/report.json").read_text())
    assert digest == report["frozen_sha256"] == (
        "451c1650f98c973310f960154c905afc92ae72542c14c6b8b61d04ee347e327e")


def test_protocol_pins_the_dataset_hash_and_seeds():
    protocol = _protocol()
    digest = hashlib.sha256((STUDY / "dataset.json").read_bytes()).hexdigest()
    assert protocol["dataset_sha256"] == digest
    assert tuple(protocol["seeds"]) == SEEDS
    assert protocol["split_of_seed"] == {str(k): v for k, v in SPLIT_OF_SEED.items()}
    assert protocol["problems"] == []


def test_generator_is_deterministic_and_seed_sensitive():
    first, _ = build()
    second, _ = build()
    assert first == second
    # the seeds are load-bearing: the split labels follow the seed, and the
    # generated rows are ordered select -> sign -> test
    assert set(SPLIT_OF_SEED) == {201, 202, 203}
    assert [r["seed"] for r in first[:1]] == [201]
    assert [r["seed"] for r in first[-1:]] == [203]
    assert [r["part"] for r in first[:1]] == ["select"]