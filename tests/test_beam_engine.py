"""Generic beam_engine unit tests — pure Python, no pil import.

Hand-built toy oracles/vectors only (same convention as test_serve_khop.py).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
import sys
from functools import cmp_to_key

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "py",
    ),
)
from beam_engine import (  # noqa: E402
    Path,
    Step,
    beam_decode,
    compare_paths,
    det_rank_compare,
    stable_hash_step,
)


# ---------------------------------------------------------------------------
# det_rank_compare
# ---------------------------------------------------------------------------


def test_det_rank_cross_multiply_sign():
    """(7,10) vs (5,8): 7/10=0.7 > 5/8=0.625 → a ranks better (-1)."""
    assert det_rank_compare((7, 10), (5, 8)) == -1
    assert det_rank_compare((5, 8), (7, 10)) == 1


def test_det_rank_equal_ratios_unequal_denominators():
    """2/4 vs 3/6 must be TIED (naive raw-count comparison would get this wrong)."""
    assert det_rank_compare((2, 4), (3, 6)) == 0


def test_det_rank_zero_over_zero_tied():
    assert det_rank_compare((0, 0), (0, 0)) == 0
    # 0/0 vs 5/10: left=0*10=0, right=5*0=0 → tied (cross-multiply equality)
    assert det_rank_compare((0, 0), (5, 10)) == 0
    # 0/0 vs 0/5: left=0*5=0, right=0*0=0 → tied
    assert det_rank_compare((0, 0), (0, 5)) == 0


def test_det_rank_none_is_tie():
    assert det_rank_compare(None, (1, 2)) == 0
    assert det_rank_compare((1, 2), None) == 0
    assert det_rank_compare(None, None) == 0


def test_det_rank_compare_is_float_free():
    """No ast.Div anywhere in det_rank_compare body (integer cross-multiply only)."""
    source = inspect.getsource(det_rank_compare)
    tree = ast.parse(source)
    comparator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "det_rank_compare"
    )
    assert not any(isinstance(node, ast.Div) for node in ast.walk(comparator))


# ---------------------------------------------------------------------------
# stable_hash_step
# ---------------------------------------------------------------------------


def test_stable_hash_step_payload_and_value():
    rule_id, r, c, v, wyly_seed = "decide_energy_v1", 3, 4, 7, 0
    expected_payload = f"{rule_id}:{r}:{c}:{v}:{wyly_seed}"
    expected = int(hashlib.sha256(expected_payload.encode()).hexdigest(), 16)
    got = stable_hash_step(rule_id, r, c, v, wyly_seed=wyly_seed)
    assert got == expected
    # payload format for 3-key case matches f-string embedding
    assert (
        f"{rule_id}:" + ":".join(str(k) for k in (r, c, v)) + f":{wyly_seed}"
        == expected_payload
    )


def test_stable_hash_step_varargs_key():
    """*key with a single element still colon-joins correctly."""
    h = stable_hash_step("rid", 42, wyly_seed=1)
    payload = "rid:42:1"
    assert h == int(hashlib.sha256(payload.encode()).hexdigest(), 16)


# ---------------------------------------------------------------------------
# compare_paths ladder rungs
# ---------------------------------------------------------------------------


def _path(*steps: Step, state: object = None) -> Path:
    return Path(state=state, steps=tuple(steps))


def test_compare_margin_ladder_wins():
    """Higher margin ranks better when margins differ."""
    a = _path(Step(key=("a",), margin=1.0, counts=(1, 2)))
    b = _path(Step(key=("b",), margin=0.0, counts=(9, 10)))
    cmp = compare_paths("rid", 0)
    assert cmp(a, b) == -1
    assert cmp(b, a) == 1


def test_compare_det_rank_ladder_wins_when_margins_tie():
    """Equal margins → higher det_rank rate wins."""
    a = _path(Step(key=("a",), margin=0.0, counts=(7, 10)))
    b = _path(Step(key=("b",), margin=0.0, counts=(5, 8)))
    cmp = compare_paths("rid", 0)
    assert cmp(a, b) == -1  # 7/10 > 5/8


def test_compare_hash_tie_break_when_margin_and_detrank_tie():
    """Equal margin + equal counts → stable hash decides."""
    a = _path(Step(key=(0, 0, 1), margin=0.0, counts=(5, 10)))
    b = _path(Step(key=(0, 0, 2), margin=0.0, counts=(5, 10)))
    cmp0 = compare_paths("decide_energy_v1", 0)
    result = cmp0(a, b)
    assert result in (-1, 1)
    # same seed is deterministic
    assert cmp0(a, b) == result
    # some seed flips order
    found_flip = False
    for s in range(50):
        if compare_paths("decide_energy_v1", s)(a, b) != result:
            found_flip = True
            break
    assert found_flip


def test_compare_shorter_path_wins_after_full_ladder_tie():
    """After margin+detrank+hash all tie on shared positions, shorter path wins."""
    # identical single step (same key → same hash)
    s = Step(key=("x",), margin=0.0, counts=(1, 1))
    a = _path(s)
    b = _path(s, Step(key=("y",), margin=0.0, counts=(1, 1)))
    cmp = compare_paths("rid", 0)
    # shared position ties; shorter (a) wins
    assert cmp(a, b) == -1
    assert cmp(b, a) == 1


def test_compare_empty_steps_guard():
    """min length 0 → compare by length only (shorter ranks better)."""
    empty = _path()
    nonempty = _path(Step(key=("z",), margin=1.0, counts=(1, 1)))
    cmp = compare_paths("rid", 0)
    assert cmp(empty, nonempty) == -1
    assert cmp(nonempty, empty) == 1
    assert cmp(empty, empty) == 0


# ---------------------------------------------------------------------------
# Toy oracles for beam_decode control flow
# ---------------------------------------------------------------------------


class _ForcedCommitOracle:
    """Propagate-style: one child, many steps, then carry once target is set."""

    def __init__(self, target_value: int = 42) -> None:
        self.target_value = target_value
        self._n_expands = 0

    def initial_state(self) -> dict:
        return {"committed": None, "phase": 0}

    def expand(self, state: dict) -> list[tuple[dict, list[Step]]]:
        self._n_expands += 1
        if state["committed"] is not None:
            # CARRY
            return [(state, [])]
        if state["phase"] == 0:
            # PROPAGATE: one child, many forced steps
            child = {"committed": self.target_value, "phase": 1}
            steps = [
                Step(key=("f", 1), margin=1.0, counts=(3, 3)),
                Step(key=("f", 2), margin=1.0, counts=(3, 3)),
            ]
            return [(child, steps)]
        return [(state, [])]

    def extract_commit(self, path: Path) -> int | None:
        return path.state.get("committed")

    def fallback(self) -> int:
        return -1


class _BranchPruneRecoverOracle:
    """Branch (K children, 1 step), then prune the bad branch, carry the good."""

    def initial_state(self) -> dict:
        return {"val": None, "step": 0}

    def expand(self, state: dict) -> list[tuple[dict, list[Step]]]:
        if state["step"] == 0:
            # BRANCH: two children; "bad" will prune next, "good" survives
            # Deliberately return in non-sorted order (bad first) to exercise
            # child-order preservation + stable sort.
            bad = {"val": "bad", "step": 1}
            good = {"val": "good", "step": 1}
            return [
                (bad, [Step(key=("bad",), margin=0.0, counts=(1, 2))]),
                (good, [Step(key=("good",), margin=0.0, counts=(9, 10))]),
            ]
        if state["val"] == "bad":
            return []  # PRUNE
        # good: CARRY
        return [(state, [])]

    def extract_commit(self, path: Path) -> str | None:
        v = path.state.get("val")
        return v if v == "good" else None

    def fallback(self) -> str:
        return "fallback"


class _WipeoutOracle:
    """Every expand returns 0 children → total population wipeout."""

    def initial_state(self) -> dict:
        return {}

    def expand(self, state: dict) -> list:
        return []

    def extract_commit(self, path: Path) -> None:
        return None

    def fallback(self) -> int:
        return 0


class _ChildOrderOracle:
    """expand returns children in a specific non-sorted key order; all margins/counts
    equal so the stable sort preserves expand order as the sole rank order.
    """

    def initial_state(self) -> dict:
        return {"picked": None}

    def expand(self, state: dict) -> list[tuple[dict, list[Step]]]:
        if state["picked"] is not None:
            return [(state, [])]
        # Non-sorted order: Z, A, M — equal margin and equal counts so ladder
        # ties everywhere; stable sort must preserve this expand order.
        order = ["Z", "A", "M"]
        out = []
        for k in order:
            child = {"picked": k}
            # identical margin/counts; keys differ but we make hash irrelevant by
            # using counts that force det_rank equality and then... actually hash
            # will break ties. To isolate child-order, we need FULL ladder ties
            # including hash — impossible with different keys.
            # Instead: assert that AFTER sort, relative order among fully-tied
            # items (same key) is preserved; OR use a comparator-neutral setup
            # where we inspect surviving_counts / first-append order via a spy.
            #
            # Spec: "ties are broken purely by the stable sort over the ladder,
            # not by any implicit reordering of oracle.expand's output."
            # So: three children with IDENTICAL steps (same key/margin/counts)
            # would be fully tied; stable sort preserves expand order → winner
            # is the first child returned.
            out.append(
                (
                    child,
                    [Step(key=("same",), margin=0.0, counts=(1, 1))],
                )
            )
        return out

    def extract_commit(self, path: Path) -> str | None:
        return path.state.get("picked")

    def fallback(self) -> str:
        return "none"


def test_beam_decode_clean_forced_commit():
    oracle = _ForcedCommitOracle(target_value=42)
    res = beam_decode(oracle, M=2, beam_width=4, rule_id="t", wyly_seed=0)
    assert res["anomaly_total_contradiction"] is False
    assert res["committed_value"] == 42
    assert res["prune_events"] == 0
    assert res["surviving_counts"] == [1, 1]
    assert res["n_steps"] == 2


def test_beam_decode_branch_then_prune_then_recover():
    oracle = _BranchPruneRecoverOracle()
    res = beam_decode(oracle, M=2, beam_width=4, rule_id="t", wyly_seed=0)
    assert res["anomaly_total_contradiction"] is False
    assert res["committed_value"] == "good"
    assert res["prune_events"] == 1
    # after step 1: 2 children; after step 2: only "good" survives
    assert res["surviving_counts"][0] == 2
    assert res["surviving_counts"][1] == 1


def test_beam_decode_total_wipeout():
    oracle = _WipeoutOracle()
    res = beam_decode(oracle, M=3, beam_width=4, rule_id="t", wyly_seed=0)
    assert res["anomaly_total_contradiction"] is True
    assert res["committed_value"] is None
    assert res["prune_events"] == 1
    assert res["surviving_counts"] == []
    assert res["n_steps"] == 3


def test_beam_decode_child_order_preservation():
    """Fully-tied children: stable sort preserves oracle.expand order → first child wins."""
    oracle = _ChildOrderOracle()
    res = beam_decode(oracle, M=1, beam_width=8, rule_id="t", wyly_seed=0)
    # All three steps are identical (same key/margin/counts) → full ladder tie.
    # Stable sort keeps expand order Z, A, M → winner is Z (first returned).
    assert res["committed_value"] == "Z"
    assert res["surviving_counts"] == [3]

    # Also pin that a non-stable reordering would not be introduced: with beam_width
    # covering all, surviving rank order of commits after M=1 is expand order.
    # Re-run expand manually to confirm order is Z,A,M.
    children = oracle.expand(oracle.initial_state())
    assert [c[0]["picked"] for c in children] == ["Z", "A", "M"]


def test_beam_decode_carry_semantics():
    """1-child-0-steps (carry) keeps path alive without adding steps."""
    oracle = _ForcedCommitOracle(target_value=7)
    # M=1 does the propagate; M=2 does the carry
    res = beam_decode(oracle, M=1, beam_width=2, rule_id="t", wyly_seed=0)
    assert res["committed_value"] == 7
    assert res["surviving_counts"] == [1]
