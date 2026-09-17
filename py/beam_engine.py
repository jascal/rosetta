"""Generic oracle-driven beam engine (no substrate-specific vocabulary).

Ports the M-step weakest-link beam control-flow from pil's Gate (b) pilot
`decide_energy` into an oracle interface: expand / extract_commit / fallback.
Sudoku, wikitext, and any other substrate live outside this module.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any, Callable, Hashable


@dataclass(frozen=True)
class Step:
    """One expansion step on a beam path.

    key: substrate-defined identity for hash tie-break (e.g. (r,c,v) for sudoku).
    margin: higher is better (forced > guessed); sudoku pins {0.0, 1.0}.
    counts: (cnt, tot) for det_rank cross-multiply, or None => det_rank tie.
    meta: opaque substrate-defined provenance for the oracle's extract_commit.
    """

    key: Hashable
    margin: float
    counts: tuple[int, int] | None
    meta: Any = None


@dataclass(frozen=True)
class Path:
    """A surviving beam hypothesis: full substrate state + ordered step trajectory."""

    state: Any
    steps: tuple[Step, ...]


def det_rank_compare(
    a: tuple[int, int] | None,
    b: tuple[int, int] | None,
) -> int:
    """Compare two (cnt, tot) rates via integer cross-multiply (no division).

    Higher rate ranks better (returns -1 so it sorts first under ascending cmp).
    Either side None => 0 (tie). 0/0 is also a tie via cross-multiply.
    """
    if a is None or b is None:
        return 0
    ca, ta = a
    cb, tb = b
    left = int(ca) * int(tb)
    right = int(cb) * int(ta)
    if left > right:
        return -1
    if left < right:
        return 1
    return 0


def stable_hash_step(rule_id: str, *key: object, wyly_seed: int) -> int:
    """Stable sha256-hex-int over ``rule_id:k0:k1:...:kn:wyly_seed``.

    Plain ``str()`` of each key element — matching an f-string with embedded ints
    for the sudoku 3-tuple case ``f"{rule_id}:{r}:{c}:{v}:{wyly_seed}"``.
    """
    payload = f"{rule_id}:" + ":".join(str(k) for k in key) + f":{wyly_seed}"
    return int(hashlib.sha256(payload.encode()).hexdigest(), 16)


def compare_paths(rule_id: str, wyly_seed: int) -> Callable[[Path, Path], int]:
    """Ascending comparator: -1 if a ranks better (sorts first), +1 if b does, 0 if tied.

    Weakest-link ladder (worst-margin step first): margin, then det_rank via
    integer cross-multiply, then stable hash; shorter path wins after full ties.
    Empty-steps guard first: compare by length only when min length is 0.
    """

    def _cmp(a: Path, b: Path) -> int:
        sa, sb = a.steps, b.steps
        n = min(len(sa), len(sb))
        if n == 0:
            if len(sa) != len(sb):
                return -1 if len(sa) < len(sb) else 1
            return 0
        order_a = sorted(range(len(sa)), key=lambda i: sa[i].margin)
        order_b = sorted(range(len(sb)), key=lambda i: sb[i].margin)
        for ia, ib in zip(order_a, order_b, strict=False):
            ma, mb = sa[ia].margin, sb[ib].margin
            if ma != mb:
                return -1 if ma > mb else 1
        for ia, ib in zip(order_a, order_b, strict=False):
            cmp = det_rank_compare(sa[ia].counts, sb[ib].counts)
            if cmp != 0:
                return cmp
        for ia, ib in zip(order_a, order_b, strict=False):
            ha = stable_hash_step(rule_id, *sa[ia].key, wyly_seed=wyly_seed)
            hb = stable_hash_step(rule_id, *sb[ib].key, wyly_seed=wyly_seed)
            if ha != hb:
                return -1 if ha < hb else 1
        if len(sa) != len(sb):
            return -1 if len(sa) < len(sb) else 1
        return 0

    return _cmp


def beam_decode(
    oracle: Any,
    M: int,
    beam_width: int,
    rule_id: str,
    wyly_seed: int,
) -> dict[str, Any]:
    """M-step oracle-driven beam; commit via extract_commit / fallback on the winner.

    Determinism contract:
      (i)   iterate live paths in current rank order;
      (ii)  append children in exact oracle.expand order (no reordering);
      (iii) stable sort by compare_paths ladder;
      (iv)  Path carries full state for extract_commit.
    """
    paths: list[Path] = [Path(oracle.initial_state(), ())]
    prune_events = 0
    surviving_counts: list[int] = []
    cmp = compare_paths(rule_id, wyly_seed)
    for _ in range(M):
        next_paths: list[Path] = []
        any_pruned = False
        for p in paths:
            children = oracle.expand(p.state)
            if not children:
                any_pruned = True
            for child_state, steps in children:
                next_paths.append(Path(child_state, tuple(p.steps) + tuple(steps)))
        if any_pruned:
            prune_events += 1
        if not next_paths:
            return {
                "committed_value": None,
                "anomaly_total_contradiction": True,
                "n_steps": M,
                "prune_events": prune_events,
                "surviving_counts": surviving_counts,
                "max_committed_margin": 0.0,
            }
        next_paths.sort(key=cmp_to_key(cmp))
        paths = next_paths[:beam_width]
        surviving_counts.append(len(paths))
    winner = paths[0]
    commit = oracle.extract_commit(winner)
    return {
        "committed_value": commit if commit is not None else oracle.fallback(),
        "anomaly_total_contradiction": False,
        "n_steps": M,
        "prune_events": prune_events,
        "surviving_counts": surviving_counts,
        "max_committed_margin": max((s.margin for s in winner.steps), default=0.0),
    }
