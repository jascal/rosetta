"""rosetta · the exercise-then-confirm harness — two bars, locked with synthetic oracles (no model, no server).

RECOVERY: a COPY oracle (a pure induction machine) is recovered under exercise; an N-GRAM oracle (output depends only on
the last token) is not — the causal perturbation tells them apart.
ADMISSION: a GENERALIZING rule (copy / succession) beats a memorized n-gram cover on a held-out region of NOVEL content;
a memorizing/wrong rule does not. This locks the "beats n-grams on holdout" bar (cover admission)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "py"))
from exercise_confirm import induction_exercise, holdout_vs_ngram, TAU  # noqa: E402
from idiom_learn import learn_relational  # noqa: E402

VOCAB = 500


def copy_oracle(ctx):
    last = ctx[-1]
    js = [j for j in range(len(ctx) - 1) if ctx[j] == last]
    if js and max(js) + 1 < len(ctx):
        return ctx[max(js) + 1]
    return last


def ngram_oracle(ctx):
    return (ctx[-1] * 7 + 3) % VOCAB


def _rels(oracle):
    insts, _seedof = induction_exercise(VOCAB, n_seqs=12, seqlen=16)
    refs = [oracle(c) for c in insts]
    return learn_relational(insts, refs, list(range(len(insts))), oracle, fill=lambda cs: None, maxL=3)


def test_copy_oracle_recovers_induction():
    l1 = next(r for r in _rels(copy_oracle) if r["L"] == 1)
    assert l1["causal"] >= TAU and l1["obs"] >= 0.5, l1


def test_ngram_oracle_not_recovered():
    l1 = next(r for r in _rels(ngram_oracle) if r["L"] == 1)
    assert l1["causal"] < 0.5, l1


def test_copy_rule_admits_over_ngram_on_novel_holdout():
    insts, seedof = induction_exercise(VOCAB, n_seqs=12, seqlen=16)
    refs = [copy_oracle(c) for c in insts]
    train = [i for i in range(len(insts)) if seedof[i] % 2 == 0]
    hold = [i for i in range(len(insts)) if seedof[i] % 2 == 1]

    def copy_rule(ctx):
        last = ctx[-1]
        js = [j for j in range(len(ctx) - 1) if ctx[j] == last]
        return ctx[max(js) + 1] if js and max(js) + 1 < len(ctx) else None

    ho = holdout_vs_ngram(insts, refs, train, hold, copy_rule, w=8)
    assert ho["admits"] and ho["circuit_match"] > ho["ngram_match"], ho
    ho_bad = holdout_vs_ngram(insts, refs, train, hold, lambda ctx: 0, w=8)   # a wrong/constant rule must NOT admit
    assert not ho_bad["admits"], ho_bad


def test_succession_admission_generalizes_past_ngram():
    A = 40                                              # integer 'letters'; ascending runs; model = next-int
    insts = [[i, i + 1, i + 2] for i in range(A - 3)]
    refs = [r[-1] + 1 for r in insts]
    cut = int(len(insts) * 0.6)
    train, hold = list(range(cut)), list(range(cut, len(insts)))   # HELD-OUT = late runs the n-gram never saw
    succ_rule = lambda ctx: ctx[-1] + 1
    ho = holdout_vs_ngram(insts, refs, train, hold, succ_rule, w=3)
    assert ho["admits"] and ho["circuit_match"] > ho["ngram_match"], ho


def test_unified_emit_argmax_certifies_all_paths():
    """The UNIFIED cover (temperature.emit_T: distributional n-gram + structural point-mass circuits) must give the right
    ARGMAX on every routing path — n-gram, frame-gated once-appearing (ABOVE the n-gram, pre-empting a colliding rule),
    succession, and induction. Souffle required (like test_reference); no model/server. (The distributional T-leg is
    exercised on real models; here we lock the argmax collapse across the routing paths.)"""
    import shutil
    import tempfile
    import pytest
    if shutil.which("souffle") is None:
        pytest.skip("souffle not on PATH")
    from temperature import emit_T, certify_argmax  # noqa: E402
    ng = {(9,): [(8, 0.0)], (500,): [(42, 0.0)]}   # distributional n-gram (single logit → point mass); (500,)→42 is a collider
    structural = {"entity_ids": {100, 101, 102}, "families": {"fam": {1: 500, 5: 501}},
                  "lord": {200: 0, 201: 1, 202: 2, 203: 3}, "lat": {0: 200, 1: 201, 2: 202, 3: 203}}
    d = tempfile.mkdtemp()
    out = os.path.join(d, "circuits.dl")
    emit_T(out, ng, 8, idioms=[], induction=True, sym={}, name="t", structural=structural)
    insts = [[7, 9],                    # n-gram: (9,) → 8
             [50, 51, 50],              # induction: copy after the previous 50 → 51
             [200, 201, 202],           # succession: ascending ordinals → 203
             [501, 100, 101, 101, 500]]  # frame-gated once-appearing wins OVER the colliding n-gram (500,)→42 → 100
    refs = {0: 8, 1: 51, 2: 203, 3: 100}
    m, n = certify_argmax(out, insts, refs, list(range(4)), 0.5)
    assert m == n == 4, (m, n)
