"""Compiled-TPR protocol pieces on synthetic data: Python/Datalog role parse, verb whitelist, router precedence."""
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
from benchmark_compiled_tpr import pairs_of, router, souffle_relation, svo_idiom, svo_whitelist, tpr_program  # noqa: E402,E501

pytestmark = pytest.mark.skipif(shutil.which("souffle") is None, reason="needs souffle")

# toy vocabulary: template tokens 1-6, fillers 10-15 (subjects/objects 10-13, verbs 14-15)
THE, the, dot, was, by = 1, 2, 3, 4, 5
META = dict(fillers=[10, 11, 12, 13, 14, 15], n_role=3,
            template={"0": THE, "3": the, "5": dot, "6": THE, "8": was, "10": by, "11": the})


def svo(i, s, vb, o):
    return dict(id=i, ctx=[THE, s, vb, the, o, dot, THE, o, was, vb, by, the], subject=s, verb=vb, object=o)


def weights(rows):
    """tp_w that scores the subject filler at role 0 above everything: the TPR always decides the subject."""
    fill = "".join(f"{t}\t{i}\n" for i, t in enumerate(META["fillers"]))
    w = "".join(f"{v}\t{f}\t0\t{100 if v == META['fillers'][f] else 0}\n" for v in (10, 11, 12, 13) for f in range(6))
    return {"tp_filler": fill, "tp_w": w, "tp_bias": "".join(f"{v}\t0\n" for v in (10, 11, 12, 13))}


def test_pairs_of_matches_datalog_parse_and_rejects_off_template():
    rows = [svo(0, 10, 14, 11), svo(1, 12, 15, 13)]
    py = {(r["id"], *p) for r in rows for p in pairs_of("SVO", r, META)}
    assert py == {(0, 0, 0), (0, 4, 1), (0, 1, 2), (1, 2, 0), (1, 5, 1), (1, 3, 2)}
    bad = svo(2, 10, 14, 11)
    bad["ctx"][7] = 12  # passive object differs from active object: off template
    dl = set(souffle_relation(tpr_program("SVO", 0, META), rows + [bad], "tp_pair", weights(rows)))
    assert dl == py


def test_copy_roles_count_from_the_end():
    row = dict(id=7, ctx=[12, 10, 11])
    meta = dict(fillers=[10, 11, 12], n_role=36)
    assert pairs_of("COPY", row, meta) == [(2, 2), (0, 1), (1, 0)]


def test_whitelist_needs_three_dev_contexts_all_subject():
    dev = [svo(i, 10, 14, 11) for i in range(3)] + [svo(3, 10, 15, 11), svo(4, 10, 15, 11)]
    refs = {0: 10, 1: 10, 2: 10, 3: 10, 4: 10}
    assert svo_whitelist(dev, refs) == {14}  # verb 15 has only two contexts
    refs[1] = 11
    assert svo_whitelist(dev, refs) == set()  # one non-subject answer drops the verb


def test_router_precedence_ngram_then_idiom_then_tpr():
    rows = [svo(0, 10, 14, 11), svo(1, 12, 14, 13), svo(2, 10, 15, 11)]
    # n-gram claims row 0 (its unique suffix: subject 10 ... verb 14) with a wrong-on-purpose token 13
    ngram = ".decl cdecide(inst:number,out:number)\ncdecide(I,13) :- tok(I,1,10), tok(I,2,14).\n"
    idiom = svo_idiom({14})  # fires on rows 0 and 1
    tp = tpr_program("SVO", 0, META)  # fires on all three
    facts = {"tc_" + k: v for k, v in weights(rows).items()}
    layers = set(souffle_relation(router(ngram, idiom, tp), rows, "layer", facts))
    assert layers == {(0, 1, 13), (1, 2, 12), (2, 3, 10)}
