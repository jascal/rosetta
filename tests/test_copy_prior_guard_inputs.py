"""Frozen role pools produce disjoint contexts without consulting feature values."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from run_copy_prior_guard import generate
from run_copy_prefix import disjoint


def test_role_generator_keeps_edits_and_all_four_splits_disjoint():
    pools={'query':list(range(1,513)),'target':list(range(1000,2024)),
           'decoy':list(range(3000,4024)),'background':list(range(5000,13192))}
    rows,pairs,used=generate(pools)
    final,final_pairs,last=generate(pools,seeds=(913,914),used=used,labels=('test_a','test_b'))
    assert len(rows)==len(final)==612
    stats=disjoint(rows+final,set())
    assert set(stats)=={'train','validation','test_a','test_b'}
    by_id={r['id']:r for r in rows}
    for p in pairs:
        a,b=by_id[p['a']],by_id[p['b']]
        assert [j for j,(t,u) in enumerate(zip(a['ctx'],b['ctx'])) if t!=u]==p['positions']
        assert len(p['positions'])==a['exposures']
        assert a['ctx'][-a['prefix']:]==b['ctx'][-b['prefix']:]
    for r in rows+final:
        if not r['negative']:
            assert r['ctx'][-1] in pools['query']
            assert r['expected'] in pools['target']
    assert all(len(last[k])<=len(pools[k]) for k in pools)
