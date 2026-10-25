"""Metadata deduplication, complete-vocabulary ranking and categorical serialization."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check


def test_plan_deduplicates_only_identical_windows(tmp_path):
    m=ROOT/'dl/copy_value_readout_plan.dl'
    facts={'row':'0\t10\t10\n1\t10\t20\n2\t30\t30\n',
        'window':''.join(f'{i}\t{j}\t{j+(1 if i==2 else 0)}\n' for i in range(3) for j in range(8)),
        'base_member':'copy\t16\t9\ncopy\t16\t12\n'}
    x=check(m,m,facts,{},evidence_dir=tmp_path/'plan')
    assert x['certified'],x
    assert x['relations']['representative']==[['0','0'],['1','0'],['2','2']]


def test_global_argmax_tie_break_and_missing_vocabulary(tmp_path):
    m=ROOT/'dl/copy_value_readout_decode.dl'
    facts={'feature':'0\n','sample':'1\t0\t2\t0\n','vocab':'3\n',
        'score':'0\t0\t1.0\n0\t1\t2.0\n0\t2\t2.0\n'}
    x=check(m,m,facts,{},evidence_dir=tmp_path/'decode')
    assert x['certified'],x
    assert x['relations']['mapped']==[['0','1']]
    assert ['0','2','2'] in x['relations']['rank']
    facts['score']='0\t0\t1.0\n0\t1\t2.0\n'
    bad=check(m,m,facts,{},evidence_dir=tmp_path/'missing')
    assert not bad['certified']
