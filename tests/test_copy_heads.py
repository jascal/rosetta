"""Synthetic treatment/control and instrument-gate tests; not model evidence."""
import shutil
import sys
from pathlib import Path

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check

pytestmark=pytest.mark.skipif(shutil.which('souffle') is None,reason='needs souffle')


def query(module,facts):
    p=ROOT/'dl'/module
    return check(p,p,facts,{})


def test_head_plan_keeps_controls_disjoint_and_matches_layer_sizes():
    facts={'row':'0\n1\n','ranked':''.join(f'{l}\t{h}\t{h}\n' for l in (16,21) for h in range(14))}
    x=query('copy_head_plan.dl',facts)
    assert x['certified'],x.get('error')
    assert len(x['relations']['request'])==38
    controls={(int(l),int(h)) for l,h,_ in x['relations']['chosen_control']}
    assert len(controls)==5
    assert not controls & {(16,7),(16,9),(16,12),(21,2),(21,13)}
    sizes={a:int(n) for a,n in x['relations']['arm']}
    assert sizes['copy_triplet']==sizes['ctrl16triplet']==3
    assert sizes['joint']==sizes['joint_control']==5
    bad=dict(facts,ranked=facts['ranked'].replace('16\t1\t1\n','16\t1\t0\n'))
    assert not query('copy_head_plan.dl',bad)['certified']


def test_instrument_requires_parity_completeness_scope_and_reports_rescue_loss():
    facts={'sample':'0\tcorrect\t10\n1\twrong_source\t20\n',
        'baseline':'0\t10\t2\t2\n1\t21\t3\t1\n',
        'request':''.join(f'{i}\t{a}\n' for i in (0,1) for a in ('none','copy_triplet','ctrl16triplet'))}
    values=[(0,'none',10,2,2),(1,'none',21,3,1),(0,'copy_triplet',11,0,0),
            (1,'copy_triplet',20,0,4),(0,'ctrl16triplet',10,2,2),(1,'ctrl16triplet',21,3,1)]
    facts['native']=''.join(f'{i}\t{a}\t{t}\t{b}\t{g}\tall_positions\n' for i,a,t,b,g in values)
    facts['upstream']=''.join(f'{i}\t{a}\t{t}\t{b}\t{g}\n' for i,a,t,b,g in values)
    x=query('copy_head_summary.dl',facts)
    assert x['certified'],x.get('error')
    assert x['relations']['loss']==[['0','copy_triplet']]
    assert x['relations']['rescue']==[['1','copy_triplet']]
    for key,value in (
        ('upstream',facts['upstream'].replace('1\tcopy_triplet\t20','1\tcopy_triplet\t99')),
        ('native',facts['native'].replace('all_positions','last_position')),
        ('native',facts['native'].replace('0\tnone\t10\t2\t2\tall_positions\n','')),
    ):
        assert not query('copy_head_summary.dl',dict(facts,**{key:value}))['certified']
