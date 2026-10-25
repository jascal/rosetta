"""Authored source/answer classes and explanatory read consistency."""
import shutil
import sys
from pathlib import Path

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check

pytestmark=pytest.mark.skipif(shutil.which('souffle') is None,reason='needs souffle')


def circuit(tmp_path):
    p=tmp_path/'circuit.dl'
    p.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\nlc_option(1,3,1).\n'+
                 (ROOT/'dl/copy_longest_runtime.dl').read_text())
    return p


def test_answer_categories_are_disjoint_and_do_not_infer_mechanisms(tmp_path):
    ctx=[1,2,3,4,9,2,3,5,8,2,3]
    facts={'tok':''.join(f'{i}\t{p}\t{t}\n' for i in range(4) for p,t in enumerate(ctx)),
           'sample':''.join(f'{i}\tg{i}\t4\n' for i in range(4)),
           'ref':'0\t4\n1\t5\n2\t9\n3\t99\n'}
    x=check(ROOT/'dl/copy_source_diagnosis.dl',circuit(tmp_path),facts,{})
    assert x['certified'],x.get('error')
    assert x['relations']['class']==[['0','correct'],['1','wrong_source'],['2','wrong_context'],['3','wrong_external']]


def test_peaks_and_promotions_are_descriptive_and_wrong_read_tokens_fail(tmp_path):
    ctx=[1,2,3,4,9,2,3,5,8,2,3]
    facts={'tok':''.join(f'0\t{p}\t{t}\n' for p,t in enumerate(ctx)),
           'sample':'0\twrong_external\t4\n','ref':'0\t99\n','explained':'0\t99\n',
           'head':'0\t1\t0\t0\t3\t4\t0.8\t2.0\n0\t2\t1\t0\t7\t5\t0.7\t1.0\n',
           'promotes':'0\t1\t1\t99\n0\t2\t1\t4\n'}
    x=check(ROOT/'dl/copy_source_explanation.dl',circuit(tmp_path),facts,{})
    assert x['certified'],x.get('error')
    assert x['relations']['peak_class']==[['0','1','gold'],['0','2','competing_successor']]
    assert x['relations']['promotes_actual']==[['0','1']]
    bad=dict(facts,head=facts['head'].replace('3\t4\t0.8','3\t99\t0.8'))
    assert not check(ROOT/'dl/copy_source_explanation.dl',circuit(tmp_path),bad,{})['certified']
