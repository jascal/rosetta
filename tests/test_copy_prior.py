"""Prior agreement is a diagnostic observation, not automatic certification."""
import shutil
import sys
from pathlib import Path

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check
from run_copy_long_consensus import library

pytestmark=pytest.mark.skipif(shutil.which('souffle') is None,reason='needs souffle')


def test_prior_comparison_and_missing_observation_gate(tmp_path):
    seq=list(range(1,33));query=seq[:16]
    ctx=seq*2+list(range(100,106))+query[-12:]+[90]+list(range(110,114))+query
    circuit=tmp_path/'circuit.dl';circuit.write_text(library())
    facts={'tok':''.join(f'0\t{p}\t{t}\n' for p,t in enumerate(ctx)),
        'sample':'0\t17\n','actual':'0\t17\n','requested':'16\t17\n16\t90\n',
        'prior':'16\t17\t2\n16\t90\t1\n'}
    verifier=ROOT/'dl/copy_prior_report.dl'
    x=check(verifier,circuit,facts,{})
    assert x['certified'],x.get('error')
    assert x['relations']['preferred']==[['0','17']]
    assert ['1','1','1','1','1','0'] in x['relations']['counts']
    opposed=dict(facts,prior='16\t17\t2\n16\t90\t3\n')
    y=check(verifier,circuit,opposed,{})
    assert y['certified']  # complete observation, scientifically wrong prior
    assert y['relations']['preferred']==[['0','90']]
    assert ['1','1','1','0','0','0'] in y['relations']['counts']
    assert not check(verifier,circuit,dict(facts,prior='16\t17\t2\n'),{})['certified']
