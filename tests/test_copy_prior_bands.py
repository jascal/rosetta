"""Discrete lexical bands retain ordering and reject incomplete observations."""
import shutil
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check

pytestmark=pytest.mark.skipif(shutil.which('souffle') is None,reason='needs souffle')


def test_bands_truncate_nonnegative_gaps_and_clip_tail():
    p=ROOT/'dl/copy_prior_bands.dl'
    facts={'query':'1\n','target':'2\n3\n4\n','maximum':'1\t5\n',
        'score':'1\t2\t4.9\n1\t3\t2.9\n1\t4\t-100\n'}
    x=check(p,p,facts,{})
    assert x['certified'],x.get('error')
    assert x['relations']['prior_band']==[['1','2','0'],['1','3','2'],['1','4','63']]
    assert not check(p,p,dict(facts,score='1\t2\t4.9\n1\t3\t2.9\n'),{})['certified']
    assert not check(p,p,dict(facts,score=facts['score'].replace('4.9','5.1')), {})['certified']
