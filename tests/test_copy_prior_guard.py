"""Static lexical specialization and guard controls, with no live model."""
import shutil
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check
from run_copy_long_consensus import library

pytestmark=pytest.mark.skipif(shutil.which('souffle') is None,reason='needs souffle')


def context():
    seq=list(range(1,33));q=seq[:16]
    return seq*2+list(range(100,106))+q[-12:]+[90]+list(range(110,114))+q


def test_metadata_specialization_keeps_all_needed_labels(tmp_path):
    circuit=tmp_path/'circuit.dl';circuit.write_text(library())
    facts={'tok':''.join(f'0\t{p}\t{t}\n' for p,t in enumerate(context())),
        'domain':'0\n','raw_prior':'16\t17\t1\n16\t90\t3\n999\t17\t5\n'}
    x=check(ROOT/'dl/copy_prior_specialize.dl',circuit,facts,{})
    assert x['certified'],x.get('error')
    assert x['relations']['selected_prior']==[['16','17','1'],['16','90','3']]
    bad=dict(facts,raw_prior='16\t17\t1\n')
    assert not check(ROOT/'dl/copy_prior_specialize.dl',circuit,bad,{})['certified']


def test_guard_abstains_for_opposing_tied_or_missing_prior(tmp_path):
    verifier=tmp_path/'verify.dl'
    verifier.write_text('.decl tok(i:number,p:number,t:number)\n.input tok\n#include "circuit.dl"\n'
        '.output pg_prediction\n.decl certified()\n.output certified\ncertified().\n')
    circuit=tmp_path/'circuit.dl'
    prefix=library()+('\n.decl pg_option(c:number,minmatch:number,support:number)\n'
        'pg_option(C,M,S) :- lg_option(C,M,S).\n.decl prior_band(q:number,t:number,b:number)\n')
    suffix=(ROOT/'dl/copy_prior_guard_runtime.dl').read_text()
    facts={'tok':''.join(f'0\t{p}\t{t}\n' for p,t in enumerate(context()))}
    for labels,expected in (
        ('prior_band(16,17,1).\nprior_band(16,90,3).\n',[['1','0','17']]),
        ('prior_band(16,17,3).\nprior_band(16,90,1).\n',[]),
        ('prior_band(16,17,1).\nprior_band(16,90,1).\n',[]),
        ('prior_band(16,17,1).\n',[]),
    ):
        circuit.write_text(prefix+labels+suffix)
        x=check(verifier,circuit,facts,{})
        assert x['certified'],x.get('error')
        assert x['relations']['pg_prediction']==expected
