from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check
from qwen35_key_fixtures import rows,facts_for

def run(tmp_path,f,name):
    core=tmp_path/'core.dl'
    if not core.exists():core.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\n'+(ROOT/'dl/copy_longest_runtime.dl').read_text())
    m=ROOT/'dl/qwen35_key_window_payload.dl';return check(m,core,f,{},evidence_dir=tmp_path/name)

def test_true_single_key_proxy_and_distinct_captured_replay(tmp_path):
    f=facts_for(rows(),True);x=run(tmp_path,f,'valid');assert x['certified']
    assert len(x['relations']['run'])==24
    assert ['0','0'] in x['relations']['checked_reference'];assert ['3','5'] in x['relations']['checked_reference']
    # Captured replay is independent of proxies, which are used only at the short position.
    before={(a,p):v for a,p,v in x['relations']['probability'] if a=='0'}
    f['proxy_key']=f['proxy_key'].replace('0\t1\t0\t1.5','0\t1\t0\t-2.0')
    y=run(tmp_path,f,'changed-window');assert y['certified']
    assert before=={(a,p):v for a,p,v in y['relations']['probability'] if a=='0'}

@pytest.mark.parametrize('field',['prior_probability','control_vector'])
def test_missing_expected_relations_fail_closed(tmp_path,field):
    f=facts_for(rows(),True);f[field]='';assert not run(tmp_path,f,'omitted')['certified']
    f=facts_for(rows(),True);f[field]='\n'.join(f[field].splitlines()[1:])+'\n';assert not run(tmp_path,f,'missing-row')['certified']

@pytest.mark.parametrize('kind',['two_short','two_long','zero_short','empty'])
def test_invalid_source_cardinality_never_certifies(tmp_path,kind):
    f=facts_for([] if kind=='empty' else rows(kind),True)
    if kind=='empty':f['context']=''
    assert not run(tmp_path,f,kind)['certified']

def test_crossed_full_donor_vector_with_short_key_probabilities_fails(tmp_path):
    f=facts_for(rows(),True)
    # 6 is the donor's complete captured route. 5 is target-query/donor-short-key.
    full_route={int(a):v for a,d,v in (r.split('\t') for r in f['control_vector'].splitlines())}[6]
    f['control_vector']=''.join(f'{a}\t{d}\t{full_route if a=="5" else v}\n' for a,d,v in (r.split('\t') for r in f['control_vector'].splitlines()))
    x=run(tmp_path,f,'crossed');assert not x['certified'];assert x['relations']['vector_bad']


def test_reconstruction_band_is_one_e_minus_five(tmp_path):
    f=facts_for(rows(),True)
    f['control_vector']=''.join(f'{a}\t{d}\t{float(v)+0.000005:.17g}\n' for a,d,v in (r.split('\t') for r in f['control_vector'].splitlines()))
    assert run(tmp_path,f,'inside-band')['certified']
    f['control_vector']=''.join(f'{a}\t{d}\t{float(v)+0.0001:.17g}\n' for a,d,v in (r.split('\t') for r in f['control_vector'].splitlines()))
    assert not run(tmp_path,f,'outside-band')['certified']


def test_whole_vocabulary_control_requires_exact_argmax_even_inside_band(tmp_path):
    m=ROOT/'dl/qwen35_key_window_logit_control.dl'
    f={'vocab':'2\n','score':'original\t0\t1.000001\noriginal\t1\t1.0\nreconstructed\t0\t1.0\nreconstructed\t1\t1.000001\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'argmax')
    assert not x['certified'];assert x['relations']['badargmax'];assert not x['relations']['badlogit']
