from pathlib import Path
import math,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_factor_payloads_separate_gate_up_and_stable_negative_activation(tmp_path):
    m=ROOT/'dl/qwen35_mlp_factor_payload.dl'
    f={'shape':'2\n','context':'0\t7\n1\t7\n','donor':'0\t1\n1\t0\n',
       'observation':'0\t0\t1.0\t2.0\n0\t1\t-1000.0\t3.0\n1\t0\t2.0\t5.0\n1\t1\t1.0\t7.0\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'factors');assert x['certified'];assert len(x['relations']['run'])==8
    values={(int(a),int(n)):float(v) for a,l,p,n,v in x['relations']['patch']}
    silu=lambda v:v/(1+math.exp(-v))
    assert math.isclose(values[2,0],silu(2)*2,rel_tol=1e-12)
    assert math.isclose(values[3,0],silu(1)*5,rel_tol=1e-12)
    assert values[0,1]==0
    f['observation']=f['observation'].replace('1\t1\t1.0\t7.0\n','')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'missing')['certified']
def test_reconstruction_requires_full_vocabulary_and_argmax(tmp_path):
    m=ROOT/'dl/qwen35_factor_reconstruction.dl'
    f={'vocab':'2\n','score':'original\t0\t1.0\noriginal\t1\t2.0\nreconstructed\t0\t1.0001\nreconstructed\t1\t2.0\n'}
    assert check(m,m,f,{},evidence_dir=tmp_path/'ok')['certified']
    f['score']=f['score'].replace('reconstructed\t1\t2.0','reconstructed\t1\t0.0')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'bad')['certified']

def test_summary_allows_fp32_rounding_but_rejects_wrong_patch(tmp_path):
    m=ROOT/'dl/qwen35_mlp_factor_summary.dl'
    f={'run':'2\t0\t1\tdonor_gate\n','expected':'0\t10\n1\t20\n','ref':'2\t20\n','reconstruction':'',
       'request':'2\t31\t7\t0\t1.123456789\n','applied':'2\t31\t7\t0\t0.5\t1.1234568357467651\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'fp32');assert x['certified'];assert x['relations']['transfer']==[['2','donor_gate']]
    f['applied']='2\t31\t7\t0\t0.5\t1.13\n'
    assert not check(m,m,f,{},evidence_dir=tmp_path/'wrong_patch')['certified']

def test_linear_reconstruction_query_matches_frozen_query(tmp_path):
    original=ROOT/'dl/qwen35_factor_reconstruction.dl'
    linear=ROOT/'dl/qwen35_factor_reconstruction_linear.dl'
    for j,score in enumerate((
        'original\t0\t-1.0\noriginal\t1\t2.0\nreconstructed\t0\t-1.0001\nreconstructed\t1\t2.0\n',
        'original\t0\t2.0\noriginal\t1\t2.0\nreconstructed\t0\t1.0\nreconstructed\t1\t2.0\n')):
        f={'vocab':'2\n','score':score}
        a=check(original,original,f,{},evidence_dir=tmp_path/f'original-{j}')
        b=check(linear,linear,f,{},evidence_dir=tmp_path/f'linear-{j}')
        assert a['certified']==b['certified'];assert a['relations']==b['relations']
