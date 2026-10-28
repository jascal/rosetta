from pathlib import Path
import math,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_attention_factors_reconstruct_and_separate_components(tmp_path):
    m=ROOT/'dl/qwen35_attention_factor_payload.dl'
    f={'context':'0\t0\told\t1\n1\t0\tnew\t1\n','shape':'1\n',
       'weight':'0\t0\t0.75\n0\t1\t0.25\n1\t0\t0.25\n1\t1\t0.75\n',
       'value':'0\t0\t0\t2.0\n0\t1\t0\t6.0\n1\t0\t0\t4.0\n1\t1\t0\t8.0\n',
       'gate':'0\t0\t0.0\n1\t0\t1.0986122886681098\n','vector':'0\t0\t1.5\n1\t0\t5.25\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'payload');assert x['certified'];assert len(x['relations']['run'])==10
    p={int(a):float(v) for a,l,pos,h,d,v in x['relations']['patch']}
    for arm,want in enumerate((1.5,5.25,2.5,2.5,2.25)):assert math.isclose(p[arm],want,rel_tol=1e-12)
    f['vector']='0\t0\t2.0\n1\t0\t5.25\n'
    assert not check(m,m,f,{},evidence_dir=tmp_path/'bad_reconstruction')['certified']

def test_factor_summary_requires_reconstruction_controls_and_fp32_site_accuracy(tmp_path):
    m=ROOT/'dl/qwen35_attention_factor_summary.dl'
    f={'run':'0\t0\t1\tself_reconstructed\n','expected':'0\t10\n1\t20\n','ref':'0\t10\n','reconstruction':'0\t1\n',
       'request':'0\t27\t7\t10\t0\t1.123456789\n','applied':'0\t27\t7\t10\t0\t0.5\t1.1234568357467651\n'}
    assert check(m,m,f,{},evidence_dir=tmp_path/'valid')['certified']
    f['reconstruction']='0\t0\n'
    assert not check(m,m,f,{},evidence_dir=tmp_path/'bad_control')['certified']
