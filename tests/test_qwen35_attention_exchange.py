from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_head_payload_selects_consistent_heads_and_same_layer_controls(tmp_path):
    m=ROOT/'dl/qwen35_attention_exchange_payload.dl'
    f={'context':'0\t0\told\t7\n1\t0\tnew\t7\n','leading':'27\t12\t1.0\t3\n27\t10\t0.5\t3\n31\t2\t0.8\t2\n',
       'shape':'2\n','random_order':''.join(f'27\t{h}\t{h}\n' for h in range(16)),
       'vector':''.join(f'{i}\t27\t{h}\t{d}\t{i+h+d}.0\n' for i in (0,1) for h in range(16) for d in range(2))}
    x=check(m,m,f,{},evidence_dir=tmp_path/'payload');assert x['certified']
    assert sorted(x['relations']['selected'])==[['27','10'],['27','12']]
    assert sorted(x['relations']['control'])==[['27','0'],['27','1']]
    assert len(x['relations']['run'])==14
    assert ['9002','27','7','10','0','10'] in x['relations']['patch']
    f['vector']=f['vector'].replace('1\t27\t12\t0\t13.0\n','')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'missing')['certified']
def test_exchange_summary_checks_identity_and_requested_sites(tmp_path):
    m=ROOT/'dl/qwen35_attention_exchange_summary.dl'
    f={'run':'1\t0\t1\tselected\n2\t0\t0\tidentity\n','expected':'0\t10\n1\t20\n','ref':'1\t20\n2\t10\n',
       'baseline_hash':'0\tabc\n','hash':'1\tdef\n2\tabc\n','npatch':'1\t1\n2\t1\n',
       'request':'1\t27\t7\t12\t0\t2.0\n2\t27\t7\t12\t0\t1.0\n',
       'applied':'1\t27\t7\t12\t0\t1.0\t2.0\n2\t27\t7\t12\t0\t1.0\t1.0\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'summary');assert x['certified'];assert x['relations']['transferred']==[['1','selected']]
    f['hash']=f['hash'].replace('2\tabc','2\twrong')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'identity_bad')['certified']
