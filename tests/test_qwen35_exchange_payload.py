from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_whole_width_payloads_and_missing_donor(tmp_path):
    m=ROOT/'dl/qwen35_exchange_payload.dl'
    f={'shape':'3\n','request':'z\t0\t1\t0\t7\tzero\ni\t0\t1\t0\t7\tidentity\nd\t0\t1\t0\t7\tdonor\n',
       'observation':'0\t0\t0\t1.0\n0\t0\t1\t2.0\n0\t0\t2\t3.0\n1\t0\t0\t4.0\n1\t0\t1\t5.0\n1\t0\t2\t6.0\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'payload');assert x['certified']
    patches={(a,int(l),int(p),int(n),float(v)) for a,l,p,n,v in x['relations']['patch']}
    assert len(patches)==9;assert ('d',0,7,2,6.0) in patches;assert ('i',0,7,2,3.0) in patches;assert ('z',0,7,2,0.0) in patches
    f['observation']=f['observation'].replace('1\t0\t2\t6.0\n','')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'missing')['certified']

def test_actual_qwen_width_zero_covers_every_neuron(tmp_path):
    m=ROOT/'dl/qwen35_exchange_payload.dl'
    x=check(m,m,{'shape':'9216\n','request':'z\t0\t1\t0\t7\tzero\n','observation':''},{},evidence_dir=tmp_path/'width')
    assert x['certified']
    assert {int(r[3]) for r in x['relations']['patch']}==set(range(9216))
