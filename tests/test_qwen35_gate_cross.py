from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_cross_candidates_and_joint_cover_both_directions(tmp_path):
    m=ROOT/'dl/qwen35_gate_cross_payload.dl'
    f={'context':'0\t7\n1\t7\n','donor':'0\t1\n1\t0\n','candidate':'603\n1765\n',
       'hybrid':'0\t603\t1.0\n0\t1765\t2.0\n1\t603\t3.0\n1\t1765\t4.0\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'plan');assert x['certified'];assert len(x['relations']['run'])==6
    assert len(x['relations']['patch'])==8
    f['hybrid']=f['hybrid'].replace('1\t603\t3.0\n','')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'missing')['certified']
def test_reproduction_rejects_changed_score_hash(tmp_path):
    m=ROOT/'dl/qwen35_gate_cross_reproduction.dl';f={'expected':'1\tabc\n','actual':'1\tabc\n'}
    assert check(m,m,f,{},evidence_dir=tmp_path/'same')['certified']
    f['actual']='1\tdef\n'
    assert not check(m,m,f,{},evidence_dir=tmp_path/'different')['certified']
