from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_shared_individual_joint_rest_and_nonshared_random_control(tmp_path):
    m=ROOT/'dl/qwen35_shared_gate_payload.dl'
    f={'context':'0\t7\n1\t7\n','donor':'0\t1\n','shared':'162\n381\n','top':'0\t162\n0\t381\n0\t9\n',
       'random_order':'162\t0\n5\t1\n381\t2\n7\t3\n','hybrid':''.join(f'0\t{n}\t1.0\n' for n in (162,381,9,5,7))}
    x=check(m,m,f,{},evidence_dir=tmp_path/'plan');assert x['certified']
    sites={(int(a),int(n)) for a,i,n in x['relations']['selected']}
    assert {n for a,n in sites if a==2}=={162,381}
    assert {n for a,n in sites if a==3}=={9}
    assert {n for a,n in sites if a==4}=={5,7}
    f['hybrid']=f['hybrid'].replace('0\t381\t1.0\n','')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'missing')['certified']
