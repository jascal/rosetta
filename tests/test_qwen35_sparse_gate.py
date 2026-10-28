from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_ranks_change_magnitude_with_neuron_id_ties_and_matched_random(tmp_path):
    m=ROOT/'dl/qwen35_sparse_gate_payload.dl'
    f={'shape':'129\n','context':'0\t7\n1\t7\n','donor':'0\t1\n','budget':'1\n4\n',
       'product':''.join(f'0\t{n}\t0.0\t{n//2}.0\n' for n in range(129)),
       'random_order':''.join(f'{n}\t{128-n}\n' for n in range(129))}
    x=check(m,m,f,{},evidence_dir=tmp_path/'payload');assert x['certified']
    rank={int(n):int(r) for i,n,r in x['relations']['rank']}
    assert rank[128]==0 and rank[126]==1 and rank[127]==2
    selected={(int(a),int(n)) for a,i,n in x['relations']['selected']}
    assert (10,128) in selected
    assert {n for a,n in selected if a==1282}=={1}
    f['random_order']=f['random_order'].replace('128\t0\n','128\t1\n')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'invalid')['certified']
