from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_readout_includes_qwen_norm_offset_and_rejects_wrong_flip_direction(tmp_path):
    m=ROOT/'dl/qwen35_gate_readout.dl'
    f={'shape':'2\n','unit':'603\n','token':'10\n20\n','down':'603\t0\t2.0\n603\t1\t3.0\n',
       'norm':'0\t1.0\n1\t0.0\n','head':'10\t0\t1.0\n10\t1\t0.0\n20\t0\t0.0\n20\t1\t2.0\n',
       'effect':'0\t603\t10\t20\t1.0\t2.0\n','observed':'0\t603\t20\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'readout');assert x['certified']
    assert sorted(x['relations']['coefficient'])==[['603','10','4'],['603','20','6']]
    assert x['relations']['margin_change']==[['0','603','2']]
    f['effect']='0\t603\t10\t20\t2.0\t1.0\n'
    assert not check(m,m,f,{},evidence_dir=tmp_path/'wrong')['certified']
