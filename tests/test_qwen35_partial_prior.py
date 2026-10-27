from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_partial_prior_sums_both_fixed_units_and_requires_products(tmp_path):
    m=ROOT/'dl/qwen35_partial_readout_prior.dl';core=ROOT/'dl/qwen35_gate_readout.dl'
    f={'shape':'1\n','unit':'603\n1765\n','token':'10\n','down':'603\t0\t2.0\n1765\t0\t3.0\n','norm':'0\t0.0\n',
       'head':'10\t0\t5.0\n','effect':'','observed':'','key':'0\t10\n','product':'0\t603\t1.0\n0\t1765\t2.0\n'}
    x=check(m,core,f,{},evidence_dir=tmp_path/'sum');assert x['certified'];assert x['relations']['prior']==[['0','10','40']]
    f['product']='0\t603\t1.0\n'
    assert not check(m,core,f,{},evidence_dir=tmp_path/'missing')['certified']
