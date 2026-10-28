from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_rejects_missing_whole_width_sites_and_wrong_before_value(tmp_path):
    m=ROOT/'dl/qwen35_routing_integrity.dl'
    f={'shape':'2\n','layer':'0\n','context':'0\t7\n','run':'99\t0\t0\t-1\tbaseline\n0\t0\t0\t0\tzero\n',
       'observation':'99\t0\t0\t1.0\n99\t0\t1\t2.0\n','request':'0\t0\t7\t0\t0.0\n0\t0\t7\t1\t0.0\n',
       'applied':'0\t0\t7\t0\t1.0\t0.0\n0\t0\t7\t1\t2.0\t0.0\n'}
    assert check(m,m,f,{},evidence_dir=tmp_path/'good')['certified']
    f['request']='0\t0\t7\t0\t0.0\n'
    assert not check(m,m,f,{},evidence_dir=tmp_path/'missing')['certified']
    f['request']+='0\t0\t7\t1\t0.0\n'
    f['applied']=f['applied'].replace('1\t2.0\t0.0','1\t8.0\t0.0')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'wrong_before')['certified']
