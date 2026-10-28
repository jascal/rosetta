from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_distinguishes_source_tracking_from_unchanged_and_long_pair(tmp_path):
    m=ROOT/'dl/qwen35_decoy_summary.dl'
    f={'planned':'0\t0\t10\t20\t30\t20\n1\t0\t11\t20\t30\t20\n2\t2\t10\t20\t31\t10\n3\t2\t11\t20\t31\t11\n4\t4\t10\t20\t32\t99\n', 'ref':'0\t30\n1\t30\n2\t10\n3\t11\n4\t99\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'summary')
    assert x['certified'];r=x['relations']
    assert sorted(r['tracking'])==[['0'],['1']]
    assert r['long_pair']==[['2']];assert r['decoy_pair']==[['0']]
    assert ['4','unchanged'] in r['effect'];assert ['4','other'] in r['outcome']
    f['ref']=f['ref'].replace('4\t99\n','')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'incomplete')['certified']
