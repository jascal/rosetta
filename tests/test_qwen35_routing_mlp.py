from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_plan_selects_first_routing_change_per_group_and_both_directions(tmp_path):
    m=ROOT/'dl/qwen35_routing_mlp_plan.dl'
    x=check(m,m,{'contrast':'0\tg\t20\t10\t10\t7\n1\tg\t20\t10\t10\t7\n2\th\t20\t30\t10\t7\n'},{},evidence_dir=tmp_path/'plan')
    assert x['certified'];assert x['relations']['selected']==[['0','g']]
    assert len(x['relations']['run'])==26
    assert ['2','0','1','0','donor'] in x['relations']['run']
    assert ['102','1','0','0','donor'] in x['relations']['run']
def test_summary_demands_identity_and_applied_payload(tmp_path):
    m=ROOT/'dl/qwen35_routing_mlp_summary.dl'
    f={'run':'99\t0\t0\t-1\tbaseline\n199\t1\t1\t-1\tbaseline\n0\t0\t0\t0\tidentity\n2\t0\t1\t0\tdonor\n',
       'expected':'0\t10\n1\t20\n','ref':'99\t10\n199\t20\n0\t10\n2\t20\n','hash':'99\ta\n199\tb\n0\ta\n2\tc\n',
       'request':'0\t0\t7\t0\t1.0\n2\t0\t7\t0\t2.0\n','applied':'0\t0\t7\t0\t1.0\t1.0\n2\t0\t7\t0\t1.0\t2.0\n'}
    x=check(m,m,f,{},evidence_dir=tmp_path/'summary');assert x['certified'];assert x['relations']['transfer']==[['2','0']]
    f['hash']=f['hash'].replace('0\ta\n','0\twrong\n')
    assert not check(m,m,f,{},evidence_dir=tmp_path/'bad_identity')['certified']
