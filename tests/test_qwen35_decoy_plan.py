from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
from certificate import check
ROOT=Path(__file__).resolve().parents[1]
def test_plan_prefers_departure_in_either_pair_member_and_keeps_control_strata(tmp_path):
    module=ROOT/'dl/qwen35_decoy_plan.dl'
    facts={'sample':'0\tg\t1\tfirst\t10\n1\tg\t1\tfirst\t11\n2\tg\t1\tfirst\t10\n3\tg\t1\tfirst\t11\n4\tg\t1\tlast\t10\n5\tg\t1\tlast\t11\n',
        'pair':'0\t1\n2\t3\n4\t5\n','ref':'0\t10\n1\t11\n2\t10\n3\t20\n4\t10\n5\t11\n'}
    x=check(module,module,facts,{},evidence_dir=tmp_path/'plan')
    assert x['certified']
    assert sorted(x['relations']['selected'])==[['2','3','g','1','first'],['4','5','g','1','last']]
    facts['ref']=facts['ref'].replace('3\t20\n','')
    assert not check(module,module,facts,{},evidence_dir=tmp_path/'missing')['certified']
