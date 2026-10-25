"""Donor planning and instrument gates, not tests of the LLM hypothesis."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check


def test_plan_retains_both_pair_directions_and_control_arms(tmp_path):
    module=ROOT/'dl/copy_head_exchange_plan.dl'
    result=check(module,module,{'selected':'0\n','known':'0\n1\n','pairing':'0\t1\n',
        'control':'16\t1\n16\t8\n21\t5\n21\t3\n'},{},evidence_dir=tmp_path/'plan')
    assert result['certified'],result
    assert len(result['relations']['request'])==38
    assert result['relations']['donor']==[['0','1'],['1','0']]


def test_summary_separates_transfer_from_identity_integrity(tmp_path):
    module=ROOT/'dl/copy_head_exchange_summary.dl'
    facts={'request':'0\tnone\tbaseline\t0\n0\tcopy\tidentity\t0\n0\tcopy\texchange\t1\n',
        'donor':'0\t1\n','historical':'0\t10\t10\t1.0\t1.0\n1\t20\t20\t2.0\t2.0\n',
        'delta':'0\tcopy\t5.0\n',
        'result':'0\tnone\tbaseline\t0\t10\t1.0\t1.0\t1.0\n0\tcopy\tidentity\t0\t10\t1.0\t1.0\t1.0\n0\tcopy\texchange\t1\t20\t0.1\t0.1\t2.0\n'}
    result=check(module,module,facts,{},evidence_dir=tmp_path/'good')
    assert result['certified'],result
    assert result['relations']['transfer']==[['0','copy','20']]
    assert result['relations']['target_transfer']==[['0','copy','20']]
    facts['result']=facts['result'].replace('identity\t0\t10\t1.0','identity\t0\t30\t1.0')
    bad=check(module,module,facts,{},evidence_dir=tmp_path/'bad')
    assert not bad['certified'] and bad['relations']['parity_bad']
