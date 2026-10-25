"""Feature keys and source-only GQA substitution are independent of answers."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check


def test_plan_requests_both_feature_contexts_and_keeps_self_donors(tmp_path):
    module=ROOT/'dl/copy_source_value_locality_plan.dl'
    facts={'donor':'0\t1\n1\t0\n','member':'copy\t16\t9\n',
        'tok':'0\t0\t42\n0\t1\t10\n1\t0\t42\n1\t1\t20\n','edit':'0\t1\n1\t1\n'}
    result=check(module,module,facts,{},evidence_dir=tmp_path/'plan')
    assert result['certified'],result
    assert len(result['relations']['request'])==14
    assert len(result['relations']['feature'])==4
    assert ['0','copy','token_self','0'] in result['relations']['request']
    assert ['0','copy','token_donor','1'] in result['relations']['request']


def test_lexical_product_preserves_other_positions_and_uses_correct_key():
    torch=pytest.importorskip('torch')
    from run_copy_source_value_locality import lexical_product
    cache={16:{'values':torch.tensor([[[1.],[10.]],[[3.],[30.]]]),'attention':torch.tensor([[.25,.75]]*4)}}
    features={('token',-1,99):{16:torch.tensor([[7.],[70.]])},
        ('bigram',42,99):{16:torch.tensor([[9.],[90.]])}}
    sites={1:[(1,42,99)]}
    assert float(lexical_product(cache,features,sites,16,3,2,'token',1))==55
    assert float(lexical_product(cache,features,sites,16,3,2,'bigram',1))==70
    assert float(cache[16]['values'][1,1,0])==30


def test_summary_rejects_missing_contextual_replay(tmp_path):
    module=ROOT/'dl/copy_source_value_locality_summary.dl';candidate=tmp_path/'base.dl'
    candidate.write_text((ROOT/'dl/copy_head_exchange_summary.dl').read_text().replace('certified','base_certified'))
    facts={'request':'0\tcopy\tsource_values\t1\n','donor':'0\t1\n',
        'historical':'0\t10\t10\t1.0\t1.0\n1\t20\t20\t2.0\t2.0\n',
        'result':'0\tcopy\tsource_values\t1\t20\t0.1\t0.1\t2.0\n','previous':'','delta':''}
    bad=check(module,candidate,facts,{},evidence_dir=tmp_path/'bad')
    assert not bad['certified'] and bad['relations']['locality_invalid']
    facts['previous']='0\tcopy\tsource_values\t1\t20\t0.1\t0.1\n'
    good=check(module,candidate,facts,{},evidence_dir=tmp_path/'good')
    assert good['certified'] and good['relations']['mode_transfer']==[['0','copy','source_values','20']]
