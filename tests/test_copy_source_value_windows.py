"""Fixed windows distinguish complete causal prefixes from real truncation."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check


def test_window_product_uses_requested_size_and_keeps_other_sites():
    torch=pytest.importorskip('torch')
    from run_copy_source_value_windows import lexical_product
    cache={16:{'values':torch.tensor([[[1.],[10.]],[[3.],[30.]]]),'attention':torch.tensor([[.25,.75]]*4)}}
    feature={(1,1,4):{16:torch.tensor([[7.],[70.]])},(1,1,8):{16:torch.tensor([[9.],[90.]])}}
    assert float(lexical_product(cache,feature,{1:[1]},16,3,2,4,1))==55
    assert float(lexical_product(cache,feature,{1:[1]},16,3,2,8,1))==70
    assert float(cache[16]['values'][1,1,0])==30


def test_windows_clamp_at_start_and_mark_truncation(tmp_path):
    module=ROOT/'dl/copy_source_value_windows_plan.dl'
    facts={'donor':'0\t1\n1\t0\n','member':'copy\t16\t9\n',
        'tok':''.join(f'{i}\t{p}\t{100+i*10+p}\n' for i in (0,1) for p in range(8)),
        'edit':'0\t7\n1\t7\n'}
    x=check(module,module,facts,{},evidence_dir=tmp_path/'plan')
    assert x['certified'],x
    assert len(x['relations']['request'])==18
    assert ['0','7','4'] in x['relations']['truncated']
    assert ['0','7','8'] not in x['relations']['truncated']
    assert ['0','7','16','0'] in x['relations']['start']
    assert [r for r in x['relations']['window_tok'] if r[:3]==['0','7','4']]==[
        ['0','7','4',str(j),str(104+j)] for j in range(4)]


def test_self_preservation_is_not_counted_as_donor_transfer(tmp_path):
    module=ROOT/'dl/copy_source_value_windows_summary.dl';candidate=tmp_path/'base.dl'
    candidate.write_text((ROOT/'dl/copy_head_exchange_summary.dl').read_text().replace('certified','base_certified'))
    facts={'request':'0\tcopy\tw4_self\t0\n','donor':'0\t1\n',
        'historical':'0\t10\t10\t1.0\t1.0\n1\t20\t20\t2.0\t2.0\n',
        'result':'0\tcopy\tw4_self\t0\t10\t1.0\t1.0\t1.0\n','previous':'','delta':''}
    x=check(module,candidate,facts,{},evidence_dir=tmp_path/'summary')
    assert x['certified'],x
    assert not x['relations']['mode_transfer']
