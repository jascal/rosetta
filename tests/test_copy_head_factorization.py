"""Check GQA indexing, isolated substitutions and strict reconstruction gate."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check


def test_factor_products_keep_routing_values_and_edit_positions_separate():
    torch=pytest.importorskip('torch')
    from run_copy_head_factorization import factors
    a={16:{'attention':torch.tensor([[.25,.75]]*4),'values':torch.tensor([[[1.],[10.]],[[3.],[30.]]])}}
    b={16:{'attention':torch.tensor([[1.,0.]]*4),'values':torch.tensor([[[5.],[50.]],[[7.],[70.]]])}}
    for mode,expected in (('identity',25),('exchange',50),('values',65),('routing',10),('source_values',55)):
        assert float(factors(a,b,16,3,1,2,mode,[1]))==expected
    assert float(a[16]['values'][1,1,0])==30


def test_factorization_rejects_bad_reconstruction_and_counts_hybrid_transfer(tmp_path):
    module=ROOT/'dl/copy_head_factorization_summary.dl'
    candidate=tmp_path/'base.dl'
    candidate.write_text((ROOT/'dl/copy_head_exchange_summary.dl').read_text().replace('certified','base_certified'))
    facts={'request':'0\tcopy\tvalues\t1\n','donor':'0\t1\n','historical':'0\t10\t10\t1.0\t1.0\n1\t20\t20\t2.0\t2.0\n',
        'result':'0\tcopy\tvalues\t1\t20\t0.1\t0.1\t2.0\n','delta':'','previous':'',
        'reconstruction':'0\t16\t9\t0.00001\n','expected_head':'0\t16\t9\n'}
    good=check(module,candidate,facts,{},evidence_dir=tmp_path/'good')
    assert good['certified'],good
    assert good['relations']['mode_transfer']==[['0','copy','values','20']]
    facts['reconstruction']='0\t16\t9\t0.001\n'
    bad=check(module,candidate,facts,{},evidence_dir=tmp_path/'bad')
    assert not bad['certified'] and bad['relations']['reconstruction_bad']
    facts['reconstruction']='0\t16\t9\t0.00001\n'
    facts['previous']='0\tcopy\tvalues\t1\t21\t0.1\t0.1\n'
    repeated=check(module,candidate,facts,{},evidence_dir=tmp_path/'bad-repeat')
    assert not repeated['certified'] and repeated['relations']['repeated_bad']
