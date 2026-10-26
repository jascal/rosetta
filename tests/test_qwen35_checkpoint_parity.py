"""Checkpoint parity rejects missing logits and context-state leakage."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check


def test_complete_scores_and_reset_are_required(tmp_path):
    m=ROOT/'dl/qwen35_checkpoint_parity.dl'
    scores=''.join(f'{i}\t0\t0.0\n{i}\t1\t1.0\n' for i in range(3))
    hashes=''.join(f'{i}\t{k}\tsame\n' for i in range(3) for k in ('ordinary','streamed'))
    facts={'domain':'0\n1\n2\n','vocab':'2\n','ordinary':scores,'streamed':scores,
        'reset_pair':'0\t2\n','logits_hash':hashes}
    x=check(m,m,facts,{},evidence_dir=tmp_path/'good')
    assert x['certified'],x
    facts['logits_hash']=hashes.replace('2\tstreamed\tsame','2\tstreamed\tchanged')
    bad=check(m,m,facts,{},evidence_dir=tmp_path/'leak')
    assert not bad['certified'] and bad['relations']['reset_bad']
    facts['logits_hash']=hashes;facts['streamed']=scores.replace('2\t1\t1.0\n','')
    missing=check(m,m,facts,{},evidence_dir=tmp_path/'missing')
    assert not missing['certified'] and missing['relations']['invalid']
