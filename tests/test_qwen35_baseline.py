"""Train-only proposal ranking and held-out rejection remain separate."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check


def test_proposals_reject_validation_labels(tmp_path):
    library=tmp_path/'library.dl'
    library.write_text('.decl candidate(c:number,b:number,a:number,d:number)\n'
        '.decl nt_candidate(c:number,i:number,t:number)\n'
        'candidate(1,0,0,0). candidate(2,0,0,0).\n'
        'nt_candidate(1,0,10). nt_candidate(1,1,20). nt_candidate(1,2,30).\n'
        'nt_candidate(2,0,99). nt_candidate(2,1,20).\n')
    m=ROOT/'dl/qwen35_baseline_propose.dl'
    facts={'tok':'','sample':'0\ttrain\ta\n1\ttrain\tb\n2\tvalidation\tc\n','ref':'0\t10\n1\t20\n'}
    good=check(m,library,facts,{},evidence_dir=tmp_path/'good')
    assert good['certified'] and good['relations']['proposed']==[['1','1'],['2','2']]
    facts['ref']+='2\t30\n'
    bad=check(m,library,facts,{},evidence_dir=tmp_path/'leak')
    assert not bad['certified']


def test_admission_rejects_validation_error_without_backfill(tmp_path):
    library=tmp_path/'library.dl'
    library.write_text('.decl nt_candidate(c:number,i:number,t:number)\n.decl nt_match(i:number,j:number,k:number)\n'+
        ''.join(f'nt_match({i},0,1).\n' for i in range(8))+
        ''.join(f'nt_candidate({c},{i},{99 if c==2 and i==7 else 10+i}).\n' for c in (1,2) for i in range(8)))
    facts={'tok':''.join(f'{i}\t0\t50\n{i}\t1\t{10+i}\n{i}\t2\t50\n' for i in range(8)),
        'sample':''.join(f"{i}\t{'train' if i<4 else 'validation'}\tg{i//2}\n" for i in range(8)),
        'ref':''.join(f'{i}\t{10+i}\n' for i in range(8)),'proposed':'1\t1\n2\t2\n',
        'pair':''.join(f'{i}\t{i+1}\tg{i//2}\n' for i in range(0,8,2)),
        'edit':''.join(f'{i}\t{i+1}\t1\n' for i in range(0,8,2)),
        'query_start':''.join(f'{i}\t2\n' for i in range(8))}
    m=ROOT/'dl/qwen35_baseline_admit.dl'
    x=check(m,library,facts,{},evidence_dir=tmp_path/'admit')
    assert x['certified'] and x['relations']['admitted']==[['1','1']]
