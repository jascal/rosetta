"""Source-selection computation controls, not LLM evidence."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check
from run_copy_longest import generate,library
from run_copy_prefix import disjoint
from copy_longest_artifact import composite

pytestmark=pytest.mark.skipif(shutil.which('souffle') is None,reason='needs souffle')


def test_longer_match_beats_recency_but_ties_small_margins_and_duplicate_targets_abstain(tmp_path):
    seq=list(range(10,34)); query=seq[:12]
    short=list(range(100,106))+query[-3:]+[200]+list(range(110,114))
    tie=list(range(100,106))+query+[200]+list(range(110,114))
    close=list(range(100,106))+query[-11:]+[200]+list(range(110,114))
    repeated=short[:]; repeated[-1]=22
    contexts=[seq+short+query, short+seq+query, seq+tie+query, seq+close+query, seq+repeated+query]
    circuit=tmp_path/'circuit.dl'
    circuit.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\nlc_option(1,6,3).\nlc_option(2,12,3).\n'+
                       (ROOT/'dl/copy_longest_runtime.dl').read_text())
    verifier=tmp_path/'verify.dl'
    verifier.write_text('.decl tok(i:number,p:number,t:number)\n.input tok\n#include "circuit.dl"\n'
                        '.output lc_prediction\n.output lc_recent\n.decl certified()\n.output certified\ncertified().\n')
    x=check(verifier,circuit,{'tok':''.join(f'{i}\t{p}\t{t}\n' for i,c in enumerate(contexts) for p,t in enumerate(c))},{})
    assert x['certified'],x.get('error')
    assert x['relations']['lc_prediction']==[['1','0','22'],['1','1','22'],['2','0','22'],['2','1','22']]
    assert ['0','200'] in x['relations']['lc_recent']
    assert ['1','22'] in x['relations']['lc_recent']


def test_fresh_competition_admission_and_boundary_counterbalance(tmp_path):
    rows,pairs=generate({300},groups=1)
    stats=disjoint(rows,{300})
    assert stats['train']['rows']==114 and stats['validation']['rows']==114
    by_id={r['id']:r for r in rows}
    for p in pairs:
        a,b=by_id[p['a']],by_id[p['b']]
        assert [j for j,(t,u) in enumerate(zip(a['ctx'],b['ctx'])) if t!=u]==p['positions']
        assert a['ctx'][-a['prefix']:]==b['ctx'][-b['prefix']:]
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['split']}\t{r['group']}\n" for r in rows),
        'ref':''.join(f"{r['id']}\t{r['expected']}\n" for r in rows),
        'pair':''.join(f"{p['a']}\t{p['b']}\t{p['group']}\n" for p in pairs),
        'edit':''.join(f"{p['a']}\t{p['b']}\t{j}\n" for p in pairs for j in p['positions'])}
    circuit=tmp_path/'circuit.dl';circuit.write_text(library())
    verifier=ROOT/'dl/copy_longest_select.dl'
    x=check(verifier,circuit,facts,{})
    assert x['certified'],x.get('error')
    assert x['relations']['ls_selected']==[['3','12','0']]
    assert not x['relations']['baseline_overlap']
    # Corrupt one off-boundary long-source reference: only unguarded variants
    # fire there, and the anchored variants must remain eligible independently.
    target=next(r for r in rows if r['length']==48 and r['order']=='long_last' and r['kind']=='repeat')
    bad=dict(facts,ref=facts['ref'].replace(f"{target['id']}\t{target['expected']}\n",f"{target['id']}\t999999\n"))
    y=check(verifier,circuit,bad,{})
    assert y['certified'],y.get('error')
    assert y['relations']['ls_selected']==[['4','12','1']]
    scores={int(c):int(n) for c,s,n,w in x['relations']['score'] if s=='train'}
    assert scores[1]==3*scores[2]  # reversed and padded sources cannot satisfy the boundary guard


def test_final_domains_are_reference_free_and_runtime_check_rejects_missing_abstention(tmp_path):
    rows,_=generate(set(),groups=1)
    circuit=tmp_path/'circuit.dl'
    circuit.write_text(composite((6,24,1)))
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
           'domain':''.join(f"{r['id']}\n" for r in rows),
           'seed':''.join(f"{r['id']}\t{r['seed']}\n" for r in rows)}
    x=check(ROOT/'dl/copy_longest_final_domains.dl',circuit,facts,{})
    assert x['certified'],x.get('error')
    assert len(x['relations']['incremental_domain'])==48
    assert not x['relations']['baseline_domain']
    control={'domain':facts['domain'],
        'expected_decision':''.join('\t'.join(r)+'\n' for r in x['relations']['composite_prediction']),
        'actual_decision':''.join('\t'.join(r)+'\n' for r in x['relations']['composite_prediction']),
        'expected_abstain':''.join('\t'.join(r)+'\n' for r in x['relations']['composite_abstain']),
        'actual_abstain':''.join('\t'.join(r)+'\n' for r in x['relations']['composite_abstain'])}
    verifier=ROOT/'dl/copy_runtime_replay_check.dl'
    assert check(verifier,verifier,control,{})['certified']
    control['actual_abstain']=''
    assert not check(verifier,verifier,control,{})['certified']
