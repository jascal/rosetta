"""Long-context consensus controls: repeated sources, baseline horizon and admission."""
import shutil
import sys
from pathlib import Path

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from certificate import check
from run_copy_long_consensus import generate,library
from run_copy_prefix import disjoint

pytestmark=pytest.mark.skipif(shutil.which('souffle') is None,reason='needs souffle')


def facts_for(rows,pairs,with_refs=True):
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['split']}\t{r['group']}\n" for r in rows)}
    if with_refs:
        facts.update(ref=''.join(f"{r['id']}\t{r['expected']}\n" for r in rows),
            pair=''.join(f"{p['a']}\t{p['b']}\t{p['group']}\n" for p in pairs),
            edit=''.join(f"{p['a']}\t{p['b']}\t{j}\n" for p in pairs for j in p['positions']))
    return facts


def test_generated_matrix_edits_and_baseline_horizon(tmp_path):
    rows,pairs=generate({300},groups=1)
    stats=disjoint(rows,{300})
    assert stats['train']['rows']==102 and stats['validation']['rows']==102
    by_id={r['id']:r for r in rows}
    for p in pairs:
        a,b=by_id[p['a']],by_id[p['b']]
        assert len(p['positions'])==a['exposures']
        assert [i for i,(t,u) in enumerate(zip(a['ctx'],b['ctx'])) if t!=u]==p['positions']
        assert a['ctx'][-a['prefix']:]==b['ctx'][-b['prefix']:]
    circuit=tmp_path/'library.dl';circuit.write_text(library())
    x=check(ROOT/'dl/copy_long_consensus_domains.dl',circuit,facts_for(rows,pairs,False),{})
    assert x['certified'],x.get('error')
    incremental={int(i) for _,i in x['relations']['incremental']}
    assert incremental and all(by_id[i]['order']=='decoy_last' for i in incremental)
    firing={int(i) for _,i,_ in x['relations']['firing']}
    assert all(not by_id[i]['negative'] for i in firing)
    assert any(by_id[i]['order']=='decoy_first' for i in firing)
    assert not x['relations']['conflict']


def test_authored_consensus_admission_and_error_rejection(tmp_path):
    rows,pairs=generate(set(),groups=2)
    circuit=tmp_path/'library.dl';circuit.write_text(library())
    facts=facts_for(rows,pairs)
    verifier=ROOT/'dl/copy_long_consensus_select.dl'
    x=check(verifier,circuit,facts,{})
    assert x['certified'],x.get('error')
    assert x['relations']['lg_selected']==[['1','16','2']]
    # Longest, three-source row belongs to every firing domain, so one wrong
    # oracle answer must reject all six without successful-case filtering.
    r=next(r for r in rows if r['length']==64 and r['exposures']==3 and r['order']=='decoy_last' and r['kind']=='repeat')
    bad=dict(facts,ref=facts['ref'].replace(f"{r['id']}\t{r['expected']}\n",f"{r['id']}\t999999\n"))
    y=check(verifier,circuit,bad,{})
    assert y['certified'],y.get('error')
    assert y['relations']['lg_selected']==[]
    assert len(y['relations']['rejected'])==6
