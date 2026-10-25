"""Confirmation inputs retain controls, disjoint vocabularies and every edit."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'py'))
from run_copy_prior_guard_trial import scaffold,materialize
from run_copy_prefix import disjoint
from certificate import check


def test_confirmation_layout_and_exact_allocations():
    pools={'query':list(range(1,513)),'target':list(range(1000,2024)),
        'decoy':list(range(3000,4024)),'background':list(range(5000,13192))}
    groups=scaffold(pools,{k:[] for k in pools},set())
    chosen={g['group']:(g['query'][0],*g['target'][:2],g['decoy'][0]) for g in groups}
    rows,pairs=materialize(groups,chosen)
    assert len(rows)==612 and len(pairs)==288
    assert len(disjoint(rows,set()))==2
    for p in pairs:
        a,b=rows[p['a']],rows[p['b']]
        assert [i for i,(x,y) in enumerate(zip(a['ctx'],b['ctx'])) if x!=y]==p['positions']
        assert len(p['positions'])==a['exposures']
        assert a['ctx'][-a['prefix']:]==b['ctx'][-b['prefix']:]
    assert sum(r['negative'] for r in rows)==36


def test_tuple_constructor_uses_ranked_features_and_rejects_missing(tmp_path):
    module=ROOT/'dl/copy_prior_tuples.dl'
    facts={'planned_group':'g\n','q_candidate':''.join(f'g\t{q}\t{q}\n' for q in range(6)),
        't_candidate':''.join(f'g\t{t}\t{t-100}\n' for t in range(100,112)),
        'd_candidate':''.join(f'g\t{d}\t{d-200}\n' for d in range(200,206)),
        'raw_prior':''.join(f'{q}\t{t}\t{1 if t<200 else 3}\n' for q in range(6)
            for t in [*range(100,112),*range(200,206)])}
    good=check(module,module,facts,{},evidence_dir=tmp_path/'good')
    assert good['certified'] and good['relations']['chosen']==[['g','0','100','101','200']]
    facts['raw_prior']=facts['raw_prior'].replace('0\t100\t1\n','')
    bad=check(module,module,facts,{},evidence_dir=tmp_path/'bad')
    assert not bad['certified'] and bad['relations']['missing']


def test_confirmation_gate_needs_two_incremental_causal_groups_per_seed(tmp_path):
    circuit=tmp_path/'circuit-source.dl'
    circuit.write_text('.decl pg_prediction(c:number,i:number,t:number)\n'
        '.decl bdecide(i:number,t:number)\n.decl lc_source(i:number,j:number,k:number,t:number)\n'+
        ''.join(f'pg_prediction(5,{i},{10+i%2}).\n' for i in range(8))+
        ''.join(f'lc_source({i},0,32,10).\n' for i in range(0,8,2)))
    facts={'domain':''.join(f'{i}\n' for i in range(8)),
        'seed':''.join(f'{i}\t{921+i//4}\n' for i in range(8)),
        'sample':''.join(f'{i}\tg{i//2}\t2\n' for i in range(8)),
        'tok':''.join(f'{i}\t0\t1\n{i}\t1\t{10+i%2}\n{i}\t2\t2\n' for i in range(8)),
        'ref':''.join(f'{i}\t{10+i%2}\n' for i in range(8)),
        'pair':''.join(f'{i}\t{i+1}\tg{i//2}\n' for i in range(0,8,2)),
        'edit':''.join(f'{i}\t{i+1}\t1\n' for i in range(0,8,2))}
    module=ROOT/'dl/copy_prior_guard_trial_summary.dl'
    good=check(module,circuit,facts,{},evidence_dir=tmp_path/'summary-good')
    assert good['certified'],good
    facts['ref']=facts['ref'].replace('7\t11\n','7\t999\n')
    bad=check(module,circuit,facts,{},evidence_dir=tmp_path/'summary-bad')
    assert not bad['certified'] and bad['relations']['wrong']
    assert ['922','1'] in bad['relations']['causal_groups']
