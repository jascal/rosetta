"""Fixed-option independent confirmation; host stages facts, Datalog decides."""
import argparse
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

from certificate import check, replay, sha256
from run_copy_prior_guard import ROOT, OUT as DEVELOPMENT, LEXICON, BASELINE, feature_gate, specialize
from run_copy_prefix import gate, collect as collect_references, journal, disjoint
from run_copy_mechanism import read, write
from copy_prior_guard_artifact import candidate, emit

OUT=ROOT/'reference/benchmarks/qwen25_copy_prior_guard_trial'
SOURCES=('py/run_copy_prior_guard_trial.py','py/run_copy_prior_guard.py',
    'py/run_copy_prefix.py','py/run_copy_mechanism.py','py/certificate.py',
    'py/copy_prior_guard_artifact.py','py/copy_longest_artifact.py',
    'dl/copy_prior_tuples.dl','dl/copy_prior_specialize.dl',
    'dl/copy_longest_runtime.dl','dl/copy_long_consensus_runtime.dl',
    'dl/copy_prior_guard_runtime.dl','dl/copy_prior_guard_trial_domains.dl',
    'dl/copy_prior_guard_trial_summary.dl','dl/copy_prior_guard_runtime_check.dl',
    'dl/equiv.dl','docs/copy-prior-guard-confirmation-plan.md')


def scaffold(pools,old_used,old_tokens):
    used={k:set(v) for k,v in old_used.items()}
    # Background membership is defined by actual preceding contexts.
    used['background']=set(pools['background']) & old_tokens
    groups=[]
    for seed in (921,922):
        rng=random.Random(seed)
        def draw(role,n):
            values=rng.sample([t for t in pools[role] if t not in used[role]],n)
            used[role].update(values)
            return values
        for length in (32,48,64):
            for j in range(3):
                for negative in (False,True):
                    groups.append(dict(seed=seed,length=length,negative=negative,
                        group=f'confirm-s{seed}-l{length}-g{j}-n{int(negative)}',
                        query=draw('query',6),target=draw('target',12),decoy=draw('decoy',6),
                        background=draw('background',length+8 if negative else length+40)))
    return groups


def materialize(groups,chosen):
    rows,pairs=[],[]
    for g in groups:
        q,t,u,d=chosen[g['group']]
        length=g['length'];prefix=length//2;bg=iter(g['background'])
        seq=[q if p==prefix-1 else t if p==prefix else next(bg) for p in range(length)]
        head=[next(bg) for _ in range(6)];tail=[next(bg) for _ in range(4)]
        gaps=[] if g['negative'] else [[next(bg) for _ in range(8)] for _ in range(4)]
        if list(bg):raise ValueError('unused allocated background')
        query=seq[:prefix]
        meta=dict(seed=g['seed'],split=f"test_{g['seed']}",group=g['group'],length=length,
                  prefix=prefix,negative=g['negative'])
        if g['negative']:
            for kind,ctx in (('no_repeat',seq),('tie',seq+head+query+[d]+tail+query)):
                rows.append(dict(meta,id=len(rows),kind=kind,ctx=ctx,expected=t))
            continue
        for exposure in (2,3):
            for short in (12,16):
                distractor=head+query[-short:]+[d]+tail
                for order in ('decoy_first','decoy_last'):
                    for gap in (0,8):
                        blocks=([distractor]+[seq]*exposure if order=='decoy_first' else [seq]*exposure+[distractor])
                        ctx,positions=[],[]
                        for j,block in enumerate(blocks):
                            if (j>0 if order=='decoy_first' else j<exposure):positions.append(len(ctx)+prefix)
                            ctx.extend(block);ctx.extend(gaps[j][:gap])
                        ctx.extend(query)
                        fields=dict(meta,exposures=exposure,short=short,order=order,gap=gap)
                        a=len(rows);rows.append(dict(fields,id=a,kind='repeat',ctx=ctx,expected=t))
                        edited=ctx[:]
                        for p in positions:edited[p]=u
                        b=len(rows);rows.append(dict(fields,id=b,kind='intervention',ctx=edited,
                            expected=u,base=a,positions=positions))
                        pairs.append(dict(a=a,b=b,group=g['group'],positions=positions))
    return rows,pairs


def base_facts(rows):
    return {'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'domain':''.join(f"{r['id']}\n" for r in rows),
        'seed':''.join(f"{r['id']}\t{r['seed']}\n" for r in rows)}


def inputs(a):
    pools=feature_gate();old=read(DEVELOPMENT/'dataset.json')
    old_tokens={t for r in old for t in r['ctx']}
    excluded=old_tokens | set(read(LEXICON/'excluded-tokens.json'))
    groups=scaffold(pools,read(DEVELOPMENT/'used-roles.json'),old_tokens)
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'scaffold.json',groups)
    facts={'planned_group':''.join(f"{g['group']}\n" for g in groups),
           'raw_prior':(LEXICON/'prior-bands.facts').read_text()}
    for role,predicate in (('query','q_candidate'),('target','t_candidate'),('decoy','d_candidate')):
        facts[predicate]=''.join(f"{g['group']}\t{t}\t{rank}\n" for g in groups for rank,t in enumerate(g[role]))
    module=ROOT/'dl/copy_prior_tuples.dl'
    tuples=check(module,module,facts,{},evidence_dir=a.out/'certificate-evidence/tuples',
        provenance={'source':'fixed ranked blocks; intrinsic bands only; no full-context references'})
    write(a.out/'tuples.json',tuples)
    if not tuples['certified']:raise ValueError('tuple construction failed; no resampling permitted')
    chosen={g:tuple(map(int,(q,t,u,d))) for g,q,t,u,d in tuples['relations']['chosen']}
    rows,pairs=materialize(groups,chosen);stats=disjoint(rows,excluded)
    write(a.out/'dataset.json',rows);write(a.out/'pairs.json',pairs)
    write(a.out/'excluded-tokens.json',sorted(excluded))
    lexical=specialize(a.out,rows);(a.out/'lexical.dl').write_text(lexical)
    circuit=emit(a.out/'artifact',candidate((5,32,2),lexical))
    (a.out/'analysis.dl').write_text(BASELINE.read_text().replace('cdecide','bdecide')+'\n'+circuit.read_text())
    domains=check(ROOT/'dl/copy_prior_guard_trial_domains.dl',a.out/'analysis.dl',base_facts(rows),{},
        evidence_dir=a.out/'certificate-evidence/domains',provenance={'source':'frozen reference-free confirmation domains'})
    write(a.out/'domains.json',domains)
    if not domains['certified']:raise ValueError(domains.get('error','no incremental domain; trial aborted'))
    exclusions=(DEVELOPMENT/'protocol.json',DEVELOPMENT/'dataset.json',DEVELOPMENT/'used-roles.json',
        LEXICON/'feature.json',LEXICON/'pools.json',LEXICON/'prior-bands.facts')
    protocol={'scope':'independent fixed hypothesis, designed guard population; host agreement only',
        'seeds':[921,922],'option':[5,32,2],'disjoint':stats,
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'exclusions':{str(p.relative_to(ROOT)):sha256(p) for p in exclusions},
        'baseline_sha256':sha256(BASELINE),'pinned_model':read(DEVELOPMENT/'protocol.json')['pinned_model']}
    write(a.out/'protocol.json',protocol)
    write(a.out/'frozen.json',{'protocol_sha256':sha256(a.out/'protocol.json')})
    print(stats,'incremental:',domains['relations']['seed_incremental'],flush=True)


def validate(a):
    if sha256(a.out/'protocol.json')!=read(a.out/'frozen.json')['protocol_sha256']:
        raise ValueError('protocol changed')
    feature_gate()
    return gate(a)


def collect(a):
    validate(a);collect_references(a)


def evaluate(a):
    rows=validate(a);observed=journal(a.out/'observations.jsonl')
    domains=read(a.out/'domains.json');pairs=read(a.out/'pairs.json')
    if set(observed)-{r['id'] for r in rows}:raise ValueError('unplanned references')
    facts=base_facts(rows)
    facts.update(sample=''.join(f"{r['id']}\t{r['group']}\t{len(r['ctx'])-r['prefix']}\n" for r in rows),
        ref=''.join(f"{i}\t{r['response']['next']}\n" for i,r in observed.items()),
        pair=''.join(f"{p['a']}\t{p['b']}\t{p['group']}\n" for p in pairs),
        edit=''.join(f"{p['a']}\t{p['b']}\t{j}\n" for p in pairs for j in p['positions']))
    summary=check(ROOT/'dl/copy_prior_guard_trial_summary.dl',a.out/'analysis.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/summary',provenance={'source':'fixed independent confirmation, all answers retained'})
    write(a.out/'summary.json',summary)
    certificates={}
    for name in ('firing','incremental','full'):
        ids=({r['id'] for r in rows} if name=='full' else {int(i) for i, in domains['relations'][name]})
        staged={'domain':''.join(f'{i}\n' for i in sorted(ids)),
            'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows if r['id'] in ids for p,t in enumerate(r['ctx'])),
            'ref':''.join(f"{i}\t{observed[i]['response']['next']}\n" for i in sorted(ids) if i in observed)}
        certificates[name]=check(ROOT/'dl/equiv.dl',a.out/'artifact/circuits.dl',staged,
            {'ndomain':int,'nmiss':int,'nuncov':int,'nmissing':int,'ninvalid':int},
            evidence_dir=a.out/f'certificate-evidence/equiv-{name}',provenance={'source':f'fixed confirmation {name} domain'})
    with tempfile.TemporaryDirectory() as scratch:
        temp=Path(scratch);(temp/'facts').mkdir();(temp/'outputs').mkdir()
        for n in ('circuits.dl','run.dl'):shutil.copyfile(a.out/'artifact'/n,temp/n)
        for n in ('tok','domain'):(temp/'facts'/f'{n}.facts').write_text(facts[n])
        proc=subprocess.run(['souffle','run.dl','-F','facts','-D','outputs'],cwd=temp,capture_output=True,text=True)
        if proc.returncode:raise ValueError(proc.stderr)
        module=ROOT/'dl/copy_prior_guard_runtime_check.dl'
        runtime=check(module,module,{'domain':facts['domain'],
            'expected_decision':''.join(f'{i}\t{t}\n' for i,t in domains['relations']['prediction']),
            'actual_decision':(temp/'outputs/cdecide.csv').read_text(),
            'actual_abstain':(temp/'outputs/abstain.csv').read_text()}, {},
            evidence_dir=a.out/'certificate-evidence/runtime',provenance={'source':'relocated Souffle-only runtime'})
        write(a.out/'runtime.json',runtime)
    write(a.out/'certificates.json',certificates)
    audits=[]
    for result in (read(a.out/'tuples.json'),read(a.out/'specialization.json'),domains,summary,*certificates.values(),runtime):
        again=replay(Path(result['evidence']))
        if again['relations']!=result['relations'] or again['certified']!=result['certified']:
            raise ValueError('model-free replay differs')
        audits.append(result['evidence'])
    validate(a)
    write(a.out/'replay-audit.json',{'queries':audits,'unchanged':True})
    # These booleans are verdicts computed by the retained Datalog queries.
    passed=all(x['certified'] for x in (summary,certificates['firing'],certificates['incremental'],runtime))
    write(a.out/'outcome.json',{'passed':passed,'scope':'stated finite confirmation domains only',
        'artifact_sha256':sha256(a.out/'artifact/circuits.dl')})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('passed:',passed,'counts:',summary['relations'].get('counts'),'causal:',summary['relations'].get('causal_groups'))
    for name,result in certificates.items():print(name,{k:result.get(k) for k in ('certified','ndomain','nmiss','nuncov','nmissing','ninvalid')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('inputs','collect','evaluate'))
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--binary',type=Path);parser.add_argument('--bundle',type=Path)
    parser.add_argument('--port',type=int,default=8396)
    a=parser.parse_args();a.out=a.out.resolve()
    if a.phase=='collect' and not(a.binary and a.bundle):parser.error('collection requires pinned binary and bundle')
    globals()[a.phase](a)


if __name__=='__main__':main()
