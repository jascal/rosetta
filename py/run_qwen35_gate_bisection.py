"""Datalog-planned adaptive development search; every round frozen before refs."""
import argparse,json
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_shared_gate import ROOT,OUT as PRIOR,read,write
from run_qwen35_sparse_gate import OUT as SPARSE
from run_qwen35_copy_inventory import OUT as INVENTORY,MODEL,gate
from qwen35_reference_evidence import extract,replay_archive
OUT=ROOT/'reference/benchmarks/qwen35_gate_bisection'
SOURCES=('py/run_qwen35_gate_bisection.py','dl/qwen35_gate_bisection_inputs.dl','dl/qwen35_gate_bisection.dl','dl/qwen35_mlp_factor_summary.dl','py/qwen35_mlp_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','py/qwen35_reference_evidence.py','py/certificate.py','dl/qwen35_reference_argmax.dl','docs/qwen35-gate-bisection.md')
def query(out,name,module,facts):
    x=check(ROOT/module,ROOT/module,facts,{},evidence_dir=out/'certificate-evidence'/name,provenance={'source':'development bisection planning/application consistency; not replacement certification'})
    write(out/f'{name}.json',x)
    if not x['certified']:raise ValueError(f'{name} query failed')
    return x
def inputs(a):
    for root in (PRIOR,SPARSE):
        for n,h in read(root/'closed.json')['files'].items():
            if sha256(root/n)!=h:raise ValueError('prior changed')
    report=read(ROOT/'reference/benchmarks/qwen35_shared_gate_replay.json')
    if not report['unchanged'] or report['closed_sha256']!=sha256(PRIOR/'closed.json'):raise ValueError('prior replay invalid')
    
    if (a.out/'protocol.json').exists():raise ValueError('refusing to overwrite frozen inputs')
    a.out.mkdir(parents=True,exist_ok=True);contexts=read(PRIOR/'contexts.json');write(a.out/'contexts.json',contexts)
    plan=read(PRIOR/'payload.json')['relations'];ranks=read(SPARSE/'payload.json')['relations']['rank']
    roots=[r for r in plan['run'] if r[3]=='rest_top32'];reflinks={}
    for arm,i,d,mode in roots:reflinks[i]=read(PRIOR/f'reference-{arm}.json')
    write(a.out/'root-witnesses.json',reflinks)
    x=query(a.out,'inputs','dl/qwen35_gate_bisection_inputs.dl',{
        'context':''.join(f"{r['id']}\t{len(r['ctx'])-1}\t{r['expected']}\n" for r in contexts),
        'donor':''.join(f'{i}\t{d}\n' for arm,i,d,mode in roots),
        'root':''.join(f'{i}\t{n}\t{v}\n' for arm,i,d,mode in roots for aa,l,p,n,v in plan['patch'] if arm==aa),
        'rank':''.join('\t'.join(r)+'\n' for r in ranks),
        'root_ref':''.join(f"{i}\t{e['query']['relations']['ref'][0][0]}\n" for i,e in reflinks.items())})
    write(a.out/'initial.json',{k:x['relations'][k] for k in ('caseinfo','member','ref')})
    write(a.out/'donors.json',{i:d for arm,i,d,mode in roots})
    write(a.out/'protocol.json',{'scope':'adaptive development bisection; no final-test or runtime artifact','max_rounds':6,'max_new_forwards':30,
        'rule':'both halves observed before branch; left if success else right if success; otherwise stop; failed roots stop; singleton success stops',
        'priors':{str((r/'closed.json').relative_to(ROOT)):sha256(r/'closed.json') for r in (PRIOR,SPARSE)},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen six roots; maximum30 new forwards',flush=True)
def checked(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('frozen input changed')
    for n,h in (p['sources']|p['priors']).items():
        if sha256(ROOT/n)!=h:raise ValueError('source/prior changed')
    return p
def journal(out):
    path=out/'observations.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    result={(r['case'],r['node']):r for r in rows}
    if len(result)!=len(rows):raise ValueError('duplicate observation')
    return result
def run(a):
    protocol=checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_mlp_reference import MlpReference
    torch.set_num_threads(2);model=MlpReference(MODEL,vocab_chunk=1024)
    contexts={r['id']:r for r in read(a.out/'contexts.json')};donors=read(a.out/'donors.json');initial=read(a.out/'initial.json');observed=journal(a.out)
    for roundid in range(protocol['max_rounds']):
        root=a.out/f'round-{roundid}';root.mkdir(exist_ok=True)
        frozen=root/'frozen.json'
        if frozen.exists():
            for n,h in read(frozen)['files'].items():
                if sha256(root/n)!=h:raise ValueError('round plan changed')
            plan=read(root/'plan.json')
        else:
            facts={k:''.join('\t'.join(row)+'\n' for row in rows) for k,rows in initial.items()}
            facts['ref']+=''.join(f"{r['case']}\t{r['node']}\t{r['next']}\n" for r in observed.values())
            plan=query(root,'plan','dl/qwen35_gate_bisection.dl',facts)
            write(frozen,{'files':{str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file() and p.name!='frozen.json'}})
        pending=plan['relations']['pending']
        if not pending:
            write(a.out/'outcome.json',{'last_round':roundid,'relations':plan['relations'],'new_forwards':len(observed)})
            for before in [read(a.out/'inputs.json')]+[read(p) for p in a.out.glob('round-*/plan.json')]+[read(p) for p in a.out.glob('site-*.json')]:
                after=replay(Path(before['evidence']))
                if before['relations']!=after['relations'] or not after['certified']:raise ValueError('query replay differs')
            for r in observed.values():
                e=r['reference'];after=replay_archive(a.out/'reference-evidence'/e['archive'],e['sha256'],row_id=r['case']*100+ r['node'])
                if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('argmax replay differs')
            write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
            print('finished',len(observed),'new forwards; leaves',plan['relations']['leaf'],'stops',plan['relations']['interaction_stop'],flush=True);return
        for cc,nn in sorted(pending,key=lambda r:tuple(map(int,r))):
            case,node=int(cc),int(nn)
            if (case,node) in observed:continue
            arm=100*case+node;patchrows=[(int(l),int(p),int(n),float(v)) for c,nd,l,p,n,v in plan['relations']['patch'] if c==cc and nd==nn]
            path=a.out/f'arm-{arm}.pt'
            if path.exists():data=torch.load(path,weights_only=True)
            else:
                data=model.probe(contexts[case]['ctx'],(),patchrows);tmp=path.with_suffix('.pending');torch.save(data,tmp);tmp.replace(path)
            archive=a.out/'reference-evidence'/f'reference-{arm:04d}.tar.gz'
            if archive.exists():
                h=sha256(archive);e={'archive':archive.name,'sha256':h,'query':replay_archive(archive,h,row_id=arm)}
            else:e=extract(data['logits'],archive,arm)
            token=int(e['query']['relations']['ref'][0][0]);d=int(donors[cc])
            query(a.out,f'site-{arm}','dl/qwen35_mlp_factor_summary.dl',{
                'run':f'{arm}\t{case}\t{d}\tbisect\n','expected':f"{case}\t{contexts[case]['expected']}\n{d}\t{contexts[d]['expected']}\n",
                'ref':f'{arm}\t{token}\n','reconstruction':'','request':''.join(f'{arm}\t{l}\t{p}\t{n}\t{v:.17g}\n' for l,p,n,v in patchrows),
                'applied':''.join(f"{arm}\t{r['layer']}\t{r['position']}\t{r['neuron']}\t{r['before']:.17g}\t{r['after']:.17g}\n" for r in data['applied'])})
            record={'case':case,'node':node,'next':token,'round':roundid,'reference':e}
            with (a.out/'observations.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            observed[case,node]=record;print('round',roundid,'case',case,'node',node,'size',len(patchrows),'ref',token,flush=True)
    raise ValueError('round bound reached without terminal query')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();{'inputs':inputs,'run':run}[a.phase](a)
if __name__=='__main__':main()
