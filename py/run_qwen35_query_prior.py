"""Frozen training-only diagnostic of query-history-conditioned lexical preferences."""
import argparse
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_copy_inventory import ROOT,OUT as INVENTORY,MODEL,gate,journal,read,write
from qwen35_reference_evidence import extract,replay_archive
OUT=ROOT/'reference/benchmarks/qwen35_query_prior'
SOURCES=('py/run_qwen35_query_prior.py','dl/qwen35_query_prior_plan.dl','dl/qwen35_query_prior_keys.dl','dl/qwen35_query_prior_summary.dl','dl/copy_longest_runtime.dl','py/qwen35_mlp_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','py/qwen35_reference_evidence.py','py/certificate.py','dl/qwen35_reference_argmax.dl')
def inputs(a):
    rows=[r for r in gate(type('Args',(),{'out':INVENTORY})()) if r['split']=='train' and not r['negative']]
    a.out.mkdir(parents=True,exist_ok=False);write(a.out/'training.json',rows)
    observed=journal(INVENTORY);write(a.out/'references.json',{r['id']:observed[r['id']]['next'] for r in rows})
    m=ROOT/'dl/qwen35_query_prior_plan.dl'
    x=check(m,m,{'sample':''.join(f"{r['id']}\t{r['group']}\t{r['prefix']}\n" for r in rows)},{},evidence_dir=a.out/'certificate-evidence/plan',provenance={'source':'three fixed query-history lengths, training groups only'})
    write(a.out/'plan.json',x)
    if not x['certified']:raise ValueError('invalid plan')
    byid={r['id']:r for r in rows};queries=[]
    for q,i,k,v in x['relations']['query']:
        r=byid[int(i)];queries.append({'id':int(q),'instance':int(i),'variant':int(v),'group':r['group'],'ctx':r['ctx'][-int(k):]})
    write(a.out/'queries.json',queries)
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['group']}\n" for r in rows),
        'query':''.join(f"{r['id']}\t{r['group']}\t{r['variant']}\n" for r in queries)}
    write(a.out/'facts.json',facts);(a.out/'input-core.dl').write_bytes((INVENTORY/'input-core.dl').read_bytes())
    m=ROOT/'dl/qwen35_query_prior_keys.dl'
    keys=check(m,a.out/'input-core.dl',facts,{},evidence_dir=a.out/'certificate-evidence/keys',provenance={'source':'source-token readout keys chosen from tokens only, no answers'})
    write(a.out/'keys.json',keys)
    if not keys['certified']:raise ValueError('missing keys')
    write(a.out/'protocol.json',{'scope':'post-hoc training diagnostic; all training answers already known; not candidate admission or final confirmation',
        'queries':len(queries),'variants':['last1','last4','full_query'],'rule':'rank observed matching-source successors by isolated-query logits; report prior agreement and fixed unique-long-source guard >=24, competitor>=12, margin>=4; no tuning',
        'validation_labels_used':False,'prior':{'path':str((INVENTORY/'closed.json').relative_to(ROOT)),'sha256':sha256(INVENTORY/'closed.json')},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen',len(queries),'query-only forwards',flush=True)
def checked(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('frozen input changed')
    for n,h in p['sources'].items():
        if sha256(ROOT/n)!=h:raise ValueError('source changed')
    if sha256(ROOT/p['prior']['path'])!=p['prior']['sha256']:raise ValueError('inventory changed')
    return p
def collect(a):
    checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_mlp_reference import MlpReference
    torch.set_num_threads(2);model=MlpReference(MODEL,vocab_chunk=1024);units=[tuple(map(int,r)) for r in read(a.out/'plan.json')['relations']['unit']]
    for r in read(a.out/'queries.json'):
        path=a.out/f"query-{r['id']}.pt"
        if path.exists():continue
        data=model.probe(r['ctx'],units,());tmp=path.with_suffix('.pending');torch.save(data,tmp);tmp.replace(path)
        print('completed query',r['id'],'variant',r['variant'],flush=True)
def evaluate(a):
    checked(a)
    import torch
    facts=read(a.out/'facts.json');facts['ref']=''.join(f'{i}\t{t}\n' for i,t in read(a.out/'references.json').items());facts['prior']=''
    keys=read(a.out/'keys.json')['relations']['key']
    for r in read(a.out/'queries.json'):
        q=r['id'];scores=torch.load(a.out/f'query-{q}.pt',weights_only=True)['logits'];path=a.out/'reference-evidence'/f'reference-{q:04d}.tar.gz'
        if path.exists():
            h=sha256(path);e={'archive':path.name,'sha256':h,'query':replay_archive(path,h,row_id=q)}
        else:e=extract(scores,path,q)
        after=replay_archive(path,e['sha256'],row_id=q)
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('reference replay differs')
        write(a.out/f'reference-{q}.json',e)
        facts['prior']+=''.join(f'{q}\t{t}\t{float(scores[int(t)]):.17g}\n' for qq,t in keys if int(qq)==q)
    m=ROOT/'dl/qwen35_query_prior_summary.dl'
    x=check(m,a.out/'input-core.dl',facts,{},evidence_dir=a.out/'certificate-evidence/summary',provenance={'source':'fixed isolated-query source preference diagnostic; not equivalence'})
    write(a.out/'summary.json',x)
    for name in ('plan','keys','summary'):
        before=read(a.out/f'{name}.json');after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    write(a.out/'replay-audit.json',{'reference_witnesses':len(read(a.out/'queries.json')),'queries':3,'unchanged':True,'summary_certified':x['certified']})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('valid',x['certified'],'prior agreement',x['relations']['priorcounts'],'guarded',x['relations']['counts'],flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','collect','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();{'inputs':inputs,'collect':collect,'evaluate':evaluate}[a.phase](a)
if __name__=='__main__':main()
