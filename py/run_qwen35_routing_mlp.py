"""Frozen development routing contrast: whole-MLP identity/zero/donor tests."""
import argparse,hashlib,json,time
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_copy_inventory import ROOT,MODEL,OUT as INVENTORY,gate,read,write
from run_qwen35_decoy_counterfactual import OUT as DECOY
from qwen35_reference_evidence import extract,replay_archive
OUT=ROOT/'reference/benchmarks/qwen35_routing_mlp'
SOURCES=('py/run_qwen35_routing_mlp.py','dl/qwen35_routing_mlp_plan.dl','dl/qwen35_routing_mlp_summary.dl','dl/qwen35_exchange_payload.dl','py/qwen35_mlp_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','py/qwen35_reference_evidence.py','py/certificate.py','dl/qwen35_reference_argmax.dl')
def query(a,name,facts):
    module=ROOT/('dl/qwen35_exchange_payload.dl' if name.startswith('payload') else f'dl/qwen35_routing_mlp_{name}.dl')
    x=check(module,module,facts,{},evidence_dir=a.out/f'certificate-evidence/{name}',provenance={'source':'development MLP routing interventions; not replacement equivalence'})
    write(a.out/f'{name}.json',x)
    if not x['certified']:raise ValueError(f'{name} failed: {x.get("error",x["relations"].get("invalid"))}')
    return x

def inputs(a):
    original={r['id']:r for r in gate(type('Args',(),{'out':INVENTORY})())}
    rows=read(DECOY/'dataset.json');summary=read(DECOY/'summary.json')
    if not summary['certified'] or not read(DECOY/'replay-audit.json')['unchanged']:raise ValueError('unverified contrast study')
    observed={r['id']:r for r in map(json.loads,(DECOY/'observations.jsonl').read_text().splitlines())}
    a.out.mkdir(parents=True,exist_ok=False)
    plan=query(a,'plan',{'contrast':''.join(f"{r['id']}\t{r['group']}\t{r['base_reference']}\t{observed[r['id']]['next']}\t{r['long_target']}\t{len(r['ctx'])-1}\n" for r in rows)})
    byid={r['id']:r for r in rows};contexts=[]
    for i,k,side,t,p in plan['relations']['context']:
        r=byid[int(k)];ctx=original[r['base']]['ctx'] if side=='old' else r['ctx']
        contexts.append({'id':int(i),'contrast':int(k),'side':side,'expected':int(t),'ctx':ctx})
    write(a.out/'contexts.json',contexts)
    write(a.out/'protocol.json',{'scope':'whole-width last-position MLP donor transfer, development only','layers':[0,3,15,31],'width':9216,
        'selection':'lowest-ID changed-to-long counterfactual per vocabulary group, both directions',
        'arms':'baseline plus identity, zero, donor at each layer; all 9216 products before down projection; no other sites',
        'interpretation':'donor-answer transfer is causal sensitivity, not standalone computation or coverage; preserve all failures',
        'priors':{str(p.relative_to(ROOT)):sha256(p) for p in (DECOY/'closed.json',INVENTORY/'closed.json')},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen',len(contexts),'contexts',len(plan['relations']['run']),'forwards',flush=True)

def checked(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('frozen input changed')
    for n,h in (p['sources']|p['priors']).items():
        if sha256(ROOT/n)!=h:raise ValueError('source/prior changed')
    return p,read(a.out/'plan.json')['relations']

def run(a):
    p,plan=checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_mlp_reference import MlpReference
    torch.set_num_threads(2);model=MlpReference(MODEL,vocab_chunk=1024)
    if model.config.intermediate_size!=p['width']:raise ValueError('width mismatch')
    contexts={r['id']:r for r in read(a.out/'contexts.json')}
    arms=sorted(plan['run'],key=lambda r:(r[4]!='baseline',int(r[0])))
    for j,(aa,tt,dd,ll,mode) in enumerate(arms):
        arm,target,donor,layer=map(int,(aa,tt,dd,ll));path=a.out/f'arm-{arm}.pt'
        if path.exists():continue
        ids=contexts[target]['ctx'];patches=[];payload=None
        if mode=='baseline':units=[(l,n) for l in p['layers'] for n in range(p['width'])]
        else:
            units=[];obs=''
            for i in {target,donor}:
                baseline=torch.load(a.out/f'arm-{100*i+99}.pt',weights_only=True)
                obs+=''.join(f"{i}\t{r['layer']}\t{r['neuron']}\t{r['product']:.17g}\n" for r in baseline['observations'] if r['layer']==layer)
            payload=query(a,f'payload-{arm}',{'shape':f"{p['width']}\n",'request':f'{arm}\t{target}\t{donor}\t{layer}\t{len(ids)-1}\t{mode}\n','observation':obs})
            patches=[(int(l),int(pos),int(n),float(v)) for _,l,pos,n,v in payload['relations']['patch']]
        started=time.monotonic();result=model.probe(ids,units,patches)
        temporary=path.with_suffix('.pending');torch.save(result,temporary);temporary.replace(path)
        print('completed',j+1,'/',len(arms),'arm',arm,mode,f'{time.monotonic()-started:.2f}s',flush=True)
    print('all interventions retained; run evaluate',flush=True)

def evaluate(a):
    p,plan=checked(a)
    import torch
    facts={'run':''.join('\t'.join(r)+'\n' for r in plan['run']),
        'expected':''.join(f'{i}\t{t}\n' for i,_,_,t,_ in plan['context']),
        'ref':'','hash':'','request':'','applied':''}
    for aa,tt,dd,ll,mode in plan['run']:
        arm=int(aa);r=torch.load(a.out/f'arm-{arm}.pt',weights_only=True);scores=r['logits'];path=a.out/'reference-evidence'/f'reference-{arm:04d}.tar.gz'
        if path.exists():
            h=sha256(path);e={'archive':path.name,'sha256':h,'query':replay_archive(path,h,row_id=arm)}
        else:e=extract(scores,path,arm)
        write(a.out/f'reference-{arm}.json',e)
        facts['ref']+=f"{arm}\t{e['query']['relations']['ref'][0][0]}\n"
        facts['hash']+=f'{arm}\t{hashlib.sha256(scores.numpy().tobytes()).hexdigest()}\n'
        if mode!='baseline':
            payload=read(a.out/f'payload-{arm}.json');after=replay(Path(payload['evidence']))
            if after['relations']!=payload['relations'] or not after['certified']:raise ValueError('payload replay differs')
            facts['request']+=''.join('\t'.join(row)+'\n' for row in payload['relations']['patch'])
            facts['applied']+=''.join(f"{arm}\t{r['layer']}\t{r['position']}\t{r['neuron']}\t{r['before']:.17g}\t{r['after']:.17g}\n" for r in r['applied'])
    x=query(a,'summary',facts)
    for name in ('plan','summary'):
        before=read(a.out/f'{name}.json');after=replay(Path(before['evidence']))
        if after['relations']!=before['relations'] or not after['certified']:raise ValueError('query replay differs')
    write(a.out/'closed.json',{'files':{str(f.relative_to(a.out)):sha256(f) for f in a.out.rglob('*') if f.is_file() and f.name!='closed.json'}})
    print('counts',x['relations']['counts'],'transfers',x['relations']['transfers'])
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();globals()[a.phase](a)
if __name__=='__main__':main()
