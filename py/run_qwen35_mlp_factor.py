"""Frozen last-MLP gate/up factorization, with numerical reconstruction controls."""
import argparse
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_routing_mlp import ROOT,MODEL,OUT as PRIOR,checked as prior_checked,read,write
from run_qwen35_copy_inventory import OUT as INVENTORY,gate
from qwen35_reference_evidence import extract,replay_archive
OUT=ROOT/'reference/benchmarks/qwen35_mlp_factor'
SOURCES=('py/run_qwen35_mlp_factor.py','dl/qwen35_mlp_factor_payload.dl','dl/qwen35_factor_reconstruction.dl','dl/qwen35_mlp_factor_summary.dl','py/qwen35_mlp_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','py/qwen35_reference_evidence.py','py/certificate.py','dl/qwen35_reference_argmax.dl')
def inputs(a):
    import torch
    prior_checked(type('Args',(),{'out':PRIOR})())
    report=read(ROOT/'reference/benchmarks/qwen35_routing_mlp_replay.json')
    if not report['unchanged'] or sha256(PRIOR/'closed.json')!=report['closed_sha256']:raise ValueError('prior replay invalid')
    for n,h in read(PRIOR/'closed.json')['files'].items():
        if sha256(PRIOR/n)!=h:raise ValueError('prior file changed')
    a.out.mkdir(parents=True,exist_ok=False)
    plan=read(PRIOR/'plan.json')['relations'];contexts=read(PRIOR/'contexts.json');write(a.out/'contexts.json',contexts)
    observations=''
    for r in contexts:
        result=torch.load(PRIOR/f"arm-{100*r['id']+99}.pt",weights_only=True)
        observations+=''.join(f"{r['id']}\t{x['neuron']}\t{x['gate']:.17g}\t{x['value']:.17g}\n" for x in result['observations'] if x['layer']==31)
    m=ROOT/'dl/qwen35_mlp_factor_payload.dl'
    result=check(m,m,{'shape':'9216\n','context':''.join(f"{r['id']}\t{len(r['ctx'])-1}\n" for r in contexts),
        'donor':''.join(f'{t}\t{d}\n' for _,t,d,l,mode in plan['run'] if l=='31' and mode=='donor'),
        'observation':observations},{},evidence_dir=a.out/'certificate-evidence/payload',provenance={'source':'all six previous directed contrasts, no filtering by transfer; Datalog gate/up hybrids'})
    write(a.out/'payload.json',result)
    if not result['certified']:raise ValueError('payload incomplete')
    assert replay(Path(result['evidence']))['relations']==result['relations']
    write(a.out/'protocol.json',{'scope':'development factorization, not standalone replacement','layer':31,'width':9216,'forwards':24,
        'arms':'reconstructed target, reconstructed donor, donor gate times target up, target gate times donor up; all six directed contrasts',
        'numerics':'stable double-precision Datalog SiLU then model FP32 substitution; reconstructed controls must match prior exact-product full logits within .002+.002abs and argmax exactly',
        'prior':{'path':str(PRIOR.relative_to(ROOT)/'closed.json'),'sha256':sha256(PRIOR/'closed.json')},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen',len(result['relations']['run']),'forwards, all payloads retained',flush=True)
def checked(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('frozen input changed')
    for n,h in p['sources'].items():
        if sha256(ROOT/n)!=h:raise ValueError('frozen source changed')
    if sha256(ROOT/p['prior']['path'])!=p['prior']['sha256']:raise ValueError('prior closure changed')
    return p,read(a.out/'payload.json')['relations']
def run(a):
    _,plan=checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_mlp_reference import MlpReference
    torch.set_num_threads(2);model=MlpReference(MODEL,vocab_chunk=1024)
    contexts={r['id']:r for r in read(a.out/'contexts.json')}
    for j,(aa,tt,dd,mode) in enumerate(sorted(plan['run'],key=lambda r:int(r[0]))):
        arm,target=int(aa),int(tt);path=a.out/f'arm-{arm}.pt'
        if path.exists():continue
        patches=[(int(l),int(p),int(n),float(v)) for a,l,p,n,v in plan['patch'] if int(a)==arm]
        result=model.probe(contexts[target]['ctx'],(),patches)
        tmp=path.with_suffix('.pending');torch.save(result,tmp);tmp.replace(path)
        print('completed',j+1,'/',len(plan['run']),arm,mode,flush=True)
def evaluate(a):
    _,plan=checked(a)
    import torch
    contexts=read(a.out/'contexts.json');facts={'run':''.join('\t'.join(r)+'\n' for r in plan['run']),
        'expected':''.join(f"{r['id']}\t{r['expected']}\n" for r in contexts),'ref':'','reconstruction':'',
        'request':''.join('\t'.join(r)+'\n' for r in plan['patch']),'applied':''}
    results=[]
    for aa,tt,dd,mode in plan['run']:
        arm,target=int(aa),int(tt);result=torch.load(a.out/f'arm-{arm}.pt',weights_only=True);scores=result['logits']
        path=a.out/'reference-evidence'/f'reference-{arm:04d}.tar.gz'
        if path.exists():
            h=sha256(path);e={'archive':path.name,'sha256':h,'query':replay_archive(path,h,row_id=arm)}
        else:e=extract(scores,path,arm)
        after=replay_archive(path,e['sha256'],row_id=arm)
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('reference replay differs')
        write(a.out/f'reference-{arm}.json',e);facts['ref']+=f"{arm}\t{after['relations']['ref'][0][0]}\n"
        facts['applied']+=''.join(f"{arm}\t{r['layer']}\t{r['position']}\t{r['neuron']}\t{r['before']:.17g}\t{r['after']:.17g}\n" for r in result['applied'])
        if mode in ('self_reconstructed','donor_reconstructed'):
            oldarm=100*target+(99 if mode=='self_reconstructed' else 95)
            original=torch.load(PRIOR/f'arm-{oldarm}.pt',weights_only=True)['logits'];m=ROOT/'dl/qwen35_factor_reconstruction.dl'
            x=check(m,m,{'vocab':f'{len(scores)}\n','score':''.join(f'{side}\t{t}\t{v:.17g}\n' for side,values in (('original',original),('reconstructed',scores)) for t,v in enumerate(values.tolist()))},{},evidence_dir=a.out/f'certificate-evidence/reconstruction-{arm}',provenance={'source':'double-Datalog reconstructed products versus earlier exact FP32 products'})
            write(a.out/f'reconstruction-{arm}.json',x);results.append(x);facts['reconstruction']+=f"{arm}\t{int(x['certified'])}\n"
    m=ROOT/'dl/qwen35_mlp_factor_summary.dl'
    x=check(m,m,facts,{},evidence_dir=a.out/'certificate-evidence/summary',provenance={'source':'MLP gate/up factorization; interpretation requires all reconstruction controls'})
    write(a.out/'summary.json',x);results.extend((read(a.out/'payload.json'),x))
    for before in results:
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    write(a.out/'replay-audit.json',{'reference_witnesses':24,'queries':len(results),'unchanged':True,'summary_certified':x['certified']})
    write(a.out/'closed.json',{'files':{str(f.relative_to(a.out)):sha256(f) for f in a.out.rglob('*') if f.is_file() and f.name!='closed.json'}})
    print('controls valid',x['certified'],'transfers',x['relations']['counts'],flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();globals()[a.phase](a)
if __name__=='__main__':main()
