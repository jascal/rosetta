"""Frozen sparse gate-change screen; Datalog chooses all subsets and payloads."""
import argparse,random
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_mlp_factor import ROOT,OUT as FACTOR,PRIOR,read,write,run as collect,checked
from qwen35_reference_evidence import extract,replay_archive
OUT=ROOT/'reference/benchmarks/qwen35_sparse_gate'
SOURCES=('py/run_qwen35_sparse_gate.py','dl/qwen35_sparse_gate_payload.dl','py/run_qwen35_mlp_factor.py','dl/qwen35_mlp_factor_summary.dl','py/qwen35_mlp_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','py/qwen35_reference_evidence.py','py/certificate.py','dl/qwen35_reference_argmax.dl')
def inputs(a):
    import torch
    for root in (FACTOR,PRIOR):
        for n,h in read(root/'closed.json')['files'].items():
            if sha256(root/n)!=h:raise ValueError('prior evidence changed')
    if not read(FACTOR/'summary.json')['certified']:raise ValueError('factor controls failed')
    report=read(ROOT/'reference/benchmarks/qwen35_mlp_factor_replay.json')
    if not report['unchanged'] or report['closed_sha256']!=sha256(FACTOR/'closed.json'):raise ValueError('factor replay invalid')
    a.out.mkdir(parents=True,exist_ok=False);contexts=read(FACTOR/'contexts.json');write(a.out/'contexts.json',contexts)
    factors=read(FACTOR/'payload.json')['relations'];hybrids={int(i):{} for i in (r['id'] for r in contexts)}
    for arm,target,donor,mode in factors['run']:
        if mode=='donor_gate':
            hybrids[int(target)]={int(n):v for aa,l,p,n,v in factors['patch'] if aa==arm}
    product=''
    for r in contexts:
        data=torch.load(PRIOR/f"arm-{100*r['id']+99}.pt",weights_only=True)
        product+=''.join(f"{r['id']}\t{x['neuron']}\t{x['product']:.17g}\t{hybrids[r['id']][x['neuron']]}\n" for x in data['observations'] if x['layer']==31)
    order=list(range(9216));random.Random(2200).shuffle(order)
    m=ROOT/'dl/qwen35_sparse_gate_payload.dl'
    x=check(m,m,{'shape':'9216\n','context':''.join(f"{r['id']}\t{len(r['ctx'])-1}\n" for r in contexts),
        'donor':''.join(f'{t}\t{d}\n' for aa,t,d,mode in factors['run'] if mode=='donor_gate'),
        'budget':'32\n128\n512\n','product':product,'random_order':''.join(f'{n}\t{rank}\n' for rank,n in enumerate(order))},{},
        evidence_dir=a.out/'certificate-evidence/payload',provenance={'source':'absolute gate-product delta ranking; fixed budgets and seed2200 random neuron order; no new outcomes'})
    write(a.out/'payload.json',x)
    if not x['certified']:raise ValueError('sparse plan invalid')
    after=replay(Path(x['evidence']))
    if after['relations']!=x['relations']:raise ValueError('payload replay differs')
    write(a.out/'protocol.json',{'scope':'development sparse gate screen; no circuit/final certificate','forwards':len(x['relations']['run']),
        'selection':'absolute donor-gate induced product change descending, neuron ID ties; budgets32/128/512; fixed seed2200 random controls; complement top128; all six directions',
        'interpretation':'sufficiency screen only; random controls may themselves transfer; complement tests redundancy/distribution; no outcome-dependent subset retuning',
        'prior':{'path':str(FACTOR.relative_to(ROOT)/'closed.json'),'sha256':sha256(FACTOR/'closed.json')},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen',len(x['relations']['run']),'forwards',flush=True)
def evaluate(a):
    _,plan=checked(a)
    import torch
    contexts=read(a.out/'contexts.json');facts={'run':''.join('\t'.join(r)+'\n' for r in plan['run']),
        'expected':''.join(f"{r['id']}\t{r['expected']}\n" for r in contexts),'ref':'','reconstruction':'',
        'request':''.join('\t'.join(r)+'\n' for r in plan['patch']),'applied':''}
    for aa,tt,dd,mode in plan['run']:
        arm=int(aa);data=torch.load(a.out/f'arm-{arm}.pt',weights_only=True);path=a.out/'reference-evidence'/f'reference-{arm:04d}.tar.gz'
        if path.exists():
            h=sha256(path);e={'archive':path.name,'sha256':h,'query':replay_archive(path,h,row_id=arm)}
        else:e=extract(data['logits'],path,arm)
        after=replay_archive(path,e['sha256'],row_id=arm)
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('argmax replay differs')
        write(a.out/f'reference-{arm}.json',e);facts['ref']+=f"{arm}\t{after['relations']['ref'][0][0]}\n"
        facts['applied']+=''.join(f"{arm}\t{r['layer']}\t{r['position']}\t{r['neuron']}\t{r['before']:.17g}\t{r['after']:.17g}\n" for r in data['applied'])
    m=ROOT/'dl/qwen35_mlp_factor_summary.dl'
    x=check(m,m,facts,{},evidence_dir=a.out/'certificate-evidence/summary',provenance={'source':'frozen sparse gate screen, not standalone equivalence'})
    write(a.out/'summary.json',x)
    for before in (read(a.out/'payload.json'),x):
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    write(a.out/'replay-audit.json',{'reference_witnesses':len(plan['run']),'queries':2,'unchanged':True,'summary_certified':x['certified']})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('valid',x['certified'],'transfers',x['relations']['counts'],flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();{'inputs':inputs,'run':collect,'evaluate':evaluate}[a.phase](a)
if __name__=='__main__':main()
