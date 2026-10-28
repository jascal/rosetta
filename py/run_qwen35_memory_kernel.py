"""Frozen minimal association probes and fixed source-mixture diagnostics."""
import argparse
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_query_prior import ROOT,OUT as QUERIES,read,write
from run_qwen35_partial_prior import OUT as PARTIAL
from run_qwen35_copy_inventory import OUT as INVENTORY,MODEL,gate
from qwen35_reference_evidence import extract,replay_archive
OUT=ROOT/'reference/benchmarks/qwen35_memory_kernel'
SOURCES=('py/run_qwen35_memory_kernel.py','dl/qwen35_memory_kernel_plan.dl','dl/qwen35_memory_kernel_summary.dl','dl/copy_longest_runtime.dl','py/qwen35_mlp_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','py/qwen35_reference_evidence.py','py/certificate.py','dl/qwen35_reference_argmax.dl')
def inputs(a):
    for root in (QUERIES,PARTIAL):
        for n,h in read(root/'closed.json')['files'].items():
            if sha256(root/n)!=h:raise ValueError('prior changed')
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ('input-core.dl','facts.json','references.json'):(a.out/name).write_bytes((QUERIES/name).read_bytes())
    m=ROOT/'dl/qwen35_memory_kernel_plan.dl'
    x=check(m,m,{'query':read(QUERIES/'facts.json')['query'],'key':''.join('\t'.join(r)+'\n' for r in read(QUERIES/'keys.json')['relations']['key'])},{},evidence_dir=a.out/'certificate-evidence/plan',provenance={'source':'all four-token query/source successor association probes; no outcome selection'})
    write(a.out/'plan.json',x)
    if not x['certified']:raise ValueError('invalid plan')
    queries={r['id']:r for r in read(QUERIES/'queries.json')};probes=[]
    for pp,qq,tt in x['relations']['probe']:
        q=queries[int(qq)]['ctx'];probes.append({'id':int(pp),'query':int(qq),'target':int(tt),'ctx':q+[int(tt)]+q})
    write(a.out/'probes.json',probes);write(a.out/'coefficients.json',read(PARTIAL/'kernel.json')['relations']['coefficient'])
    write(a.out/'protocol.json',{'scope':'post-hoc training source-conditioned memory-kernel diagnostic; not final confirmation','probes':len(probes),
        'probe':'last4 query tokens, source successor, same last4 query tokens; observe fixed final-MLP products603/1765',
        'mixtures':['equal source occurrence weights','maximum matched suffix length','matched suffix length divided by source distance'],
        'predictor':'normalize mixture weights; sum mixture product times fixed two-unit token coefficient; same strict long-source guard24/12/margin4; no tuning',
        'priors':{str((p/'closed.json').relative_to(ROOT)):sha256(p/'closed.json') for p in (QUERIES,PARTIAL)},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen',len(probes),'association forwards, three fixed mixtures',flush=True)
def checked(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('input changed')
    for n,h in (p['sources']|p['priors']).items():
        if sha256(ROOT/n)!=h:raise ValueError('source/prior changed')
def collect(a):
    checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_mlp_reference import MlpReference
    torch.set_num_threads(2);model=MlpReference(MODEL,vocab_chunk=1024);units=[tuple(map(int,r)) for r in read(a.out/'plan.json')['relations']['unit']]
    for r in read(a.out/'probes.json'):
        path=a.out/f"probe-{r['id']}.pt"
        if path.exists():continue
        data=model.probe(r['ctx'],units,());tmp=path.with_suffix('.pending');torch.save(data,tmp);tmp.replace(path)
        print('completed probe',r['id'],flush=True)
def evaluate(a):
    checked(a)
    import torch
    facts=read(a.out/'facts.json');facts['ref']=''.join(f'{i}\t{t}\n' for i,t in read(a.out/'references.json').items());facts['memory']='';facts['coefficient']=''.join('\t'.join(r)+'\n' for r in read(a.out/'coefficients.json'))
    for r in read(a.out/'probes.json'):
        p=r['id'];data=torch.load(a.out/f'probe-{p}.pt',weights_only=True);archive=a.out/'reference-evidence'/f'reference-{p}.tar.gz'
        if archive.exists():
            h=sha256(archive);e={'archive':archive.name,'sha256':h,'query':replay_archive(archive,h,row_id=p)}
        else:e=extract(data['logits'],archive,p)
        after=replay_archive(archive,e['sha256'],row_id=p)
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('reference replay differs')
        write(a.out/f'reference-{p}.json',e)
        facts['memory']+=''.join(f"{r['query']}\t{r['target']}\t{o['neuron']}\t{o['product']:.17g}\n" for o in data['observations'])
    m=ROOT/'dl/qwen35_memory_kernel_summary.dl'
    x=check(m,a.out/'input-core.dl',facts,{},evidence_dir=a.out/'certificate-evidence/summary',provenance={'source':'fixed source-conditioned memory mixture, known training only; no admission/equivalence'})
    write(a.out/'summary.json',x)
    for before in (read(a.out/'plan.json'),x):
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    write(a.out/'replay-audit.json',{'reference_witnesses':len(read(a.out/'probes.json')),'queries':2,'unchanged':True,'summary_certified':x['certified']})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('valid',x['certified'],'counts',x['relations']['counts'],flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','collect','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();{'inputs':inputs,'collect':collect,'evaluate':evaluate}[a.phase](a)
if __name__=='__main__':main()
