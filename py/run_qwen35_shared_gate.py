"""Frozen individual/joint shared-gate tests; no new outcome-based selection."""
import argparse
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_sparse_gate import ROOT,OUT as PRIOR,FACTOR,read,write,collect,checked,evaluate
OUT=ROOT/'reference/benchmarks/qwen35_shared_gate'
OVERLAP=ROOT/'reference/benchmarks/qwen35_sparse_gate_overlap'
SOURCES=('py/run_qwen35_shared_gate.py','dl/qwen35_shared_gate_payload.dl','py/run_qwen35_sparse_gate.py','py/run_qwen35_mlp_factor.py','dl/qwen35_mlp_factor_summary.dl','py/qwen35_mlp_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','py/qwen35_reference_evidence.py','py/certificate.py','dl/qwen35_reference_argmax.dl')
def inputs(a):
    for root in (PRIOR,FACTOR,OVERLAP):
        for n,h in read(root/'closed.json')['files'].items():
            if sha256(root/n)!=h:raise ValueError('prior evidence changed')
    report=read(ROOT/'reference/benchmarks/qwen35_sparse_gate_replay.json')
    if not report['unchanged'] or report['closed_sha256']!=sha256(PRIOR/'closed.json'):raise ValueError('prior replay invalid')
    a.out.mkdir(parents=True,exist_ok=False);contexts=read(PRIOR/'contexts.json');write(a.out/'contexts.json',contexts)
    sparse=read(PRIOR/'payload.json');factor=read(FACTOR/'payload.json')['relations'];shared=read(OVERLAP/'overlap.json')['relations']['shared']
    # Serialization only: Datalog retains all six directed contrasts and selects sites.
    hybrid='';donors=''
    for arm,target,donor,mode in factor['run']:
        if mode=='donor_gate':
            donors+=f'{target}\t{donor}\n'
            hybrid+=''.join(f'{target}\t{n}\t{v}\n' for aa,l,p,n,v in factor['patch'] if aa==arm)
    facts={'context':''.join(f"{r['id']}\t{len(r['ctx'])-1}\n" for r in contexts),'donor':donors,'hybrid':hybrid,
        'top':''.join(f'{i}\t{n}\n' for arm,i,n in sparse['relations']['selected'] if int(arm)==10000*int(i)+320),
        'shared':''.join(f'{n}\n' for n, in shared),
        'random_order':(Path(sparse['evidence'])/'facts/random_order.facts').read_text()}
    m=ROOT/'dl/qwen35_shared_gate_payload.dl'
    x=check(m,m,facts,{},evidence_dir=a.out/'certificate-evidence/payload',provenance={'source':'previous common gate candidates162/381, individual/joint and remaining-top32/random controls, all directions'})
    write(a.out/'payload.json',x)
    if not x['certified'] or replay(Path(x['evidence']))['relations']!=x['relations']:raise ValueError('invalid/replay changed payload')
    write(a.out/'protocol.json',{'scope':'development shared gate causality; not standalone conversion','forwards':len(x['relations']['run']),
        'arms':'neuron162, neuron381, joint, remaining top32 excluding both, and first two nonshared neurons in seed2200 random order; all six directed contrasts',
        'interpretation':'individual/joint sufficiency only on selected development cases; rest tests alternative support; no final selection or outcome-based retuning',
        'prior':{'path':str(PRIOR.relative_to(ROOT)/'closed.json'),'sha256':sha256(PRIOR/'closed.json')},
        'additional_priors':{str((p/'closed.json').relative_to(ROOT)):sha256(p/'closed.json') for p in (FACTOR,OVERLAP)},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen',len(x['relations']['run']),'forwards',flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if a.phase!='inputs':
        protocol=read(a.out/'protocol.json')
        for n,h in protocol['additional_priors'].items():
            if sha256(ROOT/n)!=h:raise ValueError('additional prior changed')
    {'inputs':inputs,'run':collect,'evaluate':evaluate}[a.phase](a)
if __name__=='__main__':main()
