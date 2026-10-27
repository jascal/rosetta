"""Freeze and test both singleton gate candidates across all directed contrasts."""
import argparse,hashlib
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_gate_bisection import OUT as PRIOR
from run_qwen35_sparse_gate import ROOT,FACTOR,read,write,collect,checked,evaluate as evaluate_effects
OUT=ROOT/'reference/benchmarks/qwen35_gate_cross'
SOURCES=('py/run_qwen35_gate_cross.py','dl/qwen35_gate_cross_payload.dl','dl/qwen35_gate_cross_reproduction.dl','py/run_qwen35_sparse_gate.py','py/run_qwen35_mlp_factor.py','dl/qwen35_mlp_factor_summary.dl','py/qwen35_mlp_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','py/qwen35_reference_evidence.py','py/certificate.py','dl/qwen35_reference_argmax.dl')
def inputs(a):
    for root in (PRIOR,FACTOR):
        for n,h in read(root/'closed.json')['files'].items():
            if sha256(root/n)!=h:raise ValueError('prior evidence changed')
    report=read(ROOT/'reference/benchmarks/qwen35_gate_bisection_replay.json')
    if not report['unchanged'] or report['closed_sha256']!=sha256(PRIOR/'closed.json'):raise ValueError('prior replay invalid')
    a.out.mkdir(parents=True,exist_ok=False);contexts=read(PRIOR/'contexts.json');write(a.out/'contexts.json',contexts)
    leaves=read(PRIOR/'outcome.json')['relations']['leaf'];factor=read(FACTOR/'payload.json')['relations'];hybrid='';donors=''
    for arm,i,d,mode in factor['run']:
        if mode=='donor_gate':
            donors+=f'{i}\t{d}\n';hybrid+=''.join(f'{i}\t{n}\t{v}\n' for aa,l,p,n,v in factor['patch'] if aa==arm)
    m=ROOT/'dl/qwen35_gate_cross_payload.dl'
    x=check(m,m,{'context':''.join(f"{r['id']}\t{len(r['ctx'])-1}\n" for r in contexts),'donor':donors,
        'candidate':''.join(f'{n}\n' for c,node,n in leaves),'hybrid':hybrid},{},evidence_dir=a.out/'certificate-evidence/payload',
        provenance={'source':'both singleton candidates and joint, all six original directions, no further outcome selection'})
    write(a.out/'payload.json',x);write(a.out/'discovery-leaves.json',leaves)
    if not x['certified'] or replay(Path(x['evidence']))['relations']!=x['relations']:raise ValueError('payload failed')
    write(a.out/'protocol.json',{'scope':'development cross-context candidate test, not fresh final confirmation','forwards':len(x['relations']['run']),
        'arms':'neuron603, neuron1765, both; all six directed contrasts; discovery singletons repeated as exact full-score hash controls',
        'interpretation':'success across cases suggests shared gate causality, not a token rule; failures retained; no neuron retuning',
        'prior':{'path':str(PRIOR.relative_to(ROOT)/'closed.json'),'sha256':sha256(PRIOR/'closed.json')},
        'factor_prior_sha256':sha256(FACTOR/'closed.json'),
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen',len(x['relations']['run']),'forwards',flush=True)
def evaluate(a):
    checked(a)
    import torch
    facts={'expected':'','actual':''}
    for case,node,neuron in read(a.out/'discovery-leaves.json'):
        old=torch.load(PRIOR/f'arm-{100*int(case)+int(node)}.pt',weights_only=True)['logits']
        arm=10000*int(case)+int(neuron);new=torch.load(a.out/f'arm-{arm}.pt',weights_only=True)['logits']
        facts['expected']+=f'{arm}\t{hashlib.sha256(old.numpy().tobytes()).hexdigest()}\n'
        facts['actual']+=f'{arm}\t{hashlib.sha256(new.numpy().tobytes()).hexdigest()}\n'
    m=ROOT/'dl/qwen35_gate_cross_reproduction.dl'
    x=check(m,m,facts,{},evidence_dir=a.out/'certificate-evidence/reproduction',provenance={'source':'repeat discovery singleton substitutions, exact whole-score hashes'})
    write(a.out/'reproduction.json',x)
    if not x['certified'] or replay(Path(x['evidence']))['relations']!=x['relations']:raise ValueError('discovery reproduction failed')
    evaluate_effects(a)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if a.phase!='inputs' and sha256(FACTOR/'closed.json')!=read(a.out/'protocol.json')['factor_prior_sha256']:raise ValueError('factor prior changed')
    {'inputs':inputs,'run':collect,'evaluate':evaluate}[a.phase](a)
if __name__=='__main__':main()
