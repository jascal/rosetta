"""Build-time weight I/O and Datalog pairwise readout decomposition, no forwards."""
import argparse
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_copy_inventory import ROOT,MODEL,OUT as INVENTORY,gate,read,write
from run_qwen35_gate_cross import OUT as PRIOR
from run_qwen35_gate_bisection import OUT as SEARCH
OUT=ROOT/'reference/benchmarks/qwen35_gate_readout'
SOURCES=('py/run_qwen35_gate_readout.py','dl/qwen35_gate_readout_keys.dl','dl/qwen35_gate_readout.dl','dl/copy_longest_runtime.dl','py/qwen35_stream_reference.py','py/certificate.py')
def inputs(a):
    for root in (PRIOR,SEARCH):
        for n,h in read(root/'closed.json')['files'].items():
            if sha256(root/n)!=h:raise ValueError('prior changed')
    report=read(ROOT/'reference/benchmarks/qwen35_gate_cross_replay.json')
    if not report['unchanged'] or report['closed_sha256']!=sha256(PRIOR/'closed.json'):raise ValueError('cross-context replay invalid')
    a.out.mkdir(parents=True,exist_ok=False);contexts=read(PRIOR/'contexts.json');write(a.out/'contexts.json',contexts)
    core=a.out/'input-core.dl';core.write_bytes((INVENTORY/'input-core.dl').read_bytes())
    m=ROOT/'dl/qwen35_gate_readout_keys.dl'
    x=check(m,core,{'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in contexts for p,t in enumerate(r['ctx'])),
        'answer':''.join(f"{r['id']}\t{r['expected']}\n" for r in contexts),
        'candidate':''.join(f'{n}\n' for c,node,n in read(SEARCH/'outcome.json')['relations']['leaf'])},{},evidence_dir=a.out/'certificate-evidence/keys',provenance={'source':'all source successors and prior observed answers; no token preference selection'})
    write(a.out/'keys.json',x)
    if not x['certified']:raise ValueError('key query failed')
    write(a.out/'protocol.json',{'scope':'build-time pairwise readout analysis of retained interventions, not token-only artifact','new_forwards':0,
        'formula':'coefficient(n,t)=sum_h down(h,n)*(1+norm_weight(h))*head(t,h); unnormalized margin change=(after-before)*(coefficient(wanted)-coefficient(old))',
        'limits':'RMS denominator is positive and common across vocabulary, so margin sign per state is unaffected; coefficient calculation uses real/double arithmetic on stored weights, not a bit-exact full-vocabulary FP32 reproduction',
        'priors':{str((p/'closed.json').relative_to(ROOT)):sha256(p/'closed.json') for p in (PRIOR,SEARCH)},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen readout analysis',flush=True)
def checked(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('input changed')
    for n,h in (p['sources']|p['priors']).items():
        if sha256(ROOT/n)!=h:raise ValueError('source/prior changed')
def run(a):
    checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_stream_reference import WeightReader
    reader=WeightReader(MODEL);keys=read(a.out/'keys.json')['relations'];neurons=[int(n) for n, in keys['unit']];tokens=[int(t) for t, in keys['token']]
    norm=reader.tensor(reader.prefix+'norm.weight').float();down=reader.tensor(reader.prefix+'layers.31.mlp.down_proj.weight').float()
    headkey='lm_head.weight' if 'lm_head.weight' in reader.files else reader.prefix+'embed_tokens.weight'
    facts={'shape':f'{len(norm)}\n','unit':''.join(f'{n}\n' for n in neurons),'token':''.join(f'{t}\n' for t in tokens),
        'norm':''.join(f'{h}\t{v:.17g}\n' for h,v in enumerate(norm.tolist())),
        'down':''.join(f'{n}\t{h}\t{v:.17g}\n' for n in neurons for h,v in enumerate(down[:,n].tolist())),
        'head':'','effect':'','observed':''}
    for t in tokens:
        values=reader.rows(headkey,t,t+1).float()[0].tolist();facts['head']+=''.join(f'{t}\t{h}\t{v:.17g}\n' for h,v in enumerate(values))
    contexts={str(r['id']):r for r in read(a.out/'contexts.json')}
    for arm,target,donor,mode in read(PRIOR/'payload.json')['relations']['run']:
        if mode=='joint':continue
        result=torch.load(PRIOR/f'arm-{arm}.pt',weights_only=True);applied=result['applied']
        if len(applied)!=1:raise ValueError('singleton metadata differs')
        r=applied[0];old=contexts[target]['expected'];wanted=contexts[donor]['expected'];nexttoken=read(PRIOR/f'reference-{arm}.json')['query']['relations']['ref'][0][0]
        facts['effect']+=f"{target}\t{r['neuron']}\t{old}\t{wanted}\t{r['before']:.17g}\t{r['after']:.17g}\n"
        facts['observed']+=f"{target}\t{r['neuron']}\t{nexttoken}\n"
    m=ROOT/'dl/qwen35_gate_readout.dl'
    x=check(m,m,facts,{},evidence_dir=a.out/'certificate-evidence/readout',provenance={'source':'Datalog real-valued pairwise output contribution; weight facts build-time only; no extraction certificate'})
    write(a.out/'readout.json',x)
    for before in (read(a.out/'keys.json'),x):
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('replay differs')
    write(a.out/'replay-audit.json',{'queries':2,'unchanged':True,'new_model_forwards':0,'readout_consistent':x['certified']})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('consistent',x['certified'],'directions',x['relations']['counts'],'sign conflicts',x['relations']['sign_conflict'],flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();{'inputs':inputs,'run':run}[a.phase](a)
if __name__=='__main__':main()
