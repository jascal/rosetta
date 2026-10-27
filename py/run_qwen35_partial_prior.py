"""Test an isolated-query two-causal-unit readout kernel on known training data."""
import argparse
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_query_prior import ROOT,OUT as QUERIES,read,write
from run_qwen35_gate_readout import OUT as READOUT
from run_qwen35_copy_inventory import OUT as INVENTORY,MODEL,gate
OUT=ROOT/'reference/benchmarks/qwen35_partial_prior'
SOURCES=('py/run_qwen35_partial_prior.py','dl/qwen35_partial_readout_prior.dl','dl/qwen35_gate_readout.dl','dl/qwen35_query_prior_summary.dl','dl/qwen35_query_prior_counts.dl','py/qwen35_stream_reference.py','py/certificate.py')
def inputs(a):
    import torch
    for root in (QUERIES,READOUT):
        for n,h in read(root/'closed.json')['files'].items():
            if sha256(root/n)!=h:raise ValueError('prior changed')
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ('input-core.dl','facts.json','references.json','keys.json'):(a.out/name).write_bytes((QUERIES/name).read_bytes())
    facts=Path(read(READOUT/'readout.json')['evidence'])/'facts'
    for name in ('norm','down','unit','shape'):(a.out/f'{name}.facts').write_bytes((facts/f'{name}.facts').read_bytes())
    products=''
    for q in read(QUERIES/'queries.json'):
        data=torch.load(QUERIES/f"query-{q['id']}.pt",weights_only=True)
        products+=''.join(f"{q['id']}\t{r['neuron']}\t{r['product']:.17g}\n" for r in data['observations'])
    (a.out/'product.facts').write_text(products)
    (a.out/'key.facts').write_text(''.join('\t'.join(row)+'\n' for row in read(QUERIES/'keys.json')['relations']['key']))
    write(a.out/'protocol.json',{'scope':'post-hoc training-only mechanism-derived predictor; not admission/final confirmation','new_forwards':0,
        'kernel':'sum over fixed causal neurons603/1765 of isolated-query product times final pairwise readout coefficient; same last1/last4/full-query histories; no weights/threshold tuning',
        'guard':'unchanged diagnostic unique longest source >=24, competitor>=12, margin>=4; strict kernel preference agrees with it',
        'priors':{str((p/'closed.json').relative_to(ROOT)):sha256(p/'closed.json') for p in (QUERIES,READOUT)},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen two-unit predictor diagnostic; zero new model forwards',flush=True)
def checked(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('input changed')
    for n,h in (p['sources']|p['priors']).items():
        if sha256(ROOT/n)!=h:raise ValueError('source/prior changed')
def query(a,name,module,circuit,facts):
    x=check(ROOT/module,circuit,facts,{},evidence_dir=a.out/f'certificate-evidence/{name}',provenance={'source':'two-unit mechanism-derived kernel, known training only; no equivalence or admission'})
    write(a.out/f'{name}.json',x)
    if not x['certified']:raise ValueError('diagnostic query failed')
    return x
def run(a):
    checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    from qwen35_stream_reference import WeightReader
    reader=WeightReader(MODEL);head='lm_head.weight' if 'lm_head.weight' in reader.files else reader.prefix+'embed_tokens.weight'
    # Project required token IDs only; the key relation was selected by Datalog.
    tokens=sorted({int(t) for q,t in read(a.out/'keys.json')['relations']['key']})
    facts={n:(a.out/f'{n}.facts').read_text() for n in ('norm','down','unit','shape','key','product')}
    facts.update({'token':''.join(f'{t}\n' for t in tokens),'head':'','effect':'','observed':''})
    for t in tokens:facts['head']+=''.join(f'{t}\t{h}\t{v:.17g}\n' for h,v in enumerate(reader.rows(head,t,t+1).float()[0].tolist()))
    kernel=query(a,'kernel','dl/qwen35_partial_readout_prior.dl',ROOT/'dl/qwen35_gate_readout.dl',facts)
    sfacts=read(a.out/'facts.json');sfacts['ref']=''.join(f'{i}\t{t}\n' for i,t in read(a.out/'references.json').items());sfacts['prior']=''.join('\t'.join(row)+'\n' for row in kernel['relations']['prior'])
    summary=query(a,'summary','dl/qwen35_query_prior_summary.dl',a.out/'input-core.dl',sfacts)
    r=summary['relations'];cfacts={'query':sfacts['query'],'disagreement':''.join(f'{q}\t{i}\n' for q,i,c in r['winner_agreement'] if c=='prior_differs_model')}
    for name in ('preference','guarded','error'):cfacts[name]=''.join('\t'.join(row)+'\n' for row in r[name])
    counts=query(a,'counts','dl/qwen35_query_prior_counts.dl',ROOT/'dl/qwen35_query_prior_counts.dl',cfacts)
    for before in (kernel,summary,counts):
        after=replay(Path(before['evidence']))
        if before['relations']!=after['relations'] or before['certified']!=after['certified']:raise ValueError('replay differs')
    write(a.out/'replay-audit.json',{'queries':3,'unchanged':True,'new_forwards':0})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('prior',counts['relations']['priorcounts'],'guarded',counts['relations']['counts'],flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();{'inputs':inputs,'run':run}[a.phase](a)
if __name__=='__main__':main()
