"""Build-time lexical feature extraction. Datalog compiles scores to discrete bands."""
import argparse
import gzip
import json
from pathlib import Path
import random
import shutil
import time

from certificate import check,sha256
from run_copy_mechanism import ROOT,server,post,read,write
from run_copy_long_consensus import EXCLUDE as PREVIOUS
from run_copy_prefix import _tokens

OUT=ROOT/'reference/benchmarks/qwen25_copy_prior_lexicon'
EXCLUDE=PREVIOUS+('reference/benchmarks/qwen25_copy_long_consensus/dataset.json',
    'reference/benchmarks/qwen25_copy_prior_pilot/dataset.json')
SOURCES=('py/build_copy_prior_lexicon.py','py/run_copy_long_consensus.py','py/run_copy_longest.py',
    'py/run_copy_prefix.py','py/run_native_trace2.py','py/run_copy_mechanism.py','py/certificate.py',
    'dl/copy_prior_bands.dl','docs/copy-prior-guard-plan.md')


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    excluded=set()
    for n in EXCLUDE:excluded.update(_tokens(n))
    sizes=(('query',512),('target',1024),('decoy',1024),('background',8192))
    tokens=random.Random(910).sample([t for t in range(200,40000) if t not in excluded],sum(n for _,n in sizes))
    pools={};start=0
    for name,n in sizes:pools[name]=tokens[start:start+n];start+=n
    write(a.out/'pools.json',pools);write(a.out/'excluded-tokens.json',sorted(excluded))
    write(a.out/'protocol.json',{'scope':'context-free finite lexical extraction before any new full contexts',
        'seed':910,'sizes':dict(sizes),'band_width':1,'tail_band':63,
        'files':{n:sha256(a.out/n) for n in ('pools.json','excluded-tokens.json')},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},'exclusions':{n:sha256(ROOT/n) for n in EXCLUDE},
        'pinned_model':read(ROOT/'reference/benchmarks/qwen25_native_trace_b/model.json')})
    print('fixed role pools:',dict(sizes))


def gate(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'changed pool: {n}')
    for n,h in (p['sources']|p['exclusions']).items():
        if sha256(ROOT/n)!=h:raise ValueError(f'changed source: {n}')
    if a.binary and sha256(a.binary)!=p['pinned_model']['binary_sha256']:raise ValueError('different executable')
    if a.bundle and sha256(str(a.bundle)+'.fieldrun.bin')!=p['pinned_model']['bundle_sha256']['.fieldrun.bin']:
        raise ValueError('different weights')
    return read(a.out/'pools.json')


def observations(path):
    result={}
    if path.exists():
        with gzip.open(path,'rt') as stream:
            for line in stream:
                r=json.loads(line)
                if r['id'] in result:raise ValueError('duplicate lexical query')
                result[r['id']]=r
    return result


def collect(a):
    pools=gate(a);path=a.out/'observations.jsonl.gz';cached=observations(path)
    if (a.out/'feature.json').exists():raise ValueError('feature module already frozen')
    if set(cached)-set(pools['query']):raise ValueError('unplanned lexical query')
    targets=sorted(pools['target']+pools['decoy'])
    with server(a,'lexicon') as port,gzip.open(path,'at') as stream:
        for j,q in enumerate(pools['query']):
            if q in cached:continue
            start=time.monotonic();x=post(port,[q],{'readouts':targets})
            stream.write(json.dumps({'id':q,'response':x,'seconds':time.monotonic()-start})+'\n');stream.flush()
            if (j+1)%32==0:print('lexical queries',j+1,'/',len(pools['query']),flush=True)


def compile_feature(a):
    pools=gate(a)
    if (a.out/'feature.json').exists():raise ValueError('feature module already frozen')
    observed=observations(a.out/'observations.jsonl.gz')
    targets=pools['target']+pools['decoy']
    facts={'query':''.join(f'{q}\n' for q in pools['query']),
        'target':''.join(f'{t}\n' for t in targets),
        'maximum':''.join(f"{q}\t{r['response']['next_logit']:.17g}\n" for q,r in observed.items()),
        'score':''.join(f"{q}\t{t}\t{s:.17g}\n" for q,r in observed.items() for t,s in r['response']['logits'])}
    source=ROOT/'dl/copy_prior_bands.dl'
    x=check(source,source,facts,{'nband':int},evidence_dir=a.out/'certificate-evidence/bands',
        provenance={'source':'complete one-token observations to categorical finite lexicon; not model-equivalence'})
    small={k:v for k,v in x.items() if k!='relations'}
    write(a.out/'band-check.json',small)
    if not x['certified']:raise ValueError(x.get('error','incomplete or invalid lexical observations'))
    shutil.copyfile(Path(x['evidence'])/'outputs/prior_band.csv',a.out/'prior-bands.facts')
    files=[p for p in a.out.rglob('*') if p.is_file()]
    write(a.out/'feature.json',{'scope':'frozen finite categorical prior module; no new full contexts yet',
        'rows':x['nband'],'facts_bytes':(a.out/'prior-bands.facts').stat().st_size,
        'files':{str(p.relative_to(a.out)):sha256(p) for p in files},
        'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen prior facts:',x['nband'],'bytes:',(a.out/'prior-bands.facts').stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('inputs','collect','compile'))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--binary',type=Path);p.add_argument('--bundle',type=Path)
    p.add_argument('--port',type=int,default=8399)
    a=p.parse_args();a.out=a.out.resolve()
    if a.phase=='collect' and not(a.binary and a.bundle):p.error('collection needs --binary and --bundle')
    {'inputs':inputs,'collect':collect,'compile':compile_feature}[a.phase](a)


if __name__=='__main__':main()
