"""Fixed 24-case approximate-alignment discovery pilot. Datalog computes every verdict."""
import argparse
import json
import random
import time
from pathlib import Path

from certificate import check, sha256
from run_copy_mechanism import server, post, read, write
from run_copy_prefix import ROOT, BASELINE, EXCLUSIONS, _tokens, journal

OUT = ROOT / 'reference/benchmarks/qwen25_copy_corruption_pilot'
SOURCE = ROOT / 'reference/benchmarks/qwen25_native_trace_c'
OPTIONS = ((1,6,2), (2,6,3), (3,12,2), (4,12,3))
SOURCES = ('py/run_copy_corruption.py','py/run_copy_mechanism.py','py/run_copy_prefix.py',
           'py/run_native_trace2.py','py/certificate.py','dl/copy_approx_runtime.dl',
           'dl/copy_corruption_report.dl','docs/copy-corruption-pilot-plan.md')


def inputs(a):
    a.out.mkdir(parents=True, exist_ok=False)
    names = list(EXCLUSIONS) + ['reference/benchmarks/qwen25_copy_prefix/dataset.json']
    excluded = set()
    for name in names:
        excluded.update(_tokens(name))
    rng = random.Random(501)
    old = read(SOURCE / 'dataset.json')
    refs = read(SOURCE / 'references.json')
    selected = [r for r in old if not r['negative'] and r['length'] in (12,24)
                and r['layout']=='plain' and r['exposures']==2]
    rows, replacements = [], {}
    for row in selected:
        group = row['group']
        if group not in replacements:
            t = rng.choice([t for t in range(200,40000) if t not in excluded])
            replacements[group] = t
            excluded.add(t)
        for offset in (0,1,3):
            ctx = row['ctx'][:]
            p = len(ctx)-1-offset
            ctx[p] = replacements[group]
            rows.append(dict(row,id=len(rows),source_id=row['id'],offset=offset,ctx=ctx,
                             original=row['ctx'],corrupt=p,oldref=refs[str(row['id'])]))
    write(a.out / 'dataset.json',rows)
    write(a.out / 'replacements.json',replacements)
    library = BASELINE.read_text().replace('cdecide','bdecide')
    library += '\n.decl ap_option(c:number,k:number,min_sources:number)\n'
    library += ''.join(f'ap_option({c},{k},{m}).\n' for c,k,m in OPTIONS)
    library += (ROOT/'dl/copy_approx_runtime.dl').read_text()
    (a.out/'library.dl').write_text(library)
    pinned = read(ROOT/'reference/benchmarks/qwen25_native_trace_b/model.json')
    write(a.out/'protocol.json',{'scope':'historical-background discovery pilot; not final confirmation',
        'seed':501,'rows':len(rows),'options':OPTIONS,
        'files':{n:sha256(a.out/n) for n in ('dataset.json','replacements.json','library.dl')},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'historical':{n:sha256(ROOT/n) for n in names},
        'baseline_sha256':sha256(BASELINE),'pinned_model':pinned})
    print(f'{len(rows)} pilot contexts fixed')


def gate(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h: raise ValueError(f'changed pilot input: {n}')
    for n,h in (p['sources']|p['historical']).items():
        if sha256(ROOT/n)!=h: raise ValueError(f'changed pilot source: {n}')
    if sha256(BASELINE)!=p['baseline_sha256']: raise ValueError('baseline changed')
    if a.binary and sha256(a.binary)!=p['pinned_model']['binary_sha256']: raise ValueError('different executable')
    if a.bundle and sha256(str(a.bundle)+'.fieldrun.bin')!=p['pinned_model']['bundle_sha256']['.fieldrun.bin']:
        raise ValueError('different weights')
    return read(a.out/'dataset.json')


def collect(a):
    rows=gate(a)
    path=a.out/'observations.jsonl'
    cached=journal(path)
    if set(cached)-{r['id'] for r in rows}: raise ValueError('unplanned observation')
    with server(a,'pilot') as port, path.open('a') as stream:
        for row in rows:
            if row['id'] in cached: continue
            start=time.monotonic()
            x=post(port,row['ctx'],{'readouts':[row['expected']]})
            stream.write(json.dumps({'id':row['id'],'response':x,'seconds':time.monotonic()-start})+'\n')
            stream.flush()
            print('pilot',row['id']+1,'/',len(rows),flush=True)


def report(a):
    rows=gate(a)
    refs=journal(a.out/'observations.jsonl')
    mapping={(r['source_id'],r['offset']):r['id'] for r in rows}
    pairs=[(mapping[(r['base'],r['offset'])],r['id'],r['positions']) for r in rows if r['kind']=='intervention']
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'orig':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['original'])),
        'sample':''.join(f"{r['id']}\t{r['group']}\t{r['length']}\t{r['offset']}\t{r['expected']}\t{r['oldref']}\n" for r in rows),
        'corrupt':''.join(f"{r['id']}\t{r['corrupt']}\n" for r in rows),
        'ref':''.join(f"{i}\t{r['response']['next']}\n" for i,r in refs.items()),
        'pair':''.join(f'{a}\t{b}\n' for a,b,_ in pairs),
        'source_edit':''.join(f'{a}\t{b}\t{p}\n' for a,b,positions in pairs for p in positions)}
    x=check(ROOT/'dl/copy_corruption_report.dl',a.out/'library.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/report',provenance={'source':'corrupted-query discovery pilot; not final model-equivalence evidence'})
    write(a.out/'report.json',x)
    if not x['certified']: raise ValueError(x.get('error','input integrity failed'))
    print('model',x['relations']['model_score'])
    print('hypotheses',x['relations']['score'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('inputs','collect','report'))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--binary',type=Path)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--port',type=int,default=8396)
    a=p.parse_args(); a.out=a.out.resolve()
    if a.phase=='collect' and not(a.binary and a.bundle): p.error('collection needs --binary and --bundle')
    globals()[a.phase](a)


if __name__=='__main__': main()
