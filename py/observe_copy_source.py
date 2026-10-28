"""Retain native explanations for a Datalog-fixed historical case sample."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

from certificate import check, replay, sha256
from run_copy_mechanism import server, read, write
from run_copy_prefix import ROOT, journal

OUT=ROOT/'reference/benchmarks/qwen25_copy_source_diagnosis'
SOURCE=ROOT/'reference/benchmarks/qwen25_copy_longest'
SOURCES=('py/observe_copy_source.py','py/run_copy_mechanism.py','py/run_copy_prefix.py',
         'py/certificate.py','dl/copy_source_diagnosis.dl','dl/copy_source_explanation.dl',
         'docs/copy-source-diagnosis-plan.md')


def freeze(a):
    if (a.out/'frozen.json').exists() or (a.out/'explanations.jsonl').exists():
        raise ValueError('already frozen or observed')
    classification=read(a.out/'classification.json')
    actual=replay(Path(classification['evidence']))
    if actual['relations']!=classification['relations'] or not actual['certified']:
        raise ValueError('classification replay failed')
    rows={r['id']:r for r in read(SOURCE/'dataset.json')}
    selected=[dict(rows[int(i)],diagnostic_class=c) for i,c in actual['relations']['selected_case']]
    write(a.out/'selected.json',selected)
    files=[p for p in a.out.rglob('*') if p.is_file()]
    write(a.out/'frozen.json',{'scope':'historical source diagnostic, not confirmation',
        'files':{str(p.relative_to(a.out)):sha256(p) for p in files},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'historical':{str(p.relative_to(ROOT)):sha256(p) for p in
            (SOURCE/'dataset.json',SOURCE/'observations.jsonl',SOURCE/'library.dl',SOURCE/'closed.json')},
        'pinned_model':read(ROOT/'reference/benchmarks/qwen25_native_trace_b/model.json')})
    print('frozen native explanation sample:',len(selected))


def gate(a):
    p=read(a.out/'frozen.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h: raise ValueError(f'changed frozen input: {n}')
    for n,h in (p['sources']|p['historical']).items():
        if sha256(ROOT/n)!=h: raise ValueError(f'changed frozen source: {n}')
    if a.binary and sha256(a.binary)!=p['pinned_model']['binary_sha256']: raise ValueError('different executable')
    if a.bundle and sha256(str(a.bundle)+'.fieldrun.bin')!=p['pinned_model']['bundle_sha256']['.fieldrun.bin']:
        raise ValueError('different weights')
    return read(a.out/'selected.json')


def collect(a):
    rows=gate(a)
    path=a.out/'explanations.jsonl'
    cached=journal(path)
    if set(cached)-{r['id'] for r in rows}: raise ValueError('unplanned explanation')
    with server(a,'explain') as port,path.open('a') as stream:
        for r in rows:
            if r['id'] in cached: continue
            req=urllib.request.Request(f'http://127.0.0.1:{port}/explain',
                json.dumps({'ids':r['ctx']}).encode(),{'Content-Type':'application/json'})
            start=time.monotonic()
            with urllib.request.urlopen(req,timeout=600) as response: x=json.load(response)
            if 'error' in x: raise ValueError(x)
            stream.write(json.dumps({'id':r['id'],'response':x,'seconds':time.monotonic()-start})+'\n')
            stream.flush()
            print('explained',r['id'],r['diagnostic_class'],flush=True)


def report(a):
    rows=gate(a); observations=journal(a.out/'explanations.jsonl'); refs=journal(SOURCE/'observations.jsonl')
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['diagnostic_class']}\t{r['expected']}\n" for r in rows),
        'ref':''.join(f"{r['id']}\t{refs[r['id']]['response']['next']}\n" for r in rows),
        'explained':'','head':'','promotes':''}
    for i,r in observations.items():
        x=r['response']; facts['explained']+=f"{i}\t{x['model_predicts']}\n"
        for rank,h in enumerate(x['head_circuits'],1):
            facts['head']+=(f"{i}\t{rank}\t{h['layer']}\t{h['head']}\t{h['attends_to']}\t{h['attends_tok']}\t{h['mass']:.17g}\t{h['dla']:.17g}\n")
            facts['promotes']+=''.join(f"{i}\t{rank}\t{k}\t{t}\n" for k,t in enumerate(h['promotes'],1))
    x=check(ROOT/'dl/copy_source_explanation.dl',SOURCE/'library.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/explanation',
        provenance={'source':'DLA-prefiltered native explanations; descriptive, no causal mechanism claim'})
    write(a.out/'report.json',x)
    if not x['certified']: raise ValueError(x.get('error','explanation consistency failed'))
    print('peak categories:',x['relations']['peak_class'])
    print('promotes actual:',x['relations']['promotes_actual'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('freeze','collect','report'))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--binary',type=Path);p.add_argument('--bundle',type=Path)
    p.add_argument('--port',type=int,default=8396)
    a=p.parse_args();a.out=a.out.resolve()
    if a.phase=='collect' and not(a.binary and a.bundle): p.error('collection needs --binary and --bundle')
    globals()[a.phase](a)


if __name__=='__main__': main()
