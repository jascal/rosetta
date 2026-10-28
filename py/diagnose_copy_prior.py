"""Frozen one-token-prior diagnostic; no rule selection or final-test collection."""
import argparse
import json
from pathlib import Path
import time

from certificate import check,sha256,replay
from run_copy_mechanism import ROOT,server,post,read,write
from run_copy_prefix import journal

OUT=ROOT/'reference/benchmarks/qwen25_copy_prior_pilot'
SOURCE=ROOT/'reference/benchmarks/qwen25_copy_long_consensus'
SOURCES=('py/diagnose_copy_prior.py','py/run_copy_mechanism.py','py/run_copy_prefix.py','py/certificate.py',
    'dl/copy_prior_plan.dl','dl/copy_prior_report.dl','docs/copy-prior-pilot-plan.md')


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[r for r in read(SOURCE/'dataset.json') if not r['negative']]
    write(a.out/'dataset.json',rows)
    facts={'tok':''.join(f"{r['id']}\t{j}\t{t}\n" for r in rows for j,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['expected']}\n" for r in rows)}
    write(a.out/'facts.json',facts)
    (a.out/'library.dl').write_text((SOURCE/'library.dl').read_text())
    x=check(ROOT/'dl/copy_prior_plan.dl',a.out/'library.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/plan',provenance={'source':'historical development prior plan, no new final references'})
    write(a.out/'plan.json',x)
    if not x['certified']:raise ValueError(x.get('error','invalid plan'))
    write(a.out/'frozen.json',{'scope':'one-token-prior diagnostic only',
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'historical':{str(p.relative_to(ROOT)):sha256(p) for p in (SOURCE/'dataset.json',SOURCE/'observations.jsonl',SOURCE/'library.dl')},
        'pinned_model':read(ROOT/'reference/benchmarks/qwen25_native_trace_b/model.json')})
    print('queries:',len(x['relations']['query']),'readouts:',len(x['relations']['readout']))


def gate(a):
    frozen=read(a.out/'frozen.json')
    for n,h in frozen['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'frozen input changed: {n}')
    for n,h in (frozen['sources']|frozen['historical']).items():
        if sha256(ROOT/n)!=h:raise ValueError(f'frozen source changed: {n}')
    if a.binary and sha256(a.binary)!=frozen['pinned_model']['binary_sha256']:raise ValueError('different executable')
    if a.bundle and sha256(str(a.bundle)+'.fieldrun.bin')!=frozen['pinned_model']['bundle_sha256']['.fieldrun.bin']:
        raise ValueError('different weights')
    return read(a.out/'plan.json')['relations']


def collect(a):
    plan=gate(a);path=a.out/'observations.jsonl';cached=journal(path)
    readouts={int(q):[] for q, in plan['query']}
    for q,t in plan['readout']:readouts[int(q)].append(int(t))
    if set(cached)-set(readouts):raise ValueError('unplanned prior query')
    with server(a,'prior') as port,path.open('a') as stream:
        for q in sorted(readouts):
            if q in cached:continue
            start=time.monotonic();x=post(port,[q],{'readouts':sorted(readouts[q])})
            stream.write(json.dumps({'id':q,'response':x,'seconds':time.monotonic()-start})+'\n');stream.flush()
            print('prior',q,flush=True)


def report(a):
    plan=gate(a);facts=read(a.out/'facts.json')
    rows=read(a.out/'dataset.json');old=journal(SOURCE/'observations.jsonl')
    facts['actual']=''.join(f"{r['id']}\t{old[r['id']]['response']['next']}\n" for r in rows)
    facts['requested']=''.join('\t'.join(r)+'\n' for r in plan['readout'])
    facts['prior']=''.join(f"{q}\t{t}\t{s:.17g}\n" for q,r in journal(a.out/'observations.jsonl').items() for t,s in r['response']['logits'])
    x=check(ROOT/'dl/copy_prior_report.dl',a.out/'library.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/report',provenance={'source':'descriptive one-token prior versus historical full-context references'})
    write(a.out/'report.json',x)
    if not x['certified']:raise ValueError(x.get('error','invalid data'))
    print('counts:',x['relations']['counts'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('inputs','collect','report'))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--binary',type=Path);p.add_argument('--bundle',type=Path)
    p.add_argument('--port',type=int,default=8398)
    a=p.parse_args();a.out=a.out.resolve()
    if a.phase=='collect' and not(a.binary and a.bundle):p.error('collection needs --binary and --bundle')
    globals()[a.phase](a)


if __name__=='__main__':main()
