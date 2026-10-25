"""Fixed historical-background source-selection pilot. Python stages I/O only."""
import argparse
import random
from pathlib import Path

from certificate import check, sha256
from run_copy_corruption import collect, gate
from run_copy_prefix import ROOT, BASELINE, EXCLUSIONS, _tokens, journal
from run_copy_mechanism import read, write

OUT = ROOT / 'reference/benchmarks/qwen25_copy_longest_pilot'
SOURCE = ROOT / 'reference/benchmarks/qwen25_native_trace_c'
OPTIONS = ((1,6,3),(2,12,3))
SOURCES = ('py/run_copy_longest_pilot.py','py/run_copy_corruption.py','py/run_copy_mechanism.py',
           'py/run_copy_prefix.py','py/run_native_trace2.py','py/certificate.py',
           'dl/copy_longest_runtime.dl','dl/copy_longest_report.dl','docs/copy-longest-source-pilot-plan.md')


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    names = list(EXCLUSIONS)+['reference/benchmarks/qwen25_copy_prefix/dataset.json',
                            'reference/benchmarks/qwen25_copy_corruption_pilot/dataset.json']
    excluded = set()
    for n in names:
        excluded.update(_tokens(n))
    rng = random.Random(502)
    old = read(SOURCE/'dataset.json')
    selected = [r for r in old if not r['negative'] and r['length'] in (12,24)
                and r['layout']=='plain' and r['exposures']==2]
    rows, noise = [], {}
    for r in selected:
        g = r['group']
        if g not in noise:
            noise[g] = rng.sample([t for t in range(200,40000) if t not in excluded],11)
            excluded.update(noise[g])
        seq, query = r['ctx'][:r['length']],r['ctx'][-r['prefix']:]
        values = noise[g]
        for short in (1,3):
            distractor = values[:6]+query[-short:]+[values[6]]+values[7:]
            for order in ('long_first','long_last'):
                ctx = (seq+distractor if order=='long_first' else distractor+seq)+query
                start = 0 if order=='long_first' else len(distractor)
                rows.append(dict(r,id=len(rows),source_id=r['id'],short=short,order=order,ctx=ctx,
                                 target_position=start+r['prefix']))
    mapping = {(r['source_id'],r['short'],r['order']):r['id'] for r in rows}
    pairs = [{'a':mapping[(r['base'],r['short'],r['order'])],'b':r['id'],'positions':[r['target_position']]}
             for r in rows if r['kind']=='intervention']
    write(a.out/'dataset.json',rows)
    write(a.out/'pairs.json',pairs)
    write(a.out/'noise.json',noise)
    library = BASELINE.read_text().replace('cdecide','bdecide')
    library += '\n.decl lc_option(c:number,minmatch:number,margin:number)\n'
    library += ''.join(f'lc_option({c},{k},{m}).\n' for c,k,m in OPTIONS)
    library += (ROOT/'dl/copy_longest_runtime.dl').read_text()
    (a.out/'library.dl').write_text(library)
    write(a.out/'protocol.json',{'scope':'historical-background discovery pilot, no final confirmation',
        'seed':502,'options':OPTIONS,'rows':len(rows),
        'files':{n:sha256(a.out/n) for n in ('dataset.json','pairs.json','noise.json','library.dl')},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'historical':{n:sha256(ROOT/n) for n in names},'baseline_sha256':sha256(BASELINE),
        'pinned_model':read(ROOT/'reference/benchmarks/qwen25_native_trace_b/model.json')})
    print(f'{len(rows)} source-selection pilot contexts fixed')


def report(a):
    rows = gate(a)
    refs = journal(a.out/'observations.jsonl')
    pairs = read(a.out/'pairs.json')
    facts = {'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['group']}\t{r['length']}\t{r['short']}\t{r['order']}\t{r['expected']}\n" for r in rows),
        'ref':''.join(f"{i}\t{r['response']['next']}\n" for i,r in refs.items()),
        'pair':''.join(f"{p['a']}\t{p['b']}\n" for p in pairs),
        'edit':''.join(f"{p['a']}\t{p['b']}\t{q}\n" for p in pairs for q in p['positions'])}
    x = check(ROOT/'dl/copy_longest_report.dl',a.out/'library.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/report',provenance={'source':'source-selection discovery, not final confirmation'})
    write(a.out/'report.json',x)
    if not x['certified']:
        raise ValueError(x.get('error','input integrity failed'))
    print('model',x['relations']['model_score'])
    print('hypotheses',x['relations']['score'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('inputs','collect','report'))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--binary',type=Path)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--port',type=int,default=8396)
    a=p.parse_args(); a.out=a.out.resolve()
    if a.phase=='collect' and not(a.binary and a.bundle):
        p.error('collection needs --binary and --bundle')
    globals()[a.phase](a)


if __name__=='__main__': main()
