"""Fresh long-context competition development. All candidate/admission decisions are Datalog."""
import argparse
import random
from pathlib import Path

from certificate import check, sha256
from run_copy_prefix import ROOT, BASELINE, EXCLUSIONS, _tokens, collect, gate, journal, disjoint
from run_copy_mechanism import read, write

OUT=ROOT/'reference/benchmarks/qwen25_copy_longest'
OPTIONS=tuple((i,k,a) for i,(k,a) in enumerate(((k,a) for k in (6,12,24) for a in (0,1)),1))
EXCLUDE=EXCLUSIONS+('reference/benchmarks/qwen25_copy_prefix/dataset.json',
    'reference/benchmarks/qwen25_copy_corruption_pilot/dataset.json',
    'reference/benchmarks/qwen25_copy_longest_pilot/dataset.json')
SOURCES=('py/run_copy_longest.py','py/run_copy_prefix.py','py/run_copy_mechanism.py',
    'py/run_native_trace2.py','py/certificate.py','dl/copy_longest_runtime.dl',
    'dl/copy_longest_anchor.dl','dl/copy_longest_select.dl','docs/copy-longest-development-plan.md')


def generate(excluded,seeds=(601,602),groups=3):
    used,rows,pairs=set(excluded),[],[]
    for seed,split in zip(seeds,('train','validation')):
        rng=random.Random(seed)
        for length,prefix in ((24,12),(48,24),(64,32)):
            for group in range(groups):
                for negative in (False,True):
                    values=rng.sample([t for t in range(200,40000) if t not in used],length+29)
                    used.update(values)
                    seq=values[:length]
                    replacement,decoy,padding=values[length:length+3]
                    head,tail=values[length+3:length+9],values[length+9:length+13]
                    between,after=values[length+13:length+21],values[length+21:]
                    query=seq[:prefix]
                    meta=dict(seed=seed,split=split,group=f's{seed}-l{length}-g{group}-n{int(negative)}',
                              length=length,prefix=prefix,negative=negative)
                    if negative:
                        for kind,ctx in (('no_repeat',seq),('tie',seq+head+query+[decoy]+tail+query)):
                            rows.append(dict(meta,id=len(rows),kind=kind,short=prefix,order='control',gap=0,
                                             ctx=ctx,expected=seq[prefix]))
                        continue
                    for short in (3,6,9):
                        distractor=head+query[-short:]+[decoy]+tail
                        for order in ('long_first','long_last','long_first_padded'):
                            for n_gap in (0,8):
                                if order=='long_last':
                                    body=distractor+between[:n_gap]+seq
                                    source=len(distractor)+n_gap
                                else:
                                    lead=[padding] if order=='long_first_padded' else []
                                    body=lead+seq+between[:n_gap]+distractor
                                    source=len(lead)
                                context=body+after[:n_gap]+query
                                a=len(rows)
                                fields=dict(meta,short=short,order=order,gap=n_gap)
                                rows.append(dict(fields,id=a,kind='repeat',ctx=context,expected=seq[prefix]))
                                edited=context[:]; edited[source+prefix]=replacement
                                b=len(rows)
                                rows.append(dict(fields,id=b,kind='intervention',ctx=edited,expected=replacement,
                                                 base=a,positions=[source+prefix]))
                                pairs.append(dict(a=a,b=b,group=meta['group'],positions=[source+prefix]))
    return rows,pairs


def library():
    text=BASELINE.read_text().replace('cdecide','bdecide')
    text+='\n.decl ls_option(c:number,minmatch:number,anchor:number)\n'
    text+=''.join(f'ls_option({c},{k},{a}).\n' for c,k,a in OPTIONS)
    text+=(ROOT/'dl/copy_longest_runtime.dl').read_text()
    text+=(ROOT/'dl/copy_longest_anchor.dl').read_text()
    return text


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    excluded=set()
    for name in EXCLUDE: excluded.update(_tokens(name))
    rows,pairs=generate(excluded)
    stats=disjoint(rows,excluded)
    write(a.out/'dataset.json',rows); write(a.out/'pairs.json',pairs)
    write(a.out/'excluded-tokens.json',sorted(excluded))
    (a.out/'library.dl').write_text(library())
    (a.out/'baseline.dl').write_text(BASELINE.read_text())
    write(a.out/'protocol.json',{'scope':'fresh development only; no final reference requests',
        'seeds':[601,602],'groups_per_length':3,'options':OPTIONS,'disjoint':stats,
        'files':{n:sha256(a.out/n) for n in ('dataset.json','pairs.json','excluded-tokens.json','library.dl','baseline.dl')},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'exclusions':{n:sha256(ROOT/n) for n in EXCLUDE},'baseline_sha256':sha256(BASELINE),
        'pinned_model':read(ROOT/'reference/benchmarks/qwen25_native_trace_b/model.json')})
    print(stats)


def select(a):
    rows=gate(a); refs=journal(a.out/'observations.jsonl'); pairs=read(a.out/'pairs.json')
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['split']}\t{r['group']}\n" for r in rows),
        'ref':''.join(f"{i}\t{r['response']['next']}\n" for i,r in refs.items()),
        'pair':''.join(f"{p['a']}\t{p['b']}\t{p['group']}\n" for p in pairs),
        'edit':''.join(f"{p['a']}\t{p['b']}\t{j}\n" for p in pairs for j in p['positions'])}
    x=check(ROOT/'dl/copy_longest_select.dl',a.out/'library.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/development-select',
        provenance={'source':'pinned f32 Qwen fresh long-source development, not final confirmation'})
    write(a.out/'selection.json',x)
    if not x['certified']: raise ValueError(x.get('error','invalid inputs'))
    for name in ('score','ngroups','incremental_score','ls_selected'):
        print(name,x['relations'][name])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('inputs','collect','select'))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--binary',type=Path); p.add_argument('--bundle',type=Path)
    p.add_argument('--port',type=int,default=8396)
    a=p.parse_args();a.out=a.out.resolve()
    if a.phase=='collect' and not(a.binary and a.bundle): p.error('collection needs --binary and --bundle')
    globals()[a.phase](a)


if __name__=='__main__': main()
