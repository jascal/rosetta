"""Fresh long-consensus development. Python stages I/O, Datalog predicts and selects."""
import argparse
from pathlib import Path
import random

from certificate import check,sha256
from run_copy_prefix import ROOT,BASELINE,_tokens,collect,gate,journal,disjoint
from run_copy_longest import EXCLUDE as PREVIOUS
from run_copy_mechanism import read,write

OUT=ROOT/'reference/benchmarks/qwen25_copy_long_consensus'
OPTIONS=tuple((i,k,s) for i,(k,s) in enumerate(((k,s) for k in (16,24,32) for s in (2,3)),1))
EXCLUDE=PREVIOUS+('reference/benchmarks/qwen25_copy_longest/dataset.json',
    'reference/benchmarks/qwen25_copy_source_diagnosis/selected.json',
    'reference/benchmarks/qwen25_copy_heads/dataset.json')
SOURCES=('py/run_copy_long_consensus.py','py/run_copy_prefix.py','py/run_copy_mechanism.py',
    'py/run_copy_longest.py','py/run_native_trace2.py','py/certificate.py',
    'dl/copy_longest_runtime.dl','dl/copy_long_consensus_runtime.dl',
    'dl/copy_long_consensus_select.dl','docs/copy-long-consensus-plan.md')


def generate(excluded,seeds=(801,802),groups=2):
    used,rows,pairs=set(excluded),[],[]
    for seed,split in zip(seeds,('train','validation')):
        rng=random.Random(seed)
        for length in (32,48,64):
            prefix=length//2
            for group in range(groups):
                for negative in (False,True):
                    values=rng.sample([t for t in range(200,40000) if t not in used],length+44)
                    used.update(values)
                    seq=values[:length]; replacement,decoy=values[length:length+2]
                    head,tail=values[length+2:length+8],values[length+8:length+12]
                    gaps=[values[length+12+j*8:length+20+j*8] for j in range(4)]
                    query=seq[:prefix]
                    meta=dict(seed=seed,split=split,group=f's{seed}-l{length}-g{group}-n{int(negative)}',
                        length=length,prefix=prefix,negative=negative)
                    if negative:
                        for kind,ctx in (('no_repeat',seq),('tie',seq+head+query+[decoy]+tail+query)):
                            rows.append(dict(meta,id=len(rows),kind=kind,exposures=0,short=prefix,
                                order='control',gap=0,ctx=ctx,expected=seq[prefix]))
                        continue
                    for exposure in (2,3):
                        for short in (12,16):
                            distractor=head+query[-short:]+[decoy]+tail
                            for order in ('decoy_first','decoy_last'):
                                for n_gap in (0,8):
                                    blocks=([distractor]+[seq]*exposure if order=='decoy_first'
                                            else [seq]*exposure+[distractor])
                                    ctx,positions=[],[]
                                    for j,block in enumerate(blocks):
                                        is_source=(j>0 if order=='decoy_first' else j<exposure)
                                        if is_source: positions.append(len(ctx)+prefix)
                                        ctx.extend(block);ctx.extend(gaps[j][:n_gap])
                                    ctx.extend(query)
                                    fields=dict(meta,exposures=exposure,short=short,order=order,gap=n_gap)
                                    a=len(rows)
                                    rows.append(dict(fields,id=a,kind='repeat',ctx=ctx,expected=seq[prefix]))
                                    edited=ctx[:]
                                    for pos in positions: edited[pos]=replacement
                                    b=len(rows)
                                    rows.append(dict(fields,id=b,kind='intervention',ctx=edited,
                                        expected=replacement,base=a,positions=positions))
                                    pairs.append(dict(a=a,b=b,group=meta['group'],positions=positions))
    return rows,pairs


def library():
    text=BASELINE.read_text().replace('cdecide','bdecide')
    text+='\n.decl lg_option(c:number,minmatch:number,support:number)\n'
    text+=''.join(f'lg_option({c},{k},{s}).\n' for c,k,s in OPTIONS)
    text+='.decl lc_option(c:number,minmatch:number,margin:number)\n'
    text+='lc_option(C,M,4) :- lg_option(C,M,_).\n'
    text+=(ROOT/'dl/copy_longest_runtime.dl').read_text()
    text+=(ROOT/'dl/copy_long_consensus_runtime.dl').read_text()
    return text


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    excluded=set()
    for n in EXCLUDE: excluded.update(_tokens(n))
    rows,pairs=generate(excluded)
    stats=disjoint(rows,excluded)
    write(a.out/'dataset.json',rows);write(a.out/'pairs.json',pairs)
    write(a.out/'excluded-tokens.json',sorted(excluded))
    (a.out/'library.dl').write_text(library())
    (a.out/'baseline.dl').write_text(BASELINE.read_text())
    write(a.out/'protocol.json',{'scope':'fresh development only, no final references',
        'seeds':[801,802],'groups_per_length':2,'options':OPTIONS,'disjoint':stats,
        'files':{n:sha256(a.out/n) for n in ('dataset.json','pairs.json','excluded-tokens.json','library.dl','baseline.dl')},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},'exclusions':{n:sha256(ROOT/n) for n in EXCLUDE},
        'baseline_sha256':sha256(BASELINE),
        'pinned_model':read(ROOT/'reference/benchmarks/qwen25_native_trace_b/model.json')})
    print(stats)


def select(a):
    rows=gate(a);refs=journal(a.out/'observations.jsonl');pairs=read(a.out/'pairs.json')
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['split']}\t{r['group']}\n" for r in rows),
        'ref':''.join(f"{i}\t{r['response']['next']}\n" for i,r in refs.items()),
        'pair':''.join(f"{p['a']}\t{p['b']}\t{p['group']}\n" for p in pairs),
        'edit':''.join(f"{p['a']}\t{p['b']}\t{j}\n" for p in pairs for j in p['positions'])}
    x=check(ROOT/'dl/copy_long_consensus_select.dl',a.out/'library.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/development-select',
        provenance={'source':'pinned f32 Qwen long-consensus development; not final confirmation'})
    write(a.out/'selection.json',x)
    if not x['certified']:raise ValueError(x.get('error','invalid inputs'))
    for n in ('score','ngroups','incremental_score','lg_selected'):print(n,x['relations'][n])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('inputs','collect','select'))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--binary',type=Path);p.add_argument('--bundle',type=Path)
    p.add_argument('--port',type=int,default=8396)
    a=p.parse_args();a.out=a.out.resolve()
    if a.phase=='collect' and not(a.binary and a.bundle):p.error('collection needs --binary and --bundle')
    globals()[a.phase](a)


if __name__=='__main__':main()
