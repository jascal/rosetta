"""Fresh prior-agreement contexts; all feature values are frozen beforehand."""
import argparse
from pathlib import Path
import random

from certificate import check,sha256
from run_copy_prefix import ROOT,BASELINE,collect,gate,journal,disjoint
from run_copy_mechanism import read,write
from build_copy_prior_lexicon import OUT as LEXICON

OUT=ROOT/'reference/benchmarks/qwen25_copy_prior_guard'
OPTIONS=tuple((i,k,s) for i,(k,s) in enumerate(((k,s) for k in (16,24,32) for s in (2,3)),1))
SOURCES=('py/run_copy_prior_guard.py','py/run_copy_prefix.py','py/run_copy_mechanism.py','py/certificate.py',
    'dl/copy_longest_runtime.dl','dl/copy_long_consensus_runtime.dl','dl/copy_prior_guard_runtime.dl',
    'dl/copy_prior_guard_select.dl','dl/copy_prior_guard_domains.dl','dl/copy_prior_specialize.dl',
    'docs/copy-prior-guard-plan.md','docs/copy-prior-guard-specialization.md')


def generate(pools,seeds=(911,912),groups=3,used=None,labels=('train','validation')):
    used={k:set((used or {}).get(k,())) for k in pools}
    rows,pairs=[],[]
    for seed,split in zip(seeds,labels):
        rng=random.Random(seed)
        def draw(role,n):
            values=rng.sample([t for t in pools[role] if t not in used[role]],n)
            used[role].update(values)
            return values
        for length in (32,48,64):
            prefix=length//2
            for group in range(groups):
                for negative in (False,True):
                    q=draw('query',1)[0];target,replacement=draw('target',2);decoy=draw('decoy',1)[0]
                    bg=draw('background',length+42)
                    seq=bg[:length];seq[prefix-1]=q;seq[prefix]=target
                    head,tail=bg[length:length+6],bg[length+6:length+10]
                    gaps=[bg[length+10+j*8:length+18+j*8] for j in range(4)]
                    query=seq[:prefix]
                    meta=dict(seed=seed,split=split,group=f'pg-s{seed}-l{length}-g{group}-n{int(negative)}',
                        length=length,prefix=prefix,negative=negative)
                    if negative:
                        for kind,ctx in (('no_repeat',seq),('tie',seq+head+query+[decoy]+tail+query)):
                            rows.append(dict(meta,id=len(rows),kind=kind,exposures=0,short=prefix,
                                order='control',gap=0,ctx=ctx,expected=target))
                        continue
                    for exposure in (2,3):
                        for short in (12,16):
                            distractor=head+query[-short:]+[decoy]+tail
                            for order in ('decoy_first','decoy_last'):
                                for n_gap in (0,8):
                                    blocks=([distractor]+[seq]*exposure if order=='decoy_first' else [seq]*exposure+[distractor])
                                    ctx,positions=[],[]
                                    for j,block in enumerate(blocks):
                                        if (j>0 if order=='decoy_first' else j<exposure):positions.append(len(ctx)+prefix)
                                        ctx.extend(block);ctx.extend(gaps[j][:n_gap])
                                    ctx.extend(query)
                                    fields=dict(meta,exposures=exposure,short=short,order=order,gap=n_gap)
                                    a=len(rows);rows.append(dict(fields,id=a,kind='repeat',ctx=ctx,expected=target))
                                    edited=ctx[:]
                                    for pos in positions:edited[pos]=replacement
                                    b=len(rows);rows.append(dict(fields,id=b,kind='intervention',ctx=edited,
                                        expected=replacement,base=a,positions=positions))
                                    pairs.append(dict(a=a,b=b,group=meta['group'],positions=positions))
    return rows,pairs,{k:sorted(v) for k,v in used.items()}


def feature_gate():
    feature=read(LEXICON/'feature.json')
    for n,h in feature['files'].items():
        if sha256(LEXICON/n)!=h:raise ValueError(f'frozen lexical feature changed: {n}')
    for n,h in feature['sources'].items():
        if sha256(ROOT/n)!=h:raise ValueError(f'frozen feature source changed: {n}')
    return read(LEXICON/'pools.json')


def rule_body(options=OPTIONS):
    text='.decl pg_option(c:number,minmatch:number,support:number)\n'
    text+=''.join(f'pg_option({c},{k},{s}).\n' for c,k,s in options)
    text+='.decl lg_option(c:number,minmatch:number,support:number)\nlg_option(C,M,S) :- pg_option(C,M,S).\n'
    text+='.decl lc_option(c:number,minmatch:number,margin:number)\nlc_option(C,M,4) :- lg_option(C,M,_).\n'
    for n in ('copy_longest_runtime.dl','copy_long_consensus_runtime.dl','copy_prior_guard_runtime.dl'):
        text+=(ROOT/'dl'/n).read_text()+'\n'
    return text


def specialize(out,rows):
    skeleton=out/'specialization-core.dl'
    skeleton.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\nlc_option(0,16,4).\n'+
        (ROOT/'dl/copy_longest_runtime.dl').read_text())
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'domain':''.join(f"{r['id']}\n" for r in rows),
        'raw_prior':(LEXICON/'prior-bands.facts').read_text()}
    x=check(ROOT/'dl/copy_prior_specialize.dl',skeleton,facts,{'nselected':int},
        evidence_dir=out/'certificate-evidence/specialization',
        provenance={'source':'reference-free lexical key specialization; no values or answers used to choose keys'})
    write(out/'specialization.json',x)
    if not x['certified']:raise ValueError(x.get('error','missing or invalid lexical facts'))
    return ('.decl prior_band(q:number,t:number,b:number)\n'+
        ''.join(f'prior_band({q},{t},{b}).\n' for q,t,b in x['relations']['selected_prior']))


def inputs(a):
    pools=feature_gate()
    a.out.mkdir(parents=True,exist_ok=False)
    rows,pairs,used=generate(pools)
    excluded=set(read(LEXICON/'excluded-tokens.json'));stats=disjoint(rows,excluded)
    write(a.out/'dataset.json',rows);write(a.out/'pairs.json',pairs);write(a.out/'used-roles.json',used)
    write(a.out/'excluded-tokens.json',sorted(excluded))
    lexical=specialize(a.out,rows)
    (a.out/'lexical.dl').write_text(lexical)
    (a.out/'baseline.dl').write_text(BASELINE.read_text())
    (a.out/'library.dl').write_text(BASELINE.read_text().replace('cdecide','bdecide')+'\n'+rule_body()+lexical)
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['split']}\t{r['group']}\n" for r in rows)}
    domains=check(ROOT/'dl/copy_prior_guard_domains.dl',a.out/'library.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/development-domains',provenance={'source':'reference-free prior-guard domains'})
    write(a.out/'development-domains.json',domains)
    if not domains['certified']:raise ValueError(domains.get('error','invalid domain query'))
    old=read(LEXICON/'protocol.json')
    write(a.out/'protocol.json',{'scope':'fresh development, shared intrinsic lexicon frozen before these contexts',
        'seeds':[911,912],'groups_per_length':3,'options':OPTIONS,'disjoint':stats,
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'exclusions':{str(p.relative_to(ROOT)):sha256(p) for p in
            (LEXICON/'feature.json',LEXICON/'pools.json',LEXICON/'prior-bands.facts')},
        'baseline_sha256':sha256(BASELINE),'pinned_model':old['pinned_model']})
    print(stats,'specialized lexical bytes:',len(lexical.encode()))
    print('domains:',domains['relations']['counts'])


def select(a):
    rows=gate(a);observed=journal(a.out/'observations.jsonl');pairs=read(a.out/'pairs.json')
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['split']}\t{r['group']}\n" for r in rows),
        'ref':''.join(f"{i}\t{r['response']['next']}\n" for i,r in observed.items()),
        'pair':''.join(f"{p['a']}\t{p['b']}\t{p['group']}\n" for p in pairs),
        'edit':''.join(f"{p['a']}\t{p['b']}\t{j}\n" for p in pairs for j in p['positions'])}
    x=check(ROOT/'dl/copy_prior_guard_select.dl',a.out/'library.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/development-select',provenance={'source':'fresh prior-guard development, no final references'})
    write(a.out/'selection.json',x)
    if not x['certified']:raise ValueError(x.get('error','invalid inputs'))
    for n in ('score','ngroups','incremental_score','pg_selected'):print(n,x['relations'][n])


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
