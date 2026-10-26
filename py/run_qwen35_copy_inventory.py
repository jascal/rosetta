"""Frozen 4B inventory: corpus/oracle I/O, train-only Datalog baseline proposals."""
import argparse
import json
from pathlib import Path
import random
import time
from certificate import check,sha256
from run_copy_mechanism import ROOT,read,write
from run_copy_prefix import disjoint
from native_trace import program,emit
from qwen35_reference_evidence import extract,replay_archive
from run_qwen35_checkpoint_smoke import MODEL,REVISION,OUT as SMOKE

OUT=ROOT/'reference/benchmarks/qwen35_copy_inventory'
SOURCES=('py/run_qwen35_copy_inventory.py','py/qwen35_reference_evidence.py','py/qwen35_lazy_reference.py',
 'py/qwen35_stream_reference.py','py/native_trace.py','py/benchmark_copy_generalization.py',
 'py/run_copy_prefix.py','py/run_copy_mechanism.py','py/certificate.py','py/run_qwen35_checkpoint_smoke.py',
 'dl/native_trace_runtime.dl','dl/qwen35_reference_argmax.dl','dl/qwen35_baseline_propose.dl',
 'dl/qwen35_baseline_admit.dl','dl/qwen35_inventory_inputs.dl','dl/copy_longest_runtime.dl','docs/qwen35-copy-inventory-plan.md')

def generate(pools,seeds=(2001,2002),groups=1,used=None,labels=('train','validation')):
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
                    meta=dict(seed=seed,split=split,group=f'q35-s{seed}-l{length}-g{group}-n{int(negative)}',
                        length=length,prefix=prefix,negative=negative)
                    if negative:
                        for kind,ctx in (('no_repeat',seq),('tie',seq+head+query+[decoy]+tail+query)):
                            rows.append(dict(meta,id=len(rows),kind=kind,exposures=0,short=prefix,
                                order='control',gap=0,ctx=ctx,expected=target))
                        continue
                    for exposure in (1,2,3):
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


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    excluded={t for r in read(SMOKE/'contexts.json') for t in r['ids']}
    values=random.Random(2000).sample([t for t in range(40000,140000) if t not in excluded],10752)
    pools={};offset=0
    for role,n in (('query',512),('target',1024),('decoy',1024),('background',8192)):
        pools[role]=values[offset:offset+n];offset+=n
    rows,pairs,used=generate(pools);stats=disjoint(rows,excluded)
    for n,data in (('dataset.json',rows),('pairs.json',pairs),('pools.json',pools),('used-roles.json',used),('excluded-tokens.json',sorted(excluded))):
        write(a.out/n,data)
    (a.out/'hypotheses.dl').write_text(program())
    core=a.out/'input-core.dl'
    core.write_text('.decl lc_option(c:number,minmatch:number,margin:number)\n'+(ROOT/'dl/copy_longest_runtime.dl').read_text())
    integrity=check(ROOT/'dl/qwen35_inventory_inputs.dl',core,{
        'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['split']}\t{r['group']}\t{int(r['negative'])}\t{r['exposures']}\t{r['prefix']}\t{r['expected']}\n" for r in rows),
        'pair':''.join(f"{p['a']}\t{p['b']}\t{p['group']}\n" for p in pairs),
        'edit':''.join(f"{p['a']}\t{p['b']}\t{j}\n" for p in pairs for j in p['positions']),
        'excluded':''.join(f'{t}\n' for t in sorted(excluded))},{},
        evidence_dir=a.out/'certificate-evidence/input-integrity',provenance={'source':'reference-free complete inventory audit'})
    write(a.out/'input-integrity.json',integrity)
    if not integrity['certified']:raise ValueError(integrity.get('error','invalid input corpus'))
    write(a.out/'protocol.json',{'scope':'new-model discovery/development inventory, not final confirmation',
        'model':'Qwen/Qwen3.5-4B','revision':REVISION,'seeds':[2001,2002],'pool_seed':2000,
        'exposures':[1,2,3],'groups_per_length':1,'disjoint':stats,'batch_size':1,'threads':2,'cache':False,
        'checkpoint_files':read(SMOKE/'protocol.json')['checkpoint_files'],
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'instrument_gate':{'path':'reference/benchmarks/qwen35_4b_probe_validation/closed.json',
            'sha256':sha256(ROOT/'reference/benchmarks/qwen35_4b_probe_validation/closed.json')}})
    print('frozen:',stats,flush=True)


def journal(out):
    result={};path=out/'observations.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['id'] in result:raise ValueError('duplicate reference')
            result[row['id']]=row
    return result


def gate(a,weights=False):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'input changed: {n}')
    for n,h in p['sources'].items():
        if sha256(ROOT/n)!=h:raise ValueError(f'source changed: {n}')
    if sha256(ROOT/p['instrument_gate']['path'])!=p['instrument_gate']['sha256']:raise ValueError('instrument evidence changed')
    if weights:
        for n,h in p['checkpoint_files'].items():
            if sha256(MODEL/n)!=h:raise ValueError(f'checkpoint changed: {n}')
    return read(a.out/'dataset.json')


def proposal_gate(out):
    frozen=read(out/'proposals-frozen.json')
    if sha256(out/'protocol.json')!=frozen['protocol_sha256']:raise ValueError('proposal protocol changed')
    for n,h in frozen['files'].items():
        if sha256(out/n)!=h:raise ValueError(f'proposal changed: {n}')


def collect(a):
    import torch
    from qwen35_lazy_reference import LazyReference
    rows=gate(a,weights=True);observed=journal(a.out)
    if a.split=='validation':proposal_gate(a.out)
    if set(observed)-{r['id'] for r in rows}:raise ValueError('unplanned references')
    for record in observed.values():
        if sha256(a.out/'reference-evidence'/record['reference']['archive'])!=record['reference']['sha256']:
            raise ValueError('reference evidence changed')
    torch.set_num_threads(2);model=LazyReference(MODEL,vocab_chunk=1024)
    selected=[r for r in rows if r['split']==a.split]
    with (a.out/'observations.jsonl').open('a') as stream:
        for j,row in enumerate(selected):
            i=row['id']
            if i in observed:continue
            path=a.out/'reference-evidence'/f'reference-{i:04d}.tar.gz';started=time.monotonic()
            if path.exists():
                digest=sha256(path);recovered=replay_archive(path,digest,row_id=i)
                evidence={'query':recovered,'archive':path.name,'sha256':digest}
            else:evidence=extract(model.logits(row['ctx']),path,i)
            if not evidence['query']['certified']:raise ValueError('reference query invalid')
            refs=evidence['query']['relations']['ref']
            if len(refs)!=1:raise ValueError('reference query is not functional')
            record={'id':i,'next':int(refs[0][0]),'reference':evidence,'seconds':time.monotonic()-started}
            stream.write(json.dumps(record)+'\n');stream.flush()
            print(a.split,j+1,'/',len(selected),'row',i,f"{record['seconds']:.2f}s",flush=True)


def propose(a):
    rows=gate(a);observed=journal(a.out)
    if any(r['split']=='validation' and r['id'] in observed for r in rows):raise ValueError('validation already observed')
    module=ROOT/'dl/qwen35_baseline_propose.dl'
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['split']}\t{r['group']}\n" for r in rows),
        'ref':''.join(f"{i}\t{r['next']}\n" for i,r in observed.items())}
    result=check(module,a.out/'hypotheses.dl',facts,{},evidence_dir=a.out/'certificate-evidence/propose',
        provenance={'source':'train-only behavior baseline ranking; validation references forbidden'})
    write(a.out/'proposals.json',result)
    if not result['certified']:raise ValueError(result.get('error','incomplete training'))
    emit(a.out/'proposals',[(int(r),int(c)) for r,c in result['relations']['proposed']])
    paths=[a.out/'proposals.json',*(a.out/'proposals').iterdir()]
    write(a.out/'proposals-frozen.json',{'protocol_sha256':sha256(a.out/'protocol.json'),
        'files':{str(p.relative_to(a.out)):sha256(p) for p in paths}})
    print('frozen train-only proposals:',result['relations']['proposed'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('inputs','collect','propose'))
    parser.add_argument('--split',choices=('train','validation'),default='train')
    parser.add_argument('--out',type=Path,default=OUT);a=parser.parse_args();a.out=a.out.resolve()
    globals()[a.phase](a)


if __name__=='__main__':main()

