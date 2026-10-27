"""Freeze and collect development-only decoy-successor counterfactuals."""
import argparse,json,random,time
from pathlib import Path
from certificate import check,sha256,replay
from run_qwen35_copy_inventory import ROOT,OUT as INVENTORY,MODEL,gate,read,write,journal
from qwen35_reference_evidence import extract,replay_archive
OUT=ROOT/'reference/benchmarks/qwen35_decoy_counterfactual'
SOURCES=('py/run_qwen35_decoy_counterfactual.py','dl/qwen35_decoy_plan.dl','py/qwen35_reference_evidence.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','dl/qwen35_reference_argmax.dl')
def inputs(a):
    rows=gate(type('Args',(),{'out':INVENTORY})()); rows=[r for r in rows if r['split']=='train' and not r['negative']]
    refs=journal(INVENTORY);pairs=read(INVENTORY/'pairs.json');ids={r['id'] for r in rows}
    a.out.mkdir(parents=True,exist_ok=False)
    module=ROOT/'dl/qwen35_decoy_plan.dl'
    result=check(module,module,{'sample':''.join(f"{r['id']}\t{r['group']}\t{r['exposures']}\t{r['order']}\t{r['expected']}\n" for r in rows),
        'pair':''.join(f"{p['a']}\t{p['b']}\n" for p in pairs if p['a'] in ids),
        'ref':''.join(f"{i}\t{refs[i]['next']}\n" for i in sorted(ids))},{},evidence_dir=a.out/'certificate-evidence/plan',provenance={'source':'training-only representative pairs; prefer departures, then lowest ID'})
    if not result['certified']:raise ValueError('invalid plan')
    write(a.out/'plan.json',result)
    assert replay(Path(result['evidence']))['relations']==result['relations']
    byid={r['id']:r for r in rows};rng=random.Random(2100)
    used=set(read(INVENTORY/'used-roles.json')['decoy']);pool=[t for t in read(INVENTORY/'pools.json')['decoy'] if t not in used]
    data=[]
    for aa,bb,g,e,o in result['relations']['selected']:
        replacement=rng.choice(pool);pool.remove(replacement)
        for old in (int(aa),int(bb)):
            r=byid[old];position=6+r['short'] if r['order']=='decoy_first' else r['exposures']*(r['length']+r['gap'])+6+r['short']
            ctx=r['ctx'][:];before=ctx[position];ctx[position]=replacement
            assert before in used and replacement not in r['ctx']
            data.append({'id':len(data),'base':old,'pair_a':int(aa),'group':g,'exposures':int(e),'order':o,'ctx':ctx,'position':position,'old_decoy':before,'new_decoy':replacement,'long_target':r['expected'],'base_reference':refs[old]['next']})
    write(a.out/'dataset.json',data)
    write(a.out/'protocol.json',{'scope':'development causal source edits; no final circuit selection or proof','seed':2100,
        'selection':'one pair per training group/exposure/order; prefer any designated-source departure, minimum ID; both long-target states receive same decoy edit',
        'contrasts':'old versus new decoy under both long-target states; observe tracking new decoy, long target, or unchanged old answer; no retuning final tests',
        'inventory_closed_sha256':sha256(INVENTORY/'closed.json'),
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen new contexts',len(data),flush=True)
def collect(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('input changed')
    for n,h in p['sources'].items():
        if sha256(ROOT/n)!=h:raise ValueError('source changed')
    if sha256(INVENTORY/'closed.json')!=p['inventory_closed_sha256']:raise ValueError('inventory changed')
    gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_lazy_reference import LazyReference
    torch.set_num_threads(2);model=LazyReference(MODEL,vocab_chunk=1024);seen=journal(a.out)
    with (a.out/'observations.jsonl').open('a') as stream:
        for r in read(a.out/'dataset.json'):
            i=r['id']
            if i in seen:continue
            path=a.out/'reference-evidence'/f'reference-{i:04d}.tar.gz';started=time.monotonic()
            if path.exists():
                h=sha256(path);e={'archive':path.name,'sha256':h,'query':replay_archive(path,h,row_id=i)}
            else:e=extract(model.logits(r['ctx']),path,i)
            if not e['query']['certified']:raise ValueError('invalid reference')
            record={'id':i,'next':int(e['query']['relations']['ref'][0][0]),'reference':e,'seconds':time.monotonic()-started}
            stream.write(json.dumps(record)+'\n');stream.flush();print('decoy',i+1,'/',36,flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','collect'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();globals()[a.phase](a)
if __name__=='__main__':main()
