"""Answer-position head exchange: frozen Datalog plan, upstream oracle I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from certificate import check,replay,sha256
from run_copy_mechanism import ROOT,read,write

OUT=ROOT/'reference/benchmarks/qwen25_copy_head_exchange'
HIST=ROOT/'reference/benchmarks/qwen25_copy_longest'
HEADS=ROOT/'reference/benchmarks/qwen25_copy_heads'
DIAG=ROOT/'reference/benchmarks/qwen25_copy_source_diagnosis'
SOURCES=('py/run_copy_head_exchange.py','py/certificate.py','py/run_copy_mechanism.py',
    'dl/copy_head_exchange_plan.dl','dl/copy_head_exchange_summary.dl','docs/copy-head-exchange-plan.md')


def query(out,name,facts):
    module=ROOT/f'dl/copy_head_exchange_{name}.dl'
    result=check(module,module,facts,{},evidence_dir=out/f'certificate-evidence/{name}',
        provenance={'source':'historical upstream answer-position exchange; instrument integrity only'})
    write(out/f'{name}.json',result)
    return result


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    rows=read(HIST/'dataset.json');pairs=read(HIST/'pairs.json')
    controls=read(HEADS/'plan.json')['relations']['chosen_control']
    chosen=query(a.out,'plan',{
        'known':''.join(f"{r['id']}\n" for r in rows),
        'selected':''.join(f"{r['id']}\n" for r in read(DIAG/'selected.json')),
        'pairing':''.join(f"{p['a']}\t{p['b']}\n" for p in pairs),
        'control':''.join(f'{l}\t{h}\n' for l,h,k in controls if int(k)<2)})
    if not chosen['certified']:raise ValueError(chosen.get('error','invalid plan'))
    ids={int(i) for i,d in chosen['relations']['donor']}
    write(a.out/'dataset.json',[r for r in rows if r['id'] in ids])
    write(a.out/'pairs.json',[p for p in pairs if p['a'] in ids and p['b'] in ids])
    historical={r['id']:r['response'] for r in map(json.loads,(HIST/'observations.jsonl').read_text().splitlines())}
    write(a.out/'historical.json',{str(i):historical[i] for i in sorted(ids)})
    # Pair construction invariants are checked before any intervention.
    facts={r['id']:r for r in rows}
    for p in read(a.out/'pairs.json'):
        x,y=facts[p['a']],facts[p['b']]
        if len(x['ctx'])!=len(y['ctx']):raise ValueError('pair length differs')
        if [j for j,(t,u) in enumerate(zip(x['ctx'],y['ctx'])) if t!=u]!=p['positions']:
            raise ValueError('unregistered pair edit')
        if x['ctx'][-x['prefix']:]!=y['ctx'][-y['prefix']:]:raise ValueError('queries differ')
    old=read(HEADS/'frozen.json');checkpoint=Path(old['checkpoint'])
    for n,h in old['checkpoint_sha256'].items():
        if sha256(checkpoint/n)!=h:raise ValueError('checkpoint changed')
    write(a.out/'frozen.json',{'scope':'historical empirical answer-position exchange; not a final circuit test',
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'historical':{str(p.relative_to(ROOT)):sha256(p) for p in
            (HIST/'dataset.json',HIST/'pairs.json',HIST/'observations.jsonl',HEADS/'plan.json',HEADS/'frozen.json',DIAG/'selected.json')},
        'checkpoint':str(checkpoint),'checkpoint_sha256':old['checkpoint_sha256'],
        'oracle':{'precision':'float32','attention':'eager','threads':2}})
    print('frozen:',len(ids),'recipients,',len(chosen['relations']['request']),'requests',flush=True)


def gate(a):
    frozen=read(a.out/'frozen.json')
    for n,h in frozen['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'frozen input changed: {n}')
    for n,h in (frozen['sources']|frozen['historical']).items():
        if sha256(ROOT/n)!=h:raise ValueError(f'frozen source changed: {n}')
    return frozen,read(a.out/'plan.json')['relations'],{r['id']:r for r in read(a.out/'dataset.json')}


def load(path):
    result={}
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line);key=(r['i'],r['set'],r['mode'],r['donor'])
            if key in result:raise ValueError('duplicate intervention')
            result[key]=r
    return result


def collect(a):
    import torch
    import transformers
    frozen,plan,rows=gate(a);checkpoint=Path(frozen['checkpoint'])
    for n,h in frozen['checkpoint_sha256'].items():
        if sha256(checkpoint/n)!=h:raise ValueError('checkpoint changed')
    torch.set_num_threads(2)
    model=transformers.AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float32,
        attn_implementation='eager',local_files_only=True).eval()
    dim=model.config.hidden_size//model.config.num_attention_heads
    sets={s:[(int(l),int(h)) for name,l,h in plan['member'] if name==s] for s, in plan['head_set']}
    members=sorted({x for values in sets.values() for x in values})
    historical=read(a.out/'historical.json');products={};unpatched={}
    with torch.no_grad():
        for i,row in sorted(rows.items()):
            cache={};hooks=[]
            for l in sorted({l for l,h in members}):
                def capture(_module,args,l=l):
                    cache[l]=args[0][0,-1].detach().clone()
                hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(capture))
            try:unpatched[i]=model(torch.tensor([row['ctx']]),use_cache=False).logits[0,-1].detach().clone()
            finally:
                for hook in hooks:hook.remove()
            products[i]=cache
        activations={'scope':'unpatched last-position o_proj inputs; hashes of contiguous float32 bytes',
            'vectors':{str(i):{str(l):hashlib.sha256(x.numpy().tobytes()).hexdigest() for l,x in layers.items()} for i,layers in products.items()},
            'delta':[]}
        for i,d in plan['donor']:
            i,d=int(i),int(d)
            for name,heads in sets.items():
                difference=sum(float((products[i][l][h*dim:(h+1)*dim]-products[d][l][h*dim:(h+1)*dim]).abs().sum()) for l,h in heads)
                activations['delta'].append([i,name,difference])
        write(a.out/'activations.json',activations)
        write(a.out/'upstream-build.json',{'torch':torch.__version__,'transformers':transformers.__version__,
            'head_dim':dim,'precision':'float32','attention':'eager','threads':2})
        path=a.out/'observations.jsonl';cached=load(path)
        requests=[(int(i),name,mode,int(d)) for i,name,mode,d in plan['request']]
        if set(cached)-set(requests):raise ValueError('unplanned intervention')
        with path.open('a') as stream:
            for j,(i,name,mode,d) in enumerate(requests):
                if (i,name,mode,d) in cached:continue
                hooks=[];started=time.monotonic()
                if mode=='baseline':logits=unpatched[i]
                else:
                    for l in sorted({l for l,h in sets[name]}):
                        heads=tuple(h for layer,h in sets[name] if layer==l)
                        def patch(_module,args,l=l,heads=heads):
                            x=args[0].clone()
                            for h in heads:
                                section=slice(h*dim,(h+1)*dim)
                                x[0,-1,section]=0.0 if mode=='zero' else products[d][l][section]
                            return (x,)+args[1:]
                        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(patch))
                    try:logits=model(torch.tensor([rows[i]['ctx']]),use_cache=False).logits[0,-1]
                    finally:
                        for hook in hooks:hook.remove()
                original=historical[str(i)]['next'];donor_ref=historical[str(d)]['next'];gold=rows[i]['expected']
                record={'i':i,'set':name,'mode':mode,'donor':d,'next':int(logits.argmax()),
                    'ref_logit':float(logits[original]),'gold_logit':float(logits[gold]),
                    'donor_logit':float(logits[donor_ref]),'seconds':time.monotonic()-started}
                stream.write(json.dumps(record)+'\n');stream.flush()
                if (j+1)%24==0:print(j+1,'/',len(requests),flush=True)


def evaluate(a):
    _,plan,rows=gate(a);observed=load(a.out/'observations.jsonl');hist=read(a.out/'historical.json')
    result=query(a.out,'summary',{
        'request':''.join(f'{i}\t{name}\t{mode}\t{d}\n' for i,name,mode,d in plan['request']),
        'donor':''.join(f'{i}\t{d}\n' for i,d in plan['donor']),
        'historical':''.join(f"{i}\t{hist[str(i)]['next']}\t{r['expected']}\t{hist[str(i)]['next_logit']:.17g}\t{dict(hist[str(i)]['logits'])[r['expected']]:.17g}\n" for i,r in rows.items()),
        'result':''.join(f"{r['i']}\t{r['set']}\t{r['mode']}\t{r['donor']}\t{r['next']}\t{r['ref_logit']:.17g}\t{r['gold_logit']:.17g}\t{r['donor_logit']:.17g}\n" for r in observed.values()),
        'delta':''.join(f'{i}\t{name}\t{x:.17g}\n' for i,name,x in read(a.out/'activations.json')['delta'])})
    audits=[]
    for name in ('plan','summary'):
        before=read(a.out/f'{name}.json');after=replay(Path(before['evidence']))
        if before['relations']!=after['relations'] or before['certified']!=after['certified']:raise ValueError('replay differs')
        audits.append(before['evidence'])
    write(a.out/'replay-audit.json',{'queries':audits,'unchanged':True})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('instrument:',result['certified'],result.get('error'))
    for name in ('counts','reference_eligible','zero_delta','parity_bad','invalid'):print(name,result.get('relations',{}).get(name))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','collect','evaluate'))
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();a.out=a.out.resolve()
    globals()[a.phase](a)


if __name__=='__main__':main()
