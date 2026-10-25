"""Separate attention routing and contextual values on a fixed historical panel."""
import argparse
import json
from pathlib import Path
import shutil
import time

from certificate import check,replay,sha256
from run_copy_head_exchange import ROOT,OUT as PREVIOUS,read,write,load

OUT=ROOT/'reference/benchmarks/qwen25_copy_head_factorization'
SOURCES=('py/run_copy_head_factorization.py','py/run_copy_head_exchange.py','py/run_copy_mechanism.py',
    'py/certificate.py','dl/copy_head_exchange_summary.dl','dl/copy_head_factorization_plan.dl',
    'dl/copy_head_factorization_summary.dl','docs/copy-head-factorization-plan.md')


def query(out,name,circuit,facts):
    result=check(ROOT/f'dl/copy_head_factorization_{name}.dl',circuit,facts,{},
        evidence_dir=out/f'certificate-evidence/{name}',provenance={'source':'historical upstream factorization; instrument integrity, not model equivalence'})
    write(out/f'{name}.json',result)
    return result


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    old=read(PREVIOUS/'plan.json')['relations']
    for n in ('dataset.json','pairs.json','historical.json'):shutil.copyfile(PREVIOUS/n,a.out/n)
    write(a.out/'members.json',old['member']);write(a.out/'donors.json',old['donor'])
    module=ROOT/'dl/copy_head_factorization_plan.dl'
    plan=query(a.out,'plan',module,{'member':''.join(f'{s}\t{l}\t{h}\n' for s,l,h in old['member']),
        'donor':''.join(f'{i}\t{d}\n' for i,d in old['donor'])})
    if not plan['certified']:raise ValueError(plan.get('error','invalid plan'))
    (a.out/'base-checker.dl').write_text((ROOT/'dl/copy_head_exchange_summary.dl').read_text().replace('certified','base_certified'))
    previous=read(PREVIOUS/'frozen.json')
    write(a.out/'frozen.json',{'scope':'historical routing/value factorization, not a final circuit test',
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'previous':{str(p.relative_to(ROOT)):sha256(p) for p in
            (PREVIOUS/'plan.json',PREVIOUS/'frozen.json',PREVIOUS/'closed.json',PREVIOUS/'observations.jsonl',PREVIOUS/'activations.json')},
        'checkpoint':previous['checkpoint'],'checkpoint_sha256':previous['checkpoint_sha256']})
    print('frozen',len(plan['relations']['request']),'requests',flush=True)


def gate(a):
    f=read(a.out/'frozen.json')
    for n,h in f['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'input changed: {n}')
    for n,h in (f['sources']|f['previous']).items():
        if sha256(ROOT/n)!=h:raise ValueError(f'source changed: {n}')
    return f,{r['id']:r for r in read(a.out/'dataset.json')}


def factors(cache,other,layer,head,dim,groups,mode,positions):
    a=other[layer]['attention'][head] if mode in ('exchange','routing') else cache[layer]['attention'][head]
    values=other[layer]['values'] if mode in ('exchange','values') else cache[layer]['values']
    v=values[:,head//groups,:]
    if mode=='source_values':
        v=v.clone()
        v[positions]=other[layer]['values'][positions,head//groups,:]
    return a @ v


def collect(a):
    import torch
    import transformers
    frozen,rows=gate(a);checkpoint=Path(frozen['checkpoint'])
    for n,h in frozen['checkpoint_sha256'].items():
        if sha256(checkpoint/n)!=h:raise ValueError('checkpoint changed')
    torch.set_num_threads(2)
    model=transformers.AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float32,
        attn_implementation='eager',local_files_only=True).eval()
    dim=model.config.hidden_size//model.config.num_attention_heads
    groups=model.config.num_attention_heads//model.config.num_key_value_heads
    members=read(a.out/'members.json')
    sets={s:[(int(l),int(h)) for name,l,h in members if name==s] for s,l,h in members}
    all_heads=sorted({(l,h) for heads in sets.values() for l,h in heads})
    positions={i:p['positions'] for p in read(a.out/'pairs.json') for i in (p['a'],p['b'])}
    products={};logits_by_id={};reconstruction=[]
    with torch.no_grad():
        for i,row in sorted(rows.items()):
            cache={l:{} for l in {l for l,h in all_heads}};hooks=[]
            for l in cache:
                attn=model.model.layers[l].self_attn
                def values(_module,args,output,l=l):
                    cache[l]['values']=output[0].detach().clone().reshape(len(row['ctx']),model.config.num_key_value_heads,dim)
                def product(_module,args,l=l):cache[l]['product']=args[0][0,-1].detach().clone()
                def attention(_module,args,output,l=l):cache[l]['attention']=output[1][0,:,-1,:].detach().clone()
                hooks.extend((attn.v_proj.register_forward_hook(values),attn.o_proj.register_forward_pre_hook(product),
                    attn.register_forward_hook(attention)))
            try:logits_by_id[i]=model(torch.tensor([row['ctx']]),use_cache=False).logits[0,-1].detach().clone()
            finally:
                for hook in hooks:hook.remove()
            products[i]=cache
            for l,h in all_heads:
                observed=cache[l]['product'][h*dim:(h+1)*dim]
                reconstructed=factors(cache,cache,l,h,dim,groups,'identity',positions[i])
                reconstruction.append([i,l,h,float((observed-reconstructed).abs().max())])
        torch.save(products,a.out/'captured-factors.pt')
        write(a.out/'reconstruction.json',reconstruction)
        write(a.out/'upstream-build.json',{'torch':torch.__version__,'transformers':transformers.__version__,
            'head_dim':dim,'num_key_value_groups':groups,'precision':'float32','attention':'eager','threads':2})
        hist=read(a.out/'historical.json');path=a.out/'observations.jsonl';cached=load(path)
        requests=[(int(i),s,m,int(d)) for i,s,m,d in read(a.out/'plan.json')['relations']['request']]
        if set(cached)-set(requests):raise ValueError('unplanned observation')
        with path.open('a') as stream:
            for j,(i,name,mode,d) in enumerate(requests):
                if (i,name,mode,d) in cached:continue
                started=time.monotonic();hooks=[]
                if mode=='baseline':logits=logits_by_id[i]
                else:
                    patches={(l,h):factors(products[i],products[d],l,h,dim,groups,mode,positions[i]) for l,h in sets[name]}
                    for l in sorted({l for l,h in patches}):
                        heads=tuple(h for layer,h in patches if layer==l)
                        def patch(_module,args,l=l,heads=heads):
                            x=args[0].clone()
                            for h in heads:x[0,-1,h*dim:(h+1)*dim]=patches[l,h]
                            return (x,)+args[1:]
                        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(patch))
                    try:logits=model(torch.tensor([rows[i]['ctx']]),use_cache=False).logits[0,-1]
                    finally:
                        for hook in hooks:hook.remove()
                record={'i':i,'set':name,'mode':mode,'donor':d,'next':int(logits.argmax()),
                    'ref_logit':float(logits[hist[str(i)]['next']]),'gold_logit':float(logits[rows[i]['expected']]),
                    'donor_logit':float(logits[hist[str(d)]['next']]),'seconds':time.monotonic()-started}
                stream.write(json.dumps(record)+'\n');stream.flush()
                if (j+1)%36==0:print(j+1,'/',len(requests),flush=True)


def evaluate(a):
    _,rows=gate(a);hist=read(a.out/'historical.json');plan=read(a.out/'plan.json')['relations'];obs=load(a.out/'observations.jsonl')
    previous=load(PREVIOUS/'observations.jsonl');members=read(a.out/'members.json')
    x=query(a.out,'summary',a.out/'base-checker.dl',{
        'request':''.join(f'{i}\t{s}\t{m}\t{d}\n' for i,s,m,d in plan['request']),
        'donor':''.join(f'{i}\t{d}\n' for i,d in read(a.out/'donors.json')),
        'historical':''.join(f"{i}\t{hist[str(i)]['next']}\t{r['expected']}\t{hist[str(i)]['next_logit']:.17g}\t{dict(hist[str(i)]['logits'])[r['expected']]:.17g}\n" for i,r in rows.items()),
        'result':''.join(f"{r['i']}\t{r['set']}\t{r['mode']}\t{r['donor']}\t{r['next']}\t{r['ref_logit']:.17g}\t{r['gold_logit']:.17g}\t{r['donor_logit']:.17g}\n" for r in obs.values()),
        'previous':''.join(f"{r['i']}\t{r['set']}\t{r['mode']}\t{r['donor']}\t{r['next']}\t{r['ref_logit']:.17g}\t{r['gold_logit']:.17g}\n" for r in previous.values() if r['mode']!='zero'),
        'delta':''.join(f'{i}\t{s}\t{x:.17g}\n' for i,s,x in read(PREVIOUS/'activations.json')['delta']),
        'expected_head':''.join(f'{i}\t{l}\t{h}\n' for i in rows for l,h in sorted({(int(l),int(h)) for s,l,h in members})),
        'reconstruction':''.join(f'{i}\t{l}\t{h}\t{e:.17g}\n' for i,l,h,e in read(a.out/'reconstruction.json'))})
    for name in ('plan','summary'):
        before=read(a.out/f'{name}.json');after=replay(Path(before['evidence']))
        if before['relations']!=after['relations'] or before['certified']!=after['certified']:raise ValueError('replay changed')
    write(a.out/'replay-audit.json',{'queries':2,'unchanged':True})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('instrument:',x['certified'],x.get('error'))
    for key in ('mode_counts','mode_transfer','mode_target','parity_bad','repeated_bad','reconstruction_bad','invalid'):
        print(key,x.get('relations',{}).get(key))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','collect','evaluate'))
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();a.out=a.out.resolve();globals()[a.phase](a)


if __name__=='__main__':main()
