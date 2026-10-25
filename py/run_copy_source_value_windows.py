"""Test fixed source-window values; Datalog fixes keys and verdicts."""
import argparse
import json
from pathlib import Path
import shutil
import time
from certificate import check,replay,sha256
from run_copy_head_factorization import ROOT,OUT as PREVIOUS,factors,read,write,load

OUT=ROOT/'reference/benchmarks/qwen25_copy_source_value_windows'
SOURCES=('py/run_copy_source_value_windows.py','py/run_copy_head_factorization.py','py/run_copy_head_exchange.py',
    'py/run_copy_mechanism.py','py/certificate.py','dl/copy_head_exchange_summary.dl',
    'dl/copy_source_value_windows_plan.dl','dl/copy_source_value_windows_summary.dl','docs/copy-source-value-windows-plan.md')


def query(out,name,circuit,facts):
    result=check(ROOT/f'dl/copy_source_value_windows_{name}.dl',circuit,facts,{},
        evidence_dir=out/f'certificate-evidence/{name}',provenance={'source':'historical source-window values; instrument consistency only'})
    write(out/f'{name}.json',result)
    return result


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    for n in ('dataset.json','pairs.json','historical.json','members.json','donors.json','base-checker.dl'):
        shutil.copyfile(PREVIOUS/n,a.out/n)
    rows=read(a.out/'dataset.json');members=read(a.out/'members.json');pairs=read(a.out/'pairs.json')
    module=ROOT/'dl/copy_source_value_windows_plan.dl'
    result=query(a.out,'plan',module,{
        'donor':''.join(f'{i}\t{d}\n' for i,d in read(a.out/'donors.json')),
        'member':''.join(f'{s}\t{l}\t{h}\n' for s,l,h in members),
        'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'edit':''.join(f'{i}\t{p}\n' for row in pairs for i in (row['a'],row['b']) for p in row['positions'])})
    if not result['certified']:raise ValueError(result.get('error','invalid plan'))
    previous=read(PREVIOUS/'frozen.json')
    write(a.out/'frozen.json',{'scope':'historical token/bigram source-window values; no final certificate',
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'previous':{str(p.relative_to(ROOT)):sha256(p) for p in
            (PREVIOUS/'closed.json',PREVIOUS/'frozen.json',PREVIOUS/'observations.jsonl',PREVIOUS/'captured-factors.pt',PREVIOUS/'upstream-build.json')},
        'checkpoint':previous['checkpoint'],'checkpoint_sha256':previous['checkpoint_sha256']})
    print('frozen:',len(result['relations']['request']),'full-context requests;',len(result['relations']['feature']),'features',flush=True)


def gate(a):
    f=read(a.out/'frozen.json')
    for n,h in f['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'frozen input changed: {n}')
    for n,h in (f['sources']|f['previous']).items():
        if sha256(ROOT/n)!=h:raise ValueError(f'frozen source changed: {n}')
    return f,{r['id']:r for r in read(a.out/'dataset.json')},read(a.out/'plan.json')['relations']


def lexical_product(cache,feature,sites,layer,head,groups,size,source_id):
    values=cache[layer]['values'][:,head//groups,:].clone()
    for p in sites[source_id]:
        values[p]=feature[source_id,p,size][layer][head//groups]
    return cache[layer]['attention'][head] @ values


def collect(a):
    import torch
    import transformers
    frozen,rows,plan=gate(a);checkpoint=Path(frozen['checkpoint'])
    for n,h in frozen['checkpoint_sha256'].items():
        if sha256(checkpoint/n)!=h:raise ValueError('checkpoint changed')
    torch.set_num_threads(2)
    model=transformers.AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float32,
        attn_implementation='eager',local_files_only=True).eval()
    build=read(PREVIOUS/'upstream-build.json');dim=build['head_dim'];groups=build['num_key_value_groups']
    cache=torch.load(PREVIOUS/'captured-factors.pt',map_location='cpu',weights_only=True)
    members=read(a.out/'members.json')
    sets={s:[(int(l),int(h)) for name,l,h in members if name==s] for s,l,h in members}
    sites={i:[] for i in rows}
    for i,p in plan['site']:sites[int(i)].append(int(p))
    windows={}
    for i,p,k,j,t in plan['window_tok']:
        windows.setdefault((int(i),int(p),int(k)),[]).append((int(j),int(t)))
    window_modes={m:int(k) for m,k,kind in plan['window_mode']}
    feature={};baselines={}
    with torch.no_grad():
        for i,p,k in plan['feature']:
            key=tuple(map(int,(i,p,k)));values={};hooks=[]
            tokens=[t for j,t in sorted(windows[key])]
            for l in (16,21):
                def capture(_module,args,output,l=l):
                    values[l]=output[0,-1].detach().clone().reshape(model.config.num_key_value_heads,dim)
                hooks.append(model.model.layers[l].self_attn.v_proj.register_forward_hook(capture))
            try:model(torch.tensor([tokens]),use_cache=False)
            finally:
                for hook in hooks:hook.remove()
            feature[key]=values
        torch.save(feature,a.out/'lexical-values.pt')
        write(a.out/'upstream-build.json',{'torch':torch.__version__,'transformers':transformers.__version__,
            'head_dim':dim,'num_key_value_groups':groups,'precision':'float32','attention':'eager','threads':2})
        hist=read(a.out/'historical.json');path=a.out/'observations.jsonl';cached=load(path)
        requests=[(int(i),s,m,int(d)) for i,s,m,d in plan['request']]
        if set(cached)-set(requests):raise ValueError('unplanned observation')
        with path.open('a') as stream:
            for j,(i,name,mode,d) in enumerate(requests):
                if (i,name,mode,d) in cached:continue
                started=time.monotonic();hooks=[]
                if mode=='baseline':
                    logits=model(torch.tensor([rows[i]['ctx']]),use_cache=False).logits[0,-1]
                else:
                    patches={}
                    for l,h in sets[name]:
                        if mode in ('identity','source_values'):
                            patches[l,h]=factors(cache[i],cache[d],l,h,dim,groups,mode,sites[i])
                        else:
                            patches[l,h]=lexical_product(cache[i],feature,sites,l,h,groups,window_modes[mode],d)
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
                if (j+1)%48==0:print(j+1,'/',len(requests),flush=True)


def evaluate(a):
    _,rows,plan=gate(a);hist=read(a.out/'historical.json');obs=load(a.out/'observations.jsonl')
    previous=load(PREVIOUS/'observations.jsonl')
    x=query(a.out,'summary',a.out/'base-checker.dl',{
        'request':''.join(f'{i}\t{s}\t{m}\t{d}\n' for i,s,m,d in plan['request']),
        'donor':''.join(f'{i}\t{d}\n' for i,d in read(a.out/'donors.json')),
        'historical':''.join(f"{i}\t{hist[str(i)]['next']}\t{r['expected']}\t{hist[str(i)]['next_logit']:.17g}\t{dict(hist[str(i)]['logits'])[r['expected']]:.17g}\n" for i,r in rows.items()),
        'result':''.join(f"{r['i']}\t{r['set']}\t{r['mode']}\t{r['donor']}\t{r['next']}\t{r['ref_logit']:.17g}\t{r['gold_logit']:.17g}\t{r['donor_logit']:.17g}\n" for r in obs.values()),
        'previous':''.join(f"{r['i']}\t{r['set']}\t{r['mode']}\t{r['donor']}\t{r['next']}\t{r['ref_logit']:.17g}\t{r['gold_logit']:.17g}\n" for r in previous.values() if r['mode'] in ('baseline','identity','source_values')),
        'delta':''})
    for name in ('plan','summary'):
        before=read(a.out/f'{name}.json');after=replay(Path(before['evidence']))
        if before['relations']!=after['relations'] or before['certified']!=after['certified']:raise ValueError('replay changed')
    write(a.out/'replay-audit.json',{'queries':2,'unchanged':True})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('instrument:',x['certified'],x.get('error'))
    for key in ('mode_counts','mode_transfer','mode_target','parity_bad','repeated_bad','locality_invalid','invalid'):
        print(key,x.get('relations',{}).get(key))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','collect','evaluate'))
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();a.out=a.out.resolve();globals()[a.phase](a)


if __name__=='__main__':main()
