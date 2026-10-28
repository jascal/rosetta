"""Whole-sequence head deletion: fixed Datalog plan, upstream/native I/O, Datalog gate."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import time
import urllib.request

from certificate import check, replay, sha256
from run_copy_mechanism import ROOT, server, read, write

OUT=ROOT/'reference/benchmarks/qwen25_copy_heads'
DIAG=ROOT/'reference/benchmarks/qwen25_copy_source_diagnosis'
HIST=ROOT/'reference/benchmarks/qwen25_copy_longest'
SOURCES=('py/run_copy_heads.py','py/run_copy_mechanism.py','py/certificate.py',
         'dl/copy_head_plan.dl','dl/copy_head_summary.dl','docs/copy-head-ablation-plan.md')


def query(out,name,module,facts):
    source=ROOT/'dl'/module
    x=check(source,source,facts,{},evidence_dir=out/'certificate-evidence'/name,
        provenance={'source':'historical head deletion; empirical instrument check, no model-equivalence claim'})
    write(out/f'{name}.json',x)
    return x


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    rows=read(DIAG/'selected.json')
    write(a.out/'dataset.json',rows)
    rng=random.Random(503); rankings=[]
    for layer in (16,21):
        heads=list(range(14)); rng.shuffle(heads)
        rankings.extend((layer,h,k) for k,h in enumerate(heads))
    write(a.out/'rankings.json',rankings)
    x=query(a.out,'plan','copy_head_plan.dl',{
        'row':''.join(f"{r['id']}\n" for r in rows),
        'ranked':''.join(f'{l}\t{h}\t{k}\n' for l,h,k in rankings)})
    if not x['certified']: raise ValueError(x.get('error','invalid plan'))
    print('requests:',len(x['relations']['request']),'controls:',x['relations']['chosen_control'])


def freeze(a):
    if (a.out/'frozen.json').exists() or (a.out/'native.jsonl').exists() or (a.out/'upstream.jsonl').exists():
        raise ValueError('already frozen or observed')
    plan=read(a.out/'plan.json'); actual=replay(Path(plan['evidence']))
    if not actual['certified'] or actual['relations']!=plan['relations']: raise ValueError('plan replay failed')
    if sha256(str(a.bundle)+'.fieldrun.bin')!=read(DIAG/'frozen.json')['pinned_model']['bundle_sha256']['.fieldrun.bin']:
        raise ValueError('different model weights')
    patch=subprocess.run(['git','diff','HEAD'],cwd=a.instrument,capture_output=True,text=True,check=True).stdout
    (a.out/'instrument.patch').write_text(patch)
    write(a.out/'instrument.json',{'checkout':str(a.instrument.resolve()),
        'base':subprocess.run(['git','rev-parse','HEAD'],cwd=a.instrument,capture_output=True,text=True,check=True).stdout.strip(),
        'files':{str(p.relative_to(a.instrument)):sha256(p) for p in
                 (a.instrument/'src/api.rs',a.instrument/'src/model.rs',a.instrument/'src/rope.rs')}})
    checkpoint={p.name:sha256(p) for p in a.checkpoint.iterdir() if p.suffix in ('.json','.safetensors')}
    write(a.out/'frozen.json',{'scope':'133 historical-case discovery interventions; no final-test circuit',
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'historical':{str(p.relative_to(ROOT)):sha256(p) for p in
            (DIAG/'selected.json',DIAG/'frozen.json',DIAG/'closed.json',HIST/'observations.jsonl')},
        'binary_sha256':sha256(a.binary),'bundle_sha256':{
            s:sha256(str(a.bundle)+s) for s in ('.fieldrun.bin','.fieldrun.json')},
        'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':checkpoint,
        'reference':{'precision':'float32','attention':'eager','threads':2},'seed':503})
    print('frozen before upstream and native requests')


def gate(a):
    p=read(a.out/'frozen.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h: raise ValueError(f'changed frozen input: {n}')
    for n,h in (p['sources']|p['historical']).items():
        if sha256(ROOT/n)!=h: raise ValueError(f'changed source: {n}')
    if a.binary and sha256(a.binary)!=p['binary_sha256']: raise ValueError('different executable')
    if a.bundle:
        for s,h in p['bundle_sha256'].items():
            if sha256(str(a.bundle)+s)!=h: raise ValueError('different bundle')
    rows={r['id']:r for r in read(a.out/'dataset.json')}
    relations=read(a.out/'plan.json')['relations']
    members={arm:[] for arm,_ in relations['arm']}
    for arm,l,h in relations['member']: members[arm].append((int(l),int(h)))
    requests=[(int(i),arm) for i,arm in relations['request']]
    return p,rows,members,requests


def load_journal(path):
    cached={}
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line); key=(r['i'],r['arm'])
            if key in cached: raise ValueError('duplicate intervention')
            cached[key]=r
    return cached


def reference(a):
    import torch
    import transformers
    frozen,rows,members,requests=gate(a)
    checkpoint=Path(frozen['checkpoint'])
    for n,h in frozen['checkpoint_sha256'].items():
        if sha256(checkpoint/n)!=h: raise ValueError('checkpoint changed')
    torch.set_num_threads(2)
    model=transformers.AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float32,
        attn_implementation='eager',local_files_only=True).eval()
    write(a.out/'upstream-build.json',{'torch':torch.__version__,'transformers':transformers.__version__,
        'head_dim':model.config.hidden_size//model.config.num_attention_heads,
        'precision':'float32','attention':'eager','threads':2})
    dim=model.config.hidden_size//model.config.num_attention_heads
    historical={r['id']:r['response'] for r in map(json.loads,(HIST/'observations.jsonl').read_text().splitlines())}
    path=a.out/'upstream.jsonl'; cached=load_journal(path)
    if set(cached)-set(requests): raise ValueError('unplanned upstream intervention')
    with path.open('a') as stream,torch.no_grad():
        for j,(i,arm) in enumerate(requests):
            if (i,arm) in cached: continue
            hooks=[]
            by_layer={}
            for l,h in members[arm]: by_layer.setdefault(l,[]).append(h)
            for l,heads in by_layer.items():
                def mask(_module,args,heads=tuple(heads)):
                    x=args[0].clone()
                    for h in heads: x[...,h*dim:(h+1)*dim]=0.0
                    return (x,)+args[1:]
                hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(mask))
            start=time.monotonic()
            try:
                logits=model(torch.tensor([rows[i]['ctx']]),use_cache=False).logits[0,-1]
                next_id=int(logits.argmax())
                record={'i':i,'arm':arm,'next':next_id,'next_logit':float(logits[next_id]),
                    'base_logit':float(logits[historical[i]['next']]),
                    'gold_logit':float(logits[rows[i]['expected']]),'seconds':time.monotonic()-start}
                stream.write(json.dumps(record)+'\n');stream.flush()
            finally:
                for hook in hooks: hook.remove()
            if (j+1)%20==0: print('upstream',j+1,'/',len(requests),flush=True)


def native(a):
    _,rows,members,requests=gate(a)
    historical={r['id']:r['response'] for r in map(json.loads,(HIST/'observations.jsonl').read_text().splitlines())}
    path=a.out/'native.jsonl'; cached=load_journal(path)
    if set(cached)-set(requests): raise ValueError('unplanned native intervention')
    with server(a,'head-native') as port,path.open('a') as stream:
        for j,(i,arm) in enumerate(requests):
            if (i,arm) in cached: continue
            targets=sorted({historical[i]['next'],rows[i]['expected']})
            req=urllib.request.Request(f'http://127.0.0.1:{port}/probe/heads',
                json.dumps({'ids':rows[i]['ctx'],'probe':{'heads':members[arm],'readouts':targets}}).encode(),
                {'Content-Type':'application/json'})
            start=time.monotonic()
            with urllib.request.urlopen(req,timeout=180) as response: x=json.load(response)
            if 'error' in x: raise ValueError(x)
            logits=dict(x['logits'])
            stream.write(json.dumps({'i':i,'arm':arm,'next':x['next'],'next_logit':x['next_logit'],
                'base_logit':logits[historical[i]['next']],'gold_logit':logits[rows[i]['expected']],
                'scope':x['scope'],'response':x,'seconds':time.monotonic()-start})+'\n');stream.flush()
            if (j+1)%20==0: print('native',j+1,'/',len(requests),flush=True)


def evaluate(a):
    _,rows,_,requests=gate(a)
    historical={r['id']:r['response'] for r in map(json.loads,(HIST/'observations.jsonl').read_text().splitlines())}
    native_rows=load_journal(a.out/'native.jsonl'); upstream_rows=load_journal(a.out/'upstream.jsonl')
    facts={'sample':''.join(f"{i}\t{r['diagnostic_class']}\t{r['expected']}\n" for i,r in rows.items()),
        'baseline':''.join(f"{i}\t{historical[i]['next']}\t{historical[i]['next_logit']:.17g}\t{dict(historical[i]['logits'])[r['expected']]:.17g}\n" for i,r in rows.items()),
        'request':''.join(f'{i}\t{arm}\n' for i,arm in requests),
        'native':''.join(f"{r['i']}\t{r['arm']}\t{r['next']}\t{r['base_logit']:.17g}\t{r['gold_logit']:.17g}\t{r['scope']}\n" for r in native_rows.values()),
        'upstream':''.join(f"{r['i']}\t{r['arm']}\t{r['next']}\t{r['base_logit']:.17g}\t{r['gold_logit']:.17g}\n" for r in upstream_rows.values())}
    x=query(a.out,'summary','copy_head_summary.dl',facts)
    print('instrument gate:',x['certified'])
    if not x['certified']: raise ValueError(x.get('error',{'parity_bad':x['relations']['parity_bad'],'empty_bad':x['relations']['empty_bad']}))
    print('counts:',x['relations']['counts'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('inputs','freeze','reference','native','evaluate'))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--binary',type=Path);p.add_argument('--bundle',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--instrument',type=Path,default=Path('/tmp/rosetta-copy-attention'))
    p.add_argument('--port',type=int,default=8397)
    a=p.parse_args();a.out=a.out.resolve()
    if a.phase in ('freeze','native') and not(a.binary and a.bundle): p.error('phase requires --binary and --bundle')
    if a.phase=='freeze' and not a.checkpoint: p.error('freeze needs --checkpoint')
    globals()[a.phase](a)


if __name__=='__main__': main()
