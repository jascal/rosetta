"""Actual4B full-attention instrument gate, planned and checked in Datalog."""
import argparse,hashlib
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_copy_inventory import ROOT,MODEL,OUT as INVENTORY,gate as checkpoint_gate,read,write
from run_qwen35_probe_validation import OUT as PRIOR
OUT=ROOT/'reference/benchmarks/qwen35_attention_validation'
SOURCES=('py/run_qwen35_attention_validation.py','py/qwen35_attention_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','dl/qwen35_attention_plan.dl','dl/qwen35_attention_identity.dl','dl/qwen35_attention_sites.dl','dl/qwen35_checkpoint_parity.dl','py/certificate.py')
def query(out,name,module,facts):
    x=check(ROOT/module,ROOT/module,facts,{},evidence_dir=out/f'certificate-evidence/{name}',provenance={'source':'actual4B attention instrument consistency, not copy extraction/equivalence'})
    write(out/f'{name}.json',x)
    if not x['certified']:raise ValueError(f'{name} failed')
    return x

def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False);ctx=read(PRIOR/'context.json');write(a.out/'context.json',ctx)
    cfg=read(MODEL/'config.json')['text_config'];layers=cfg.get('layer_types',['linear_attention' if (i+1)%4 else 'full_attention' for i in range(cfg['num_hidden_layers'])])
    x=query(a.out,'plan','dl/qwen35_attention_plan.dl',{'shape':f"{cfg['num_hidden_layers']}\t{cfg['num_attention_heads']}\t{cfg['head_dim']}\t{len(ctx['ids'])-1}\n",'full_layer':''.join(f'{i}\n' for i,t in enumerate(layers) if t=='full_attention')})
    write(a.out/'protocol.json',{'scope':'attention instrument only, one previous five-token smoke context','forwards':7,
        'sites':'heads3/0 and31/15, last-query attention and post-gate vector; identity512 scalars; zero256 scalars last31/15 and source-position0 at3/0; independent StreamReference controls',
        'precision':'stored BF16 widened FP32, two CPU threads','parity':'full-vocabulary .002+.002abs tolerance and exact argmax; identity/cleanup hashes and observations exact; all sites and attention row masses checked',
        'prior':{'path':str((PRIOR/'closed.json').relative_to(ROOT)),'sha256':sha256(PRIOR/'closed.json')},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen seven actual4B forwards',flush=True)
def checked(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('input changed')
    for n,h in p['sources'].items():
        if sha256(ROOT/n)!=h:raise ValueError('source changed')
    if sha256(ROOT/p['prior']['path'])!=p['prior']['sha256']:raise ValueError('prior changed')
    return read(a.out/'context.json')['ids'],read(a.out/'plan.json')['relations']
def independent(model,ids,patches):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    original=Qwen3_5DecoderLayer.forward;applied=[];width=model.config.head_dim
    def forward(module,*args,**kwargs):
        layer=(module.linear_attn if module.layer_type=='linear_attention' else module.self_attn).layer_idx
        values=[(p,h,d,v) for l,p,h,d,v in patches if l==layer]
        if not values:return original(module,*args,**kwargs)
        def substitute(_m,inputs):
            x=inputs[0].clone()
            for p,h,d,v in values:
                before=float(x[0,p,h*width+d]);x[0,p,h*width+d]=v
                applied.append({'layer':layer,'position':p,'head':h,'dimension':d,'before':before,'after':float(x[0,p,h*width+d])})
            return (x,)+inputs[1:]
        hook=module.self_attn.o_proj.register_forward_pre_hook(substitute)
        try:return original(module,*args,**kwargs)
        finally:hook.remove()
    Qwen3_5DecoderLayer.forward=forward
    try:return {'logits':model.logits(ids),'vectors':[],'attention':[],'applied':applied}
    finally:Qwen3_5DecoderLayer.forward=original

def run(a):
    ids,plan=checked(a);checkpoint_gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_attention_reference import AttentionReference
    from qwen35_stream_reference import StreamReference
    torch.set_num_threads(2);oracle=AttentionReference(MODEL,vocab_chunk=1024);direct=StreamReference(MODEL,vocab_chunk=1024)
    heads=[tuple(map(int,r)) for r in plan['head']];requests={}
    for arm,order in sorted(plan['run_arm'],key=lambda r:int(r[1])):
        if arm=='identity':
            base=torch.load(a.out/'arm-baseline.pt',weights_only=True)
            identity=query(a.out,'identity','dl/qwen35_attention_identity.dl',{
                'identity_site':''.join('\t'.join(r)+'\n' for r in plan['identity_site']),
                'vector':''.join(f"{r['layer']}\t{r['position']}\t{r['head']}\t{r['dimension']}\t{r['value']:.17g}\n" for r in base['vectors'])})
        rows=identity['relations']['patch'] if arm=='identity' else [r[1:] for r in plan['zero'] if r[0]==arm]
        patches=[(int(l),int(p),int(h),int(d),float(v)) for l,p,h,d,v in rows];requests[arm]=patches
        path=a.out/f'arm-{arm}.pt'
        if path.exists():continue
        data=independent(direct,ids,patches) if arm.startswith('direct_') else oracle.probe(ids,heads,patches)
        tmp=path.with_suffix('.pending');torch.save(data,tmp);tmp.replace(path);print('completed',arm,flush=True)
    evaluate(a,requests)
def evaluate(a,requests):
    import torch
    ids,plan=checked(a);width=read(MODEL/'config.json')['text_config']['head_dim']
    data={arm:torch.load(a.out/f'arm-{arm}.pt',weights_only=True) for arm,order in plan['run_arm']}
    data['prior']=torch.load(PRIOR/'arm-baseline.pt',weights_only=True)
    comparisons=[('baseline','identity'),('baseline','cleanup'),('zero_last','direct_last'),('zero_source','direct_source'),('baseline','prior')]
    parity={'domain':''.join(f'{i}\n' for i in range(len(comparisons))),'vocab':f"{len(data['baseline']['logits'])}\n",'ordinary':'','streamed':'','reset_pair':'','logits_hash':''}
    for i,(left,right) in enumerate(comparisons):
        for rel,backend,key in (('ordinary','ordinary',left),('streamed','streamed',right)):
            scores=data[key]['logits'];parity[rel]+=''.join(f'{i}\t{t}\t{v:.17g}\n' for t,v in enumerate(scores.tolist()))
            parity['logits_hash']+=f'{i}\t{backend}\t{hashlib.sha256(scores.numpy().tobytes()).hexdigest()}\n'
    query(a.out,'parity','dl/qwen35_checkpoint_parity.dl',parity)
    facts={'shape':f'{width}\t{len(ids)-1}\n','arm':''.join(f'{arm}\n' for arm,order in plan['run_arm']),
        'observe':''.join('\t'.join(r)+'\n' for r in plan['observe']),'vector':'','attention':'','request':'','applied':'','hash':''}
    for arm,order in plan['run_arm']:
        d=data[arm];facts['hash']+=f"{arm}\t{hashlib.sha256(d['logits'].numpy().tobytes()).hexdigest()}\n"
        facts['vector']+=''.join(f"{arm}\t{r['layer']}\t{r['position']}\t{r['head']}\t{r['dimension']}\t{r['value']:.17g}\n" for r in d['vectors'])
        facts['attention']+=''.join(f"{arm}\t{r['layer']}\t{r['position']}\t{r['head']}\t{r['source']}\t{r['value']:.17g}\n" for r in d['attention'])
        facts['request']+=''.join(f'{arm}\t{l}\t{p}\t{h}\t{n}\t{v:.17g}\n' for l,p,h,n,v in requests[arm])
        facts['applied']+=''.join(f"{arm}\t{r['layer']}\t{r['position']}\t{r['head']}\t{r['dimension']}\t{r['before']:.17g}\t{r['after']:.17g}\n" for r in d['applied'])
    result=query(a.out,'sites','dl/qwen35_attention_sites.dl',facts)
    for name in ('plan','identity','parity','sites'):
        before=read(a.out/f'{name}.json');after=replay(Path(before['evidence']))
        if not after['certified'] or after['relations']!=before['relations']:raise ValueError('replay differs')
    write(a.out/'replay-audit.json',{'queries':4,'unchanged':True})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('instrument passed; dose',result['relations']['dose'],flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();{'inputs':inputs,'run':run}[a.phase](a)
if __name__=='__main__':main()
