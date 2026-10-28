"""Frozen actual-checkpoint MLP validation; all plans and verdicts are Datalog."""
import argparse
import hashlib
from pathlib import Path
import resource
import time
from certificate import check,replay,sha256
from run_copy_mechanism import ROOT,read,write
from run_qwen35_checkpoint_smoke import MODEL,REVISION,OUT as SMOKE

OUT=ROOT/'reference/benchmarks/qwen35_4b_probe_validation'
SOURCES=('py/run_qwen35_probe_validation.py','py/qwen35_mlp_reference.py','py/qwen35_lazy_reference.py',
    'py/qwen35_stream_reference.py','py/run_qwen35_checkpoint_smoke.py','py/certificate.py','py/run_copy_mechanism.py',
    'dl/qwen35_probe_plan.dl','dl/qwen35_probe_identity.dl','dl/qwen35_probe_summary.dl','docs/qwen35-mlp-probe-plan.md')


def query(out,name,facts):
    module=ROOT/f'dl/qwen35_probe_{name}.dl'
    result=check(module,module,facts,{},evidence_dir=out/f'certificate-evidence/{name}',
        provenance={'source':'actual 4B MLP instrument consistency; no mechanism/equivalence claim'})
    write(out/f'{name}.json',result)
    return result


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    context=read(SMOKE/'contexts.json')[1];write(a.out/'context.json',context)
    cfg=read(MODEL/'config.json')['text_config']
    result=query(a.out,'plan',{'shape':f"{cfg['num_hidden_layers']}\t{cfg['intermediate_size']}\t{len(context['ids'])-1}\n"})
    if not result['certified']:raise ValueError(result.get('error','invalid plan'))
    write(a.out/'frozen.json',{'scope':'MLP instrument validation on one known smoke context; not a copy study',
        'model':'Qwen/Qwen3.5-4B','revision':REVISION,
        'checkpoint_files':read(SMOKE/'protocol.json')['checkpoint_files'],
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'prior':{str(p.relative_to(ROOT)):sha256(p) for p in (SMOKE/'parity.json',SMOKE/'closed.json',SMOKE/'protocol.json')},
        'threads':2,'precision':'stored checkpoint weights widened to FP32','cache':False,
        'tolerance':{'absolute':0.002,'relative':0.002}})
    print('frozen 7 forwards, 8 last-position units, independent source/answer zero controls',flush=True)


def gate(a):
    f=read(a.out/'frozen.json')
    for n,h in f['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'frozen input changed: {n}')
    for n,h in (f['sources']|f['prior']).items():
        if sha256(ROOT/n)!=h:raise ValueError(f'frozen source changed: {n}')
    for n,h in f['checkpoint_files'].items():
        if sha256(MODEL/n)!=h:raise ValueError(f'checkpoint changed: {n}')
    return read(a.out/'context.json'),read(a.out/'plan.json')['relations']


def independent_zero(model,ids,patches):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    original=Qwen3_5DecoderLayer.forward;applied=[]
    def forward(module,*args,**kwargs):
        layer=(module.linear_attn if module.layer_type=='linear_attention' else module.self_attn).layer_idx
        values=[(p,n,v) for l,p,n,v in patches if l==layer]
        if not values:return original(module,*args,**kwargs)
        def substitute(_module,inputs):
            products=inputs[0].clone()
            for p,n,v in values:
                before=float(products[0,p,n]);products[0,p,n]=v
                applied.append({'layer':layer,'position':p,'neuron':n,'before':before,'after':float(products[0,p,n])})
            return (products,)+inputs[1:]
        hook=module.mlp.down_proj.register_forward_pre_hook(substitute)
        try:return original(module,*args,**kwargs)
        finally:hook.remove()
    Qwen3_5DecoderLayer.forward=forward
    try:return {'logits':model.logits(ids),'observations':[],'applied':applied}
    finally:Qwen3_5DecoderLayer.forward=original


def obs_facts(arm,observations):
    return ''.join(f"{arm}\t{r['layer']}\t{r['position']}\t{r['neuron']}\t{r['gate']:.17g}\t{r['value']:.17g}\t{r['product']:.17g}\n" for r in observations)


def run(a):
    import torch
    from qwen35_mlp_reference import MlpReference
    from qwen35_stream_reference import StreamReference
    context,plan=gate(a);ids=context['ids'];torch.set_num_threads(2)
    oracle=MlpReference(MODEL,vocab_chunk=1024);direct=StreamReference(MODEL,vocab_chunk=1024)
    units=[tuple(map(int,r)) for r in plan['unit']];identity=None;patch_requests={};timings=[]
    for arm,order in sorted(plan['run_arm'],key=lambda r:int(r[1])):
        path=a.out/f'arm-{arm}.pt'
        if arm=='identity':
            baseline=torch.load(a.out/'arm-baseline.pt',weights_only=True)
            identity=query(a.out,'identity',{'identity_site':''.join(f'{l}\t{p}\t{n}\n' for l,p,n in plan['identity_site']),
                'observation':obs_facts('baseline',baseline['observations'])})
            if not identity['certified']:raise ValueError(identity.get('error','incomplete identity payload'))
        rows=(identity['relations']['patch'] if arm=='identity' else [r[1:] for r in plan['zero'] if r[0]==arm])
        patches=[(int(l),int(p),int(n),float(v)) for l,p,n,v in rows];patch_requests[arm]=patches
        if path.exists():continue
        started=time.monotonic()
        result=(independent_zero(direct,ids,patches) if arm.startswith('direct_') else oracle.probe(ids,units,patches))
        torch.save(result,path);elapsed=time.monotonic()-started
        timings.append({'arm':arm,'seconds':elapsed});print('completed',arm,f'{elapsed:.2f}s',flush=True)
    facts={'arm':''.join(f'{arm}\n' for arm, in plan['arm']),
        'compare':''.join(f'{a}\t{b}\n' for a,b in plan['compare']),
        'observe':''.join(f'{a}\t{l}\t{p}\t{n}\n' for a,l,p,n in plan['observe']),
        'vocab':f'{oracle.config.vocab_size}\n','score':'','observation':'','patch_request':'','applied':'','logits_hash':''}
    for arm,order in plan['run_arm']:
        result=torch.load(a.out/f'arm-{arm}.pt',weights_only=True);scores=result['logits']
        facts['score']+=''.join(f'{arm}\t{t}\t{x:.17g}\n' for t,x in enumerate(scores.tolist()))
        facts['logits_hash']+=f'{arm}\t{hashlib.sha256(scores.numpy().tobytes()).hexdigest()}\n'
        facts['observation']+=obs_facts(arm,result['observations'])
        facts['patch_request']+=''.join(f'{arm}\t{l}\t{p}\t{n}\t{v:.17g}\n' for l,p,n,v in patch_requests[arm])
        facts['applied']+=''.join(f"{arm}\t{r['layer']}\t{r['position']}\t{r['neuron']}\t{r['before']:.17g}\t{r['after']:.17g}\n" for r in result['applied'])
    prior=Path(read(SMOKE/'parity.json')['evidence'])/'facts'
    prior_scores=[]
    for line in (prior/'ordinary.facts').read_text().splitlines():
        i,t,x=line.split('\t')
        if i=='1':prior_scores.append(f'prior\t{t}\t{x}\n')
    facts['score']+=''.join(prior_scores)
    for line in (prior/'logits_hash.facts').read_text().splitlines():
        i,backend,h=line.split('\t')
        if i=='1' and backend=='ordinary':facts['logits_hash']+=f'prior\t{h}\n'
    result=query(a.out,'summary',facts)
    for name in ('plan','identity','summary'):
        before=read(a.out/f'{name}.json');after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('replay differs')
    write(a.out/'resources.json',{'timings':timings,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    write(a.out/'replay-audit.json',{'queries':3,'unchanged':True})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('instrument:',result['certified'],result.get('error'))
    for n in ('ref','dose','changed_scores','invalid','logit_bad','argmax_bad','observation_bad','hash_bad'):
        print(n,result.get('relations',{}).get(n))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run'))
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();a.out=a.out.resolve();globals()[a.phase](a)


if __name__=='__main__':main()
