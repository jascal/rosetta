"""Freeze all-head observations on the six existing source-choice contrasts."""
import argparse,hashlib
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_attention_validation import ROOT,OUT as VALIDATION,read,write
from run_qwen35_routing_mlp import OUT as PRIOR
from run_qwen35_copy_inventory import OUT as INVENTORY,MODEL,gate
OUT=ROOT/'reference/benchmarks/qwen35_attention_profile'
SOURCES=('py/run_qwen35_attention_profile.py','py/qwen35_attention_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','dl/qwen35_attention_sites.dl','dl/qwen35_gate_cross_reproduction.dl','dl/qwen35_attention_profile.dl','dl/qwen35_attention_profile_plan.dl','dl/copy_longest_runtime.dl','py/certificate.py')
def inputs(a):
    report=read(ROOT/'reference/benchmarks/qwen35_attention_validation_replay.json')
    if not report['unchanged'] or report['closed_sha256']!=sha256(VALIDATION/'closed.json'):raise ValueError('instrument gate invalid')
    a.out.mkdir(parents=True,exist_ok=False);contexts=read(PRIOR/'contexts.json');write(a.out/'contexts.json',contexts)
    cfg=read(MODEL/'config.json')['text_config'];layers=cfg.get('layer_types',['linear_attention' if (i+1)%4 else 'full_attention' for i in range(cfg['num_hidden_layers'])])
    m=ROOT/'dl/qwen35_attention_profile_plan.dl'
    plan=check(m,m,{'heads':f"{cfg['num_attention_heads']}\n",'full_layer':''.join(f'{l}\n' for l,t in enumerate(layers) if t=='full_attention')},{},evidence_dir=a.out/'certificate-evidence/plan',provenance={'source':'complete architecture-defined full-attention head inventory'})
    write(a.out/'plan.json',plan)
    if not plan['certified']:raise ValueError('invalid head inventory')
    heads=[tuple(map(int,r)) for r in plan['relations']['head']]
    write(a.out/'heads.json',heads);(a.out/'input-core.dl').write_bytes((INVENTORY/'input-core.dl').read_bytes())
    write(a.out/'protocol.json',{'scope':'six known development contrasts; all full-attention heads; no causal selection or final claim','forwards':len(contexts),'heads':len(heads),
        'ranking':'sum across old/new pairs of change in (attention at long-source successor minus competing-source successor); report top4 with layer/head ties; ranks observational only',
        'priors':{str((p/'closed.json').relative_to(ROOT)):sha256(p/'closed.json') for p in (VALIDATION,PRIOR)},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen',len(contexts),'contexts,',len(heads),'heads',flush=True)
def checked(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('input changed')
    for n,h in (p['sources']|p['priors']).items():
        if sha256(ROOT/n)!=h:raise ValueError('source/prior changed')
def run(a):
    checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_attention_reference import AttentionReference
    torch.set_num_threads(2);model=AttentionReference(MODEL,vocab_chunk=1024);heads=[tuple(h) for h in read(a.out/'heads.json')]
    contexts=read(a.out/'contexts.json')
    for r in contexts:
        path=a.out/f"case-{r['id']}.pt"
        if path.exists():continue
        d=model.probe(r['ctx'],heads,());tmp=path.with_suffix('.pending');torch.save(d,tmp);tmp.replace(path);print('captured case',r['id'],flush=True)
    evaluate(a)
def evaluate(a):
    import torch
    checked(a);contexts=read(a.out/'contexts.json');heads=read(a.out/'heads.json');width=read(MODEL/'config.json')['text_config']['head_dim']
    sf={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in contexts for p,t in enumerate(r['ctx'])),
        'context':''.join(f"{r['id']}\t{r['contrast']}\t{r['side']}\n" for r in contexts),'head':''.join(f'{l}\t{h}\n' for l,h in heads),'attention':''}
    reproduction={'expected':'','actual':''}
    for r in contexts:
        i=r['id'];d=torch.load(a.out/f'case-{i}.pt',weights_only=True);old=torch.load(PRIOR/f'arm-{100*i+99}.pt',weights_only=True)
        digest=hashlib.sha256(d['logits'].numpy().tobytes()).hexdigest();reproduction['actual']+=f'{i}\t{digest}\n';reproduction['expected']+=f"{i}\t{hashlib.sha256(old['logits'].numpy().tobytes()).hexdigest()}\n"
        facts={'shape':f"{width}\t{len(r['ctx'])-1}\n",'arm':'capture\n','observe':''.join(f'capture\t{l}\t{h}\n' for l,h in heads),
            'hash':f'capture\t{digest}\n','request':'','applied':'',
            'vector':''.join(f"capture\t{o['layer']}\t{o['position']}\t{o['head']}\t{o['dimension']}\t{o['value']:.17g}\n" for o in d['vectors']),
            'attention':''.join(f"capture\t{o['layer']}\t{o['position']}\t{o['head']}\t{o['source']}\t{o['value']:.17g}\n" for o in d['attention'])}
        m=ROOT/'dl/qwen35_attention_sites.dl';x=check(m,m,facts,{},evidence_dir=a.out/f'certificate-evidence/sites-{i}',provenance={'source':'complete all-head capture consistency'})
        write(a.out/f'sites-{i}.json',x)
        if not x['certified']:raise ValueError('capture incomplete')
        sf['attention']+=''.join(f"{i}\t{o['layer']}\t{o['head']}\t{o['source']}\t{o['value']:.17g}\n" for o in d['attention'])
    m=ROOT/'dl/qwen35_gate_cross_reproduction.dl';x=check(m,m,reproduction,{},evidence_dir=a.out/'certificate-evidence/reproduction',provenance={'source':'all-head no-op capture must reproduce prior full-score hashes'})
    write(a.out/'reproduction.json',x)
    if not x['certified']:raise ValueError('capture changes logits')
    m=ROOT/'dl/qwen35_attention_profile.dl';x=check(m,a.out/'input-core.dl',sf,{},evidence_dir=a.out/'certificate-evidence/profile',provenance={'source':'observational source-position attention shifts; not causal head admission'})
    write(a.out/'profile.json',x)
    for before in [read(a.out/'plan.json'),read(a.out/'reproduction.json'),x]+[read(p) for p in a.out.glob('sites-*.json')]:
        after=replay(Path(before['evidence']))
        if not after['certified'] or before['relations']!=after['relations']:raise ValueError('profile replay differs')
    write(a.out/'replay-audit.json',{'queries':len(contexts)+3,'unchanged':True})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('leading observational heads',x['relations']['leading'],flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();{'inputs':inputs,'run':run}[a.phase](a)
if __name__=='__main__':main()
