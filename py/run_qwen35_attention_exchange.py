"""Frozen causal donor exchange of profile-selected versus matched random heads."""
import argparse,hashlib,random
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_attention_profile import ROOT,OUT as PRIOR,read,write
from run_qwen35_attention_validation import OUT as VALIDATION
from run_qwen35_copy_inventory import OUT as INVENTORY,MODEL,gate
from qwen35_reference_evidence import extract,replay_archive
OUT=ROOT/'reference/benchmarks/qwen35_attention_exchange'
SOURCES=('py/run_qwen35_attention_exchange.py','dl/qwen35_attention_exchange_payload.dl','dl/qwen35_attention_exchange_summary.dl','py/qwen35_attention_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','py/qwen35_reference_evidence.py','dl/qwen35_reference_argmax.dl','py/certificate.py')
def inputs(a):
    import torch
    for root in (PRIOR,VALIDATION):
        for n,h in read(root/'closed.json')['files'].items():
            if sha256(root/n)!=h:raise ValueError('prior changed')
    for name,root in (('profile',PRIOR),('validation',VALIDATION)):
        r=read(ROOT/f'reference/benchmarks/qwen35_attention_{name}_replay.json')
        if not r['unchanged'] or r['closed_sha256']!=sha256(root/'closed.json'):raise ValueError('prior replay invalid')
    a.out.mkdir(parents=True,exist_ok=False);contexts=read(PRIOR/'contexts.json');write(a.out/'contexts.json',contexts)
    heads=read(PRIOR/'heads.json');rng=random.Random(2300);order=''
    for l in sorted({l for l,h in heads}):
        values=[h for layer,h in heads if layer==l];rng.shuffle(values)
        order+=''.join(f'{l}\t{h}\t{r}\n' for r,h in enumerate(values))
    vectors='';hashes={}
    for r in contexts:
        data=torch.load(PRIOR/f"case-{r['id']}.pt",weights_only=True)
        vectors+=''.join(f"{r['id']}\t{o['layer']}\t{o['head']}\t{o['dimension']}\t{o['value']:.17g}\n" for o in data['vectors'])
        hashes[str(r['id'])]=hashlib.sha256(data['logits'].numpy().tobytes()).hexdigest()
    write(a.out/'baseline-hashes.json',hashes)
    width=read(MODEL/'config.json')['text_config']['head_dim'];m=ROOT/'dl/qwen35_attention_exchange_payload.dl'
    x=check(m,m,{'shape':f'{width}\n','context':''.join(f"{r['id']}\t{r['contrast']}\t{r['side']}\t{len(r['ctx'])-1}\n" for r in contexts),
        'leading':''.join('\t'.join(r)+'\n' for r in read(PRIOR/'profile.json')['relations']['leading']),
        'random_order':order,'vector':vectors},{},evidence_dir=a.out/'certificate-evidence/payload',provenance={'source':'leading heads positive in all three pairs; matched same-layer random heads seed2300; all directions and exact identities'})
    write(a.out/'payload.json',x)
    if not x['certified'] or replay(Path(x['evidence']))['relations']!=x['relations']:raise ValueError('payload failed')
    write(a.out/'protocol.json',{'scope':'causal head-vector transfer on six known contrasts, not final circuit confirmation','forwards':len(x['relations']['run']),
        'selection':'among frozen leading4 heads, require positive source shifts in all3 pairs; two heads per selected layer; same-count layer-matched random controls exclude selected; seed2300',
        'arms':'each selected/random head separately, selected/random joint, selected joint identity; last-position post-gate vectors before output projection; both directions',
        'priors':{str((p/'closed.json').relative_to(ROOT)):sha256(p/'closed.json') for p in (PRIOR,VALIDATION)},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen',len(x['relations']['run']),'forwards selected',x['relations']['selected'],'controls',x['relations']['control'],flush=True)
def checked(a):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('input changed')
    for n,h in (p['sources']|p['priors']).items():
        if sha256(ROOT/n)!=h:raise ValueError('source/prior changed')
    return read(a.out/'payload.json')['relations']
def run(a):
    plan=checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_attention_reference import AttentionReference
    torch.set_num_threads(2);model=AttentionReference(MODEL,vocab_chunk=1024);contexts={r['id']:r for r in read(a.out/'contexts.json')}
    for j,(aa,tt,dd,mode) in enumerate(sorted(plan['run'],key=lambda r:int(r[0]))):
        arm=int(aa);path=a.out/f'arm-{arm}.pt'
        if path.exists():continue
        patches=[(int(l),int(p),int(h),int(d),float(v)) for a,l,p,h,d,v in plan['patch'] if a==aa]
        data=model.probe(contexts[int(tt)]['ctx'],(),patches);tmp=path.with_suffix('.pending');torch.save(data,tmp);tmp.replace(path)
        print('completed',j+1,'/',len(plan['run']),arm,mode,flush=True)
def evaluate(a):
    plan=checked(a)
    import torch
    facts={'run':''.join('\t'.join(r)+'\n' for r in plan['run']),
        'expected':''.join(f"{r['id']}\t{r['expected']}\n" for r in read(a.out/'contexts.json')),
        'baseline_hash':''.join(f'{i}\t{h}\n' for i,h in read(a.out/'baseline-hashes.json').items()),
        'npatch':''.join('\t'.join(r)+'\n' for r in plan['npatch']),
        'request':''.join('\t'.join(r)+'\n' for r in plan['patch']),'ref':'','hash':'','applied':''}
    for aa,tt,dd,mode in plan['run']:
        arm=int(aa);d=torch.load(a.out/f'arm-{arm}.pt',weights_only=True);archive=a.out/'reference-evidence'/f'reference-{arm}.tar.gz'
        if archive.exists():
            h=sha256(archive);e={'archive':archive.name,'sha256':h,'query':replay_archive(archive,h,row_id=arm)}
        else:e=extract(d['logits'],archive,arm)
        after=replay_archive(archive,e['sha256'],row_id=arm)
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('reference replay differs')
        write(a.out/f'reference-{arm}.json',e);facts['ref']+=f"{arm}\t{after['relations']['ref'][0][0]}\n"
        facts['hash']+=f"{arm}\t{hashlib.sha256(d['logits'].numpy().tobytes()).hexdigest()}\n"
        facts['applied']+=''.join(f"{arm}\t{o['layer']}\t{o['position']}\t{o['head']}\t{o['dimension']}\t{o['before']:.17g}\t{o['after']:.17g}\n" for o in d['applied'])
    m=ROOT/'dl/qwen35_attention_exchange_summary.dl'
    x=check(m,m,facts,{},evidence_dir=a.out/'certificate-evidence/summary',provenance={'source':'head donor transfer and matched controls, known development; not extracted circuit equivalence'})
    write(a.out/'summary.json',x)
    for before in (read(a.out/'payload.json'),x):
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    write(a.out/'replay-audit.json',{'reference_witnesses':len(plan['run']),'queries':2,'unchanged':True,'summary_certified':x['certified']})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('valid',x['certified'],'counts',x['relations']['counts'],flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();{'inputs':inputs,'run':run,'evaluate':evaluate}[a.phase](a)
if __name__=='__main__':main()
