"""Frozen routing/value/gate factorization of head27/10, with reconstruction gates."""
import argparse,hashlib
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_attention_exchange import ROOT,OUT as EXCHANGE,read,write
from run_qwen35_attention_profile import OUT as PROFILE
from run_qwen35_copy_inventory import OUT as INVENTORY,MODEL,gate
from qwen35_reference_evidence import extract,replay_archive
OUT=ROOT/'reference/benchmarks/qwen35_attention_factor'
SOURCES=('py/run_qwen35_attention_factor.py','py/qwen35_attention_factor_reference.py','py/qwen35_attention_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','dl/qwen35_attention_factor_payload.dl','dl/qwen35_attention_factor_summary.dl','dl/qwen35_factor_reconstruction_linear.dl','dl/qwen35_gate_cross_reproduction.dl','py/qwen35_reference_evidence.py','dl/qwen35_reference_argmax.dl','py/certificate.py')
CONTROLS=('self_reconstructed','donor_reconstructed')
def inputs(a):
    for root in (EXCHANGE,PROFILE):
        for n,h in read(root/'closed.json')['files'].items():
            if sha256(root/n)!=h:raise ValueError('prior changed')
    report=read(ROOT/'reference/benchmarks/qwen35_attention_exchange_replay.json')
    if not report['unchanged'] or not report['summary_certified'] or report['closed_sha256']!=sha256(EXCHANGE/'closed.json'):raise ValueError('prior replay invalid')
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ('contexts.json','baseline-hashes.json'):(a.out/name).write_bytes((EXCHANGE/name).read_bytes())
    write(a.out/'protocol.json',{'scope':'known development head27/10 factorization; not final circuit confirmation','capture_forwards':6,'substitution_forwards':30,
        'modes':['self_reconstructed','donor_reconstructed','donor_routing','donor_values','donor_gate'],
        'formula':'sigmoid(raw_gate_from) * sum_j attention_from(j)*value_from(j,dimension); stable Datalog double arithmetic, then model FP32 substitution',
        'controls':'capture hashes exact to previous no-op; vector reconstruction1e-5+1e-5abs; twelve full-vocabulary reconstructed controls .002+.002abs and exact argmax must all pass before hybrids',
        'freeze':'policy before captures; full payload and capture hashes before substitutions; all six directions including failed transfers',
        'priors':{str((p/'closed.json').relative_to(ROOT)):sha256(p/'closed.json') for p in (EXCHANGE,PROFILE)},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen six captures plus thirty substitutions',flush=True)
def checked(a,payload=False):
    p=read(a.out/'protocol.json')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError('frozen input changed')
    for n,h in (p['sources']|p['priors']).items():
        if sha256(ROOT/n)!=h:raise ValueError('source/prior changed')
    if payload:
        for n,h in read(a.out/'payload-frozen.json')['files'].items():
            if sha256(a.out/n)!=h:raise ValueError('frozen payload/capture changed')
    return read(a.out/'contexts.json')
def capture(a):
    contexts=checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_attention_factor_reference import AttentionFactorReference
    torch.set_num_threads(2);model=AttentionFactorReference(MODEL,vocab_chunk=1024)
    for r in contexts:
        path=a.out/f"capture-{r['id']}.pt"
        if path.exists():continue
        d=model.capture(r['ctx'],[(27,10)]);tmp=path.with_suffix('.pending');torch.save(d,tmp);tmp.replace(path);print('captured',r['id'],flush=True)
def prepare(a):
    contexts=checked(a)
    if (a.out/'payload-frozen.json').exists():checked(a,True);print('payload already frozen');return
    import torch
    facts={'shape':f"{read(MODEL/'config.json')['text_config']['head_dim']}\n",'context':''.join(f"{r['id']}\t{r['contrast']}\t{r['side']}\t{len(r['ctx'])-1}\n" for r in contexts),
        'weight':'','value':'','gate':'','vector':''}
    hashes={'expected':''.join(f'{i}\t{h}\n' for i,h in read(a.out/'baseline-hashes.json').items()),'actual':''}
    for r in contexts:
        i=r['id'];d=torch.load(a.out/f'capture-{i}.pt',weights_only=True);hashes['actual']+=f"{i}\t{hashlib.sha256(d['logits'].numpy().tobytes()).hexdigest()}\n"
        facts['weight']+=''.join(f"{i}\t{o['source']}\t{o['value']:.17g}\n" for o in d['attention'])
        facts['value']+=''.join(f"{i}\t{o['source']}\t{o['dimension']}\t{o['value']:.17g}\n" for o in d['values'])
        facts['gate']+=''.join(f"{i}\t{o['dimension']}\t{o['value']:.17g}\n" for o in d['raw_gate'])
        facts['vector']+=''.join(f"{i}\t{o['dimension']}\t{o['value']:.17g}\n" for o in d['vectors'])
    m=ROOT/'dl/qwen35_gate_cross_reproduction.dl';x=check(m,m,hashes,{},evidence_dir=a.out/'certificate-evidence/capture-reproduction',provenance={'source':'raw-factor hooks must preserve all prior score bytes'})
    write(a.out/'capture-reproduction.json',x)
    if not x['certified']:raise ValueError('capture changes outputs')
    m=ROOT/'dl/qwen35_attention_factor_payload.dl';x=check(m,m,facts,{},evidence_dir=a.out/'certificate-evidence/payload',provenance={'source':'all fixed routing/value/gate substitutions and scalar reconstructions'})
    write(a.out/'payload.json',x)
    if not x['certified']:raise ValueError('factor reconstruction failed')
    for name in ('capture-reproduction','payload'):
        b=read(a.out/f'{name}.json');r=replay(Path(b['evidence']))
        if not r['certified'] or r['relations']!=b['relations']:raise ValueError('preparation replay differs')
    write(a.out/'payload-frozen.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='payload-frozen.json'}})
    print('all captures exact; scalar reconstruction passed;30 payloads frozen',flush=True)
def reconstruction(a,arm,target,mode,scores):
    import torch
    path=a.out/f'capture-{target}.pt' if mode=='self_reconstructed' else EXCHANGE/f'arm-{10000*target+2710}.pt'
    prior=torch.load(path,weights_only=True)['logits'];m=ROOT/'dl/qwen35_factor_reconstruction_linear.dl'
    x=check(m,m,{'vocab':f'{len(scores)}\n','score':''.join(f'{s}\t{t}\t{v:.17g}\n' for s,values in (('original',prior),('reconstructed',scores)) for t,v in enumerate(values.tolist()))},{},evidence_dir=a.out/f'certificate-evidence/reconstruction-{arm}',provenance={'source':'factor reconstruction versus exact target or exact head-donor substitution'})
    write(a.out/f'reconstruction-{arm}.json',x)
    if not x['certified']:raise ValueError(f'reconstruction{arm} failed; no hybrid interpretation allowed')
    return x
def run(a):
    contexts=checked(a,True);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_attention_reference import AttentionReference
    torch.set_num_threads(2);model=AttentionReference(MODEL,vocab_chunk=1024);contexts={r['id']:r for r in contexts};plan=read(a.out/'payload.json')['relations']
    arms=sorted(plan['run'],key=lambda r:(r[3] not in CONTROLS,int(r[0])))
    for j,(aa,tt,dd,mode) in enumerate(arms):
        arm,target=int(aa),int(tt);path=a.out/f'arm-{arm}.pt'
        if path.exists():d=torch.load(path,weights_only=True)
        else:
            patches=[(int(l),int(p),int(h),int(n),float(v)) for a,l,p,h,n,v in plan['patch'] if a==aa]
            d=model.probe(contexts[target]['ctx'],(),patches);tmp=path.with_suffix('.pending');torch.save(d,tmp);tmp.replace(path);print('completed',j+1,'/',len(arms),arm,mode,flush=True)
        if mode in CONTROLS:reconstruction(a,arm,target,mode,d['logits'])
def evaluate(a):
    contexts=checked(a,True)
    import torch
    plan=read(a.out/'payload.json')['relations'];facts={'run':''.join('\t'.join(r)+'\n' for r in plan['run']),
        'expected':''.join(f"{r['id']}\t{r['expected']}\n" for r in contexts),
        'request':''.join('\t'.join(r)+'\n' for r in plan['patch']),'applied':'','ref':'','reconstruction':''};queries=[]
    for aa,tt,dd,mode in plan['run']:
        arm,target=int(aa),int(tt);d=torch.load(a.out/f'arm-{arm}.pt',weights_only=True);archive=a.out/'reference-evidence'/f'reference-{arm}.tar.gz'
        if archive.exists():
            h=sha256(archive);e={'archive':archive.name,'sha256':h,'query':replay_archive(archive,h,row_id=arm)}
        else:e=extract(d['logits'],archive,arm)
        after=replay_archive(archive,e['sha256'],row_id=arm)
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('argmax replay differs')
        write(a.out/f'reference-{arm}.json',e);facts['ref']+=f"{arm}\t{after['relations']['ref'][0][0]}\n"
        facts['applied']+=''.join(f"{arm}\t{o['layer']}\t{o['position']}\t{o['head']}\t{o['dimension']}\t{o['before']:.17g}\t{o['after']:.17g}\n" for o in d['applied'])
        if mode in CONTROLS:
            x=reconstruction(a,arm,target,mode,d['logits']);queries.append(x);facts['reconstruction']+=f'{arm}\t1\n'
    m=ROOT/'dl/qwen35_attention_factor_summary.dl';x=check(m,m,facts,{},evidence_dir=a.out/'certificate-evidence/summary',provenance={'source':'head factorization after all full-score reconstruction gates; not circuit equivalence'})
    write(a.out/'summary.json',x);queries.extend([read(a.out/'capture-reproduction.json'),read(a.out/'payload.json'),x])
    for b in queries:
        r=replay(Path(b['evidence']))
        if b['certified']!=r['certified'] or b['relations']!=r['relations']:raise ValueError('query replay differs')
    write(a.out/'replay-audit.json',{'reference_witnesses':len(plan['run']),'queries':len(queries),'unchanged':True,'summary_certified':x['certified']})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('valid',x['certified'],'counts',x['relations']['counts'],flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','capture','prepare','run','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();globals()[a.phase](a)
if __name__=='__main__':main()
