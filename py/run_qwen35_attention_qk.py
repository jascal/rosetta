"""Frozen query/key routing factorization, reusing the validated substitution engine."""
import argparse,hashlib
from pathlib import Path
from certificate import check,replay,sha256
import run_qwen35_attention_factor as engine
from run_qwen35_copy_inventory import ROOT,MODEL,OUT as INVENTORY,gate,read,write
OUT=ROOT/'reference/benchmarks/qwen35_attention_qk'
PRIOR=engine.OUT
SOURCES=('py/run_qwen35_attention_qk.py','py/run_qwen35_attention_factor.py','py/qwen35_attention_qk_reference.py','py/qwen35_attention_factor_reference.py','py/qwen35_attention_reference.py','py/qwen35_lazy_reference.py','py/qwen35_stream_reference.py','dl/qwen35_attention_qk_payload.dl','dl/copy_longest_runtime.dl','dl/qwen35_attention_factor_summary.dl','dl/qwen35_factor_reconstruction_linear.dl','dl/qwen35_gate_cross_reproduction.dl','py/qwen35_reference_evidence.py','dl/qwen35_reference_argmax.dl','py/certificate.py')
def inputs(a):
    for n,h in read(PRIOR/'closed.json')['files'].items():
        if sha256(PRIOR/n)!=h:raise ValueError('prior changed')
    report=read(ROOT/'reference/benchmarks/qwen35_attention_factor_replay.json')
    if not report['unchanged'] or not report['summary_certified'] or report['closed_sha256']!=sha256(PRIOR/'closed.json'):raise ValueError('prior replay invalid')
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ('contexts.json','baseline-hashes.json'):(a.out/name).write_bytes((PRIOR/name).read_bytes())
    (a.out/'input-core.dl').write_bytes((INVENTORY/'input-core.dl').read_bytes())
    write(a.out/'protocol.json',{'scope':'Q/K routing decomposition on six known contrasts, not final confirmation','capture_forwards':6,'substitution_forwards':36,
        'modes':['self_reconstructed','donor_reconstructed','donor_query','donor_keys','donor_source_keys','donor_short_keys'],
        'fixed':'head27/10, target contextual values and output gate in every mode; aligned positions; actual normalized/rotated Q/K, scale and causal-mask row captured at official eager boundary',
        'source_keys':'successors after suffix matches>=12; long is unique longest successor; short has different successor; other keys stay target',
        'controls':'self route versus original target; full donor route versus prior routing-only substitution; probability/vector tolerances1e-5+1e-5abs;12 whole-score controls .002+.002abs and exact argmax before hybrids',
        'reuse':'validated attention-factor collection/evaluation engine with only reconstruction reference resolver changed; numerical payload and captures freeze before substitutions',
        'priors':{str((p/'closed.json').relative_to(ROOT)):sha256(p/'closed.json') for p in (PRIOR,INVENTORY)},
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},'sources':{n:sha256(ROOT/n) for n in SOURCES}})
    print('frozen six captures plus36 substitutions',flush=True)
def capture(a):
    contexts=engine.checked(a);gate(type('Args',(),{'out':INVENTORY})(),weights=True)
    import torch
    from qwen35_attention_qk_reference import AttentionQKReference
    torch.set_num_threads(2);model=AttentionQKReference(MODEL,vocab_chunk=1024)
    for r in contexts:
        path=a.out/f"capture-{r['id']}.pt"
        if path.exists():continue
        d=model.capture(r['ctx'],[(27,10)]);tmp=path.with_suffix('.pending');torch.save(d,tmp);tmp.replace(path);print('captured',r['id'],flush=True)
def prepare(a):
    contexts=engine.checked(a)
    if (a.out/'payload-frozen.json').exists():engine.checked(a,True);print('payload already frozen');return
    import torch
    facts={'shape':f"{read(MODEL/'config.json')['text_config']['head_dim']}\n",
        'context':''.join(f"{r['id']}\t{r['contrast']}\t{r['side']}\t{len(r['ctx'])-1}\n" for r in contexts),
        'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in contexts for p,t in enumerate(r['ctx'])),
        'qvec':'','kvec':'','scale':'','mask':'','weight':'','value':'','gate':'','control_vector':''}
    hashes={'expected':''.join(f'{i}\t{h}\n' for i,h in read(a.out/'baseline-hashes.json').items()),'actual':''}
    for r in contexts:
        i=r['id'];d=torch.load(a.out/f'capture-{i}.pt',weights_only=True)
        hashes['actual']+=f"{i}\t{hashlib.sha256(d['logits'].numpy().tobytes()).hexdigest()}\n"
        facts['qvec']+=''.join(f"{i}\t{o['dimension']}\t{o['value']:.17g}\n" for o in d['rotated_query'])
        facts['kvec']+=''.join(f"{i}\t{o['source']}\t{o['dimension']}\t{o['value']:.17g}\n" for o in d['rotated_keys'])
        facts['scale']+=''.join(f"{i}\t{o['value']:.17g}\n" for o in d['scale'])
        facts['mask']+=''.join(f"{i}\t{o['source']}\t{o['value']:.17g}\n" for o in d['mask'])
        facts['weight']+=''.join(f"{i}\t{o['source']}\t{o['value']:.17g}\n" for o in d['attention'])
        facts['value']+=''.join(f"{i}\t{o['source']}\t{o['dimension']}\t{o['value']:.17g}\n" for o in d['values'])
        facts['gate']+=''.join(f"{i}\t{o['dimension']}\t{o['value']:.17g}\n" for o in d['raw_gate'])
        facts['control_vector']+=''.join(f"{i}\tself\t{o['dimension']}\t{o['value']:.17g}\n" for o in d['vectors'])
        prior=torch.load(PRIOR/f'arm-{5*i+2}.pt',weights_only=True)
        facts['control_vector']+=''.join(f"{i}\tdonor\t{o['dimension']}\t{o['after']:.17g}\n" for o in prior['applied'])
    m=ROOT/'dl/qwen35_gate_cross_reproduction.dl';x=check(m,m,hashes,{},evidence_dir=a.out/'certificate-evidence/capture-reproduction',provenance={'source':'Q/K capture wrapper preserves full score bytes'})
    write(a.out/'capture-reproduction.json',x)
    if not x['certified']:raise ValueError('capture changes outputs')
    m=ROOT/'dl/qwen35_attention_qk_payload.dl';x=check(m,a.out/'input-core.dl',facts,{},evidence_dir=a.out/'certificate-evidence/payload',provenance={'source':'fixed Q/K and source-position routing factors; Datalog dot products/softmax/vector reconstruction'})
    write(a.out/'payload.json',x)
    if not x['certified']:raise ValueError('Q/K reconstruction failed')
    for name in ('capture-reproduction','payload'):
        b=read(a.out/f'{name}.json');after=replay(Path(b['evidence']))
        if not after['certified'] or after['relations']!=b['relations']:raise ValueError('preparation replay differs')
    write(a.out/'payload-frozen.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='payload-frozen.json'}})
    print('captures exact; Q/K probability and vector reconstruction passed;36 payloads frozen',flush=True)
def reconstruction(a,arm,target,mode,scores):
    import torch
    path=a.out/f'capture-{target}.pt' if mode=='self_reconstructed' else PRIOR/f'arm-{5*target+2}.pt'
    prior=torch.load(path,weights_only=True)['logits'];m=ROOT/'dl/qwen35_factor_reconstruction_linear.dl'
    x=check(m,m,{'vocab':f'{len(scores)}\n','score':''.join(f'{s}\t{t}\t{v:.17g}\n' for s,values in (('original',prior),('reconstructed',scores)) for t,v in enumerate(values.tolist()))},{},evidence_dir=a.out/f'certificate-evidence/reconstruction-{arm}',provenance={'source':'Q/K reconstructed self versus target, or donor route versus retained exact-routing intervention'})
    write(a.out/f'reconstruction-{arm}.json',x)
    if not x['certified']:raise ValueError(f'Q/K reconstruction{arm} failed; stop hybrid interpretation')
    return x
def delegated(a,operation):
    old=engine.reconstruction;engine.reconstruction=reconstruction
    try:getattr(engine,operation)(a)
    finally:engine.reconstruction=old
def run(a):delegated(a,'run')
def evaluate(a):delegated(a,'evaluate')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','capture','prepare','run','evaluate'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();globals()[a.phase](a)
if __name__=='__main__':main()
