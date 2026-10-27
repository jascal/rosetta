"""Audit every retained product/request/applied site without changing the frozen run."""
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_routing_mlp import ROOT,OUT,checked,read,write

def main():
    import torch
    a=type('Args',(),{'out':OUT})();p,plan=checked(a)
    rows=read(OUT/'contexts.json')
    facts={'shape':f"{p['width']}\n",'layer':''.join(f'{l}\n' for l in p['layers']),
        'context':''.join(f"{r['id']}\t{len(r['ctx'])-1}\n" for r in rows),
        'run':''.join('\t'.join(r)+'\n' for r in plan['run']),
        'observation':'','request':'','applied':''}
    for arm,target,donor,layer,mode in plan['run']:
        data=torch.load(OUT/f'arm-{arm}.pt',weights_only=True)
        if mode=='baseline':
            facts['observation']+=''.join(f"{arm}\t{r['layer']}\t{r['neuron']}\t{r['product']:.17g}\n" for r in data['observations'])
        else:
            payload=read(OUT/f'payload-{arm}.json')
            facts['request']+=''.join('\t'.join(r)+'\n' for r in payload['relations']['patch'])
        facts['applied']+=''.join(f"{arm}\t{r['layer']}\t{r['position']}\t{r['neuron']}\t{r['before']:.17g}\t{r['after']:.17g}\n" for r in data['applied'])
    m=ROOT/'dl/qwen35_routing_integrity.dl'
    x=check(m,m,facts,{},evidence_dir=OUT/'certificate-evidence/site-integrity',provenance={'source':'supplemental complete-width observation/payload/application audit; unchanged original scientific plan'})
    write(OUT/'site-integrity.json',x)
    again=replay(Path(x['evidence']))
    if not x['certified'] or again['relations']!=x['relations']:raise ValueError('site integrity failed')
    write(OUT/'site-integrity-audit.json',{'replay_unchanged':True,'sources':{n:sha256(ROOT/n) for n in ('py/audit_qwen35_routing_integrity.py','dl/qwen35_routing_integrity.dl')}})
    print('all baseline observations and requested/applied sites verified')
if __name__=='__main__':main()
