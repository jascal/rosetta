"""Validate the completed baseline forwards before interpreting patched runs."""
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_routing_mlp import ROOT,OUT,checked,read,write
from qwen35_reference_evidence import extract,replay_archive

def main():
    import torch
    _,plan=checked(type('Args',(),{'out':OUT})());expected={i:t for i,_,_,t,_ in plan['context']}
    facts={'expected':'','ref':''}
    for arm,target,donor,layer,mode in plan['run']:
        if mode!='baseline':continue
        data=torch.load(OUT/f'arm-{arm}.pt',weights_only=True)
        path=OUT/'reference-evidence'/f'reference-{int(arm):04d}.tar.gz'
        if path.exists():
            h=sha256(path);e={'archive':path.name,'sha256':h,'query':replay_archive(path,h,row_id=int(arm))}
        else:e=extract(data['logits'],path,int(arm))
        after=replay_archive(path,e['sha256'],row_id=int(arm))
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('argmax replay changed')
        write(OUT/f'reference-{arm}.json',e)
        facts['expected']+=f'{arm}\t{expected[target]}\n';facts['ref']+=f"{arm}\t{after['relations']['ref'][0][0]}\n"
    m=ROOT/'dl/qwen35_routing_baseline.dl'
    x=check(m,m,facts,{},evidence_dir=OUT/'certificate-evidence/baseline-reproduction',provenance={'source':'new observed baseline argmax versus original retained reference'})
    write(OUT/'baseline-reproduction.json',x)
    if not x['certified'] or replay(Path(x['evidence']))['relations']!=x['relations']:raise ValueError('baseline reproduction failed')
    print('six baseline references reproduced; all six score archives replay unchanged',flush=True)
if __name__=='__main__':main()
