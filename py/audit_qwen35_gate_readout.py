"""Replay readout arithmetic and inherited singleton argmax evidence without weights."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_gate_readout import ROOT,OUT,PRIOR,checked,read,write
from qwen35_reference_evidence import replay_archive

def main():
    checked(type('Args',(),{'out':OUT})())
    for n,h in read(OUT/'closed.json')['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('closed evidence changed')
    for name in ('keys','readout'):
        before=read(OUT/f'{name}.json');after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('readout replay differs')
    witnesses=0
    for arm,target,donor,mode in read(PRIOR/'payload.json')['relations']['run']:
        if mode=='joint':continue
        e=read(PRIOR/f'reference-{arm}.json');after=replay_archive(PRIOR/'reference-evidence'/e['archive'],e['sha256'],row_id=int(arm))
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('inherited singleton reference differs')
        witnesses+=1
    report={'closed_sha256':sha256(OUT/'closed.json'),'queries':2,'inherited_reference_witnesses':witnesses,'new_forwards':0,'unchanged':True,'audit_source_sha256':sha256(ROOT/'py/audit_qwen35_gate_readout.py')}
    write(ROOT/'reference/benchmarks/qwen35_gate_readout_replay.json',report);print(report)
if __name__=='__main__':main()
