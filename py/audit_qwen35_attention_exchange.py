"""Model-free replay of head exchange payloads, global references and controls."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_attention_exchange import ROOT,OUT,checked,read,write
from qwen35_reference_evidence import replay_archive

def main():
    plan=checked(type('Args',(),{'out':OUT})())
    for n,h in read(OUT/'closed.json')['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('closed exchange evidence changed')
    for arm,target,donor,mode in plan['run']:
        before=read(OUT/f'reference-{arm}.json');after=replay_archive(OUT/'reference-evidence'/before['archive'],before['sha256'],row_id=int(arm))
        if not after['certified'] or before['query']['relations']!=after['relations']:raise ValueError('reference replay differs')
    for name in ('payload','summary'):
        before=read(OUT/f'{name}.json');after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    result={'closed_sha256':sha256(OUT/'closed.json'),'reference_witnesses':len(plan['run']),'queries':2,'unchanged':True,'summary_certified':read(OUT/'summary.json')['certified'],'audit_source_sha256':sha256(ROOT/'py/audit_qwen35_attention_exchange.py')}
    write(ROOT/'reference/benchmarks/qwen35_attention_exchange_replay.json',result);print(result)
if __name__=='__main__':main()
