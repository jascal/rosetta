"""Model-free replay of head factorization, including scalar/global controls."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_attention_factor import ROOT,OUT,checked,read,write,CONTROLS
from qwen35_reference_evidence import replay_archive

def main():
    checked(type('Args',(),{'out':OUT})(),True)
    for n,h in read(OUT/'closed.json')['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('closed factor evidence changed')
    plan=read(OUT/'payload.json')['relations'];queries=[read(OUT/f'{name}.json') for name in ('capture-reproduction','payload','summary')]
    for arm,target,donor,mode in plan['run']:
        e=read(OUT/f'reference-{arm}.json');after=replay_archive(OUT/'reference-evidence'/e['archive'],e['sha256'],row_id=int(arm))
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('argmax replay differs')
        if mode in CONTROLS:queries.append(read(OUT/f'reconstruction-{arm}.json'))
    for before in queries:
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    result={'closed_sha256':sha256(OUT/'closed.json'),'reference_witnesses':len(plan['run']),'queries':len(queries),'unchanged':True,'summary_certified':read(OUT/'summary.json')['certified'],'audit_source_sha256':sha256(ROOT/'py/audit_qwen35_attention_factor.py')}
    write(ROOT/'reference/benchmarks/qwen35_attention_factor_replay.json',result);print(result)
if __name__=='__main__':main()
