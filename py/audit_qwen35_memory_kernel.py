"""Model-free replay of the source-conditioned memory diagnostic."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_memory_kernel import ROOT,OUT,checked,read,write
from qwen35_reference_evidence import replay_archive

def main():
    checked(type('Args',(),{'out':OUT})())
    for n,h in read(OUT/'closed.json')['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('closed evidence changed')
    for r in read(OUT/'probes.json'):
        e=read(OUT/f"reference-{r['id']}.json");after=replay_archive(OUT/'reference-evidence'/e['archive'],e['sha256'],row_id=r['id'])
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('reference replay differs')
    for name in ('plan','summary'):
        before=read(OUT/f'{name}.json');after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    result={'closed_sha256':sha256(OUT/'closed.json'),'reference_witnesses':len(read(OUT/'probes.json')),'queries':2,'unchanged':True,'summary_certified':read(OUT/'summary.json')['certified'],'audit_source_sha256':sha256(ROOT/'py/audit_qwen35_memory_kernel.py')}
    write(ROOT/'reference/benchmarks/qwen35_memory_kernel_replay.json',result);print(result)
if __name__=='__main__':main()
