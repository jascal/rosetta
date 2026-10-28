"""Model-free replay of factorization controls, payloads and complete argmaxes."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_mlp_factor import ROOT,OUT,checked,read,write
from qwen35_reference_evidence import replay_archive

def main():
    _,plan=checked(type('Args',(),{'out':OUT})())
    for n,h in read(OUT/'closed.json')['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('closed evidence changed')
    results=[read(OUT/'payload.json'),read(OUT/'summary.json')]
    for aa,tt,dd,mode in plan['run']:
        e=read(OUT/f'reference-{aa}.json');after=replay_archive(OUT/'reference-evidence'/e['archive'],e['sha256'],row_id=int(aa))
        if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('reference replay differs')
        if mode in ('self_reconstructed','donor_reconstructed'):results.append(read(OUT/f'reconstruction-{aa}.json'))
    for before in results:
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    report={'closed_sha256':sha256(OUT/'closed.json'),'reference_witnesses':len(plan['run']),'queries':len(results),'unchanged':True,'summary_certified':read(OUT/'summary.json')['certified'],'audit_source_sha256':sha256(ROOT/'py/audit_qwen35_mlp_factor.py')}
    write(ROOT/'reference/benchmarks/qwen35_mlp_factor_replay.json',report);print(report)
if __name__=='__main__':main()
