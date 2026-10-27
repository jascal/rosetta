"""Supplement frozen raw diagnostic relations with explicit per-instance counts."""
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_query_prior import ROOT,OUT as PRIOR,read,write
OUT=ROOT/'reference/benchmarks/qwen35_query_prior_counts'
def main():
    OUT.mkdir(exist_ok=False);raw=read(PRIOR/'summary.json')['relations']
    facts={'query':read(PRIOR/'facts.json')['query']}
    for n in ('preference','guarded','error'):facts[n]=''.join('\t'.join(r)+'\n' for r in raw[n])
    facts['disagreement']=''.join(f'{q}\t{i}\n' for q,i,c in raw['winner_agreement'] if c=='prior_differs_model')
    m=ROOT/'dl/qwen35_query_prior_counts.dl'
    x=check(m,m,facts,{},evidence_dir=OUT/'certificate-evidence',provenance={'source':'per-instance recount; frozen summary aggregates counted query IDs through existential anonymous joins; raw decisions unchanged'})
    write(OUT/'counts.json',x)
    if not x['certified'] or replay(Path(x['evidence']))['relations']!=x['relations']:raise ValueError('count replay differs')
    write(OUT/'closed.json',{'prior_closed_sha256':sha256(PRIOR/'closed.json'),'files':{str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print(x['relations'])
if __name__=='__main__':main()
