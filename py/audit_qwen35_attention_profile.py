"""Model-free replay of complete attention capture and observational head ranking."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_attention_profile import ROOT,OUT,checked,read,write

def main():
    checked(type('Args',(),{'out':OUT})())
    for n,h in read(OUT/'closed.json')['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('closed profile changed')
    queries=[read(OUT/f'{n}.json') for n in ('plan','reproduction','profile')]+[read(p) for p in OUT.glob('sites-*.json')]
    for before in queries:
        after=replay(Path(before['evidence']))
        if not after['certified'] or before['relations']!=after['relations']:raise ValueError('profile replay differs')
    result={'closed_sha256':sha256(OUT/'closed.json'),'queries':len(queries),'unchanged':True,'scope':'observational profile, not causal head certificate','audit_source_sha256':sha256(ROOT/'py/audit_qwen35_attention_profile.py')}
    write(ROOT/'reference/benchmarks/qwen35_attention_profile_replay.json',result);print(result)
if __name__=='__main__':main()
