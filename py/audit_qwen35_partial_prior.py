"""Replay the mechanism-derived training predictor without model/checkpoint access."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_partial_prior import ROOT,OUT,checked,read,write

def main():
    checked(type('Args',(),{'out':OUT})())
    for n,h in read(OUT/'closed.json')['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('closed evidence changed')
    for name in ('kernel','summary','counts'):
        before=read(OUT/f'{name}.json');after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    result={'closed_sha256':sha256(OUT/'closed.json'),'queries':3,'new_forwards':0,'unchanged':True,'audit_source_sha256':sha256(ROOT/'py/audit_qwen35_partial_prior.py')}
    write(ROOT/'reference/benchmarks/qwen35_partial_prior_replay.json',result);print(result)
if __name__=='__main__':main()
