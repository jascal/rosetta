"""Model-free replay of actual4B attention instrument validation."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_attention_validation import ROOT,OUT,checked,read,write

def main():
    checked(type('Args',(),{'out':OUT})())
    for n,h in read(OUT/'closed.json')['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('closed validation evidence changed')
    for name in ('plan','identity','parity','sites'):
        before=read(OUT/f'{name}.json');after=replay(Path(before['evidence']))
        if not after['certified'] or before['relations']!=after['relations']:raise ValueError('validation replay differs')
    result={'closed_sha256':sha256(OUT/'closed.json'),'queries':4,'unchanged':True,'scope':'attention instrument gate, not extraction success','audit_source_sha256':sha256(ROOT/'py/audit_qwen35_attention_validation.py')}
    write(ROOT/'reference/benchmarks/qwen35_attention_validation_replay.json',result);print(result)
if __name__=='__main__':main()
