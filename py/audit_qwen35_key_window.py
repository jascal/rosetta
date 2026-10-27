"""Replay the one bounded window test, including retained negative comparisons."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_key_window import ROOT,OUT,checked,read,write,require

def main():
    a=type('Args',(),{'out':OUT})();checked(a,(OUT/'payload-frozen.json').exists())
    for n,h in read(OUT/'closed.json')['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('closed window evidence changed')
    queries=0
    for p in OUT.glob('*.json'):
        if p.name in ('protocol.json','outcome.json','closed.json','payload-frozen.json','replay-audit.json'):continue
        before=read(p)
        if not isinstance(before,dict) or 'evidence' not in before:continue
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('window query replay differs')
        if 'key-match-' in p.name:require(before,('invalid','bad','result','counts','certified'),('result','counts'),False)
        queries+=1
    witnesses=0
    if (OUT/'reference-evidence').exists():
        from qwen35_reference_evidence import replay_archive
        for p in OUT.glob('reference-*.json'):
            e=read(p)
            if not isinstance(e,dict) or 'archive' not in e:continue
            arm=int(p.stem.split('-')[-1]);after=replay_archive(OUT/'reference-evidence'/e['archive'],e['sha256'],row_id=arm)
            if not after['certified'] or after['relations']!=e['query']['relations']:raise ValueError('argmax witness replay differs')
            witnesses+=1
    report={'closed_sha256':sha256(OUT/'closed.json'),'queries':queries,'reference_witnesses':witnesses,'unchanged':True,'outcome':read(OUT/'outcome.json'),'model_coverage_certificate':False,'audit_source_sha256':sha256(ROOT/'py/audit_qwen35_key_window.py')}
    write(ROOT/'reference/benchmarks/qwen35_key_window_replay.json',report);print({'queries':queries,'reference_witnesses':witnesses,'unchanged':True,'model_coverage_certificate':False})
if __name__=='__main__':main()
