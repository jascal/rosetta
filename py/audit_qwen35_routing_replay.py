"""Model-free replay of the closed MLP routing study, including every argmax."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_routing_mlp import ROOT,OUT,checked,read,write
from qwen35_reference_evidence import replay_archive

def audit(out=OUT):
    a=type('Args',(),{'out':out})();protocol,plan=checked(a)
    closed=read(out/'closed.json')
    for n,h in closed['files'].items():
        if sha256(out/n)!=h:raise ValueError(f'closed evidence changed: {n}')
    queries=0
    for name in ('plan','summary','site-integrity','baseline-reproduction'):
        before=read(out/f'{name}.json');after=replay(Path(before['evidence']))
        if not after['certified'] or before['relations']!=after['relations']:raise ValueError(f'query replay differs: {name}')
        queries+=1
    refs=0
    for arm,target,donor,layer,mode in plan['run']:
        if mode!='baseline':
            before=read(out/f'payload-{arm}.json');after=replay(Path(before['evidence']))
            if not after['certified'] or before['relations']!=after['relations']:raise ValueError('payload replay differs')
            queries+=1
        before=read(out/f'reference-{arm}.json')
        after=replay_archive(out/'reference-evidence'/before['archive'],before['sha256'],row_id=int(arm))
        if not after['certified'] or before['query']['relations']!=after['relations']:raise ValueError('argmax replay differs')
        refs+=1
        if refs%10==0:print('replayed',refs,'/',len(plan['run']),flush=True)
    return {'scope':'model-free closed development intervention evidence; not extracted circuit equivalence',
        'closed_sha256':sha256(out/'closed.json'),'reference_witnesses':refs,'other_queries':queries,'unchanged':True,
        'audit_source_sha256':sha256(ROOT/'py/audit_qwen35_routing_replay.py')}
def main():
    result=audit();write(ROOT/'reference/benchmarks/qwen35_routing_mlp_replay.json',result);print(result)
if __name__=='__main__':main()
