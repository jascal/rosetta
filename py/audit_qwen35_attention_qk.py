"""Model-free replay of query/key routing factorization, including scalar/global controls."""
from pathlib import Path
from certificate import replay,sha256
from run_qwen35_attention_qk import ROOT,OUT,read,write
from run_qwen35_attention_factor import CONTROLS
from qwen35_reference_evidence import replay_archive

def checked_history(root, out, repair, window):
    """Audit historical bytes while recognizing only the pinned checker repair."""
    protocol=read(out/'protocol.json')
    receipt=read(repair/'repair-receipt.json')
    pinned=read(window/'protocol.json')['priors']
    name=str((repair/'repair-receipt.json').relative_to(root))
    if pinned.get(name)!=sha256(repair/'repair-receipt.json'):
        raise ValueError('repair receipt changed')
    if receipt['prior_closed_sha256']!=sha256(out/'closed.json'):
        raise ValueError('historical Q/K evidence changed')
    for name,digest in receipt['repaired_sources'].items():
        if sha256(root/name)!=digest:raise ValueError('repaired source changed')
    for name,digest in (protocol['sources']|protocol['priors']).items():
        actual=sha256(root/name)
        if actual==digest:continue
        if name!='dl/qwen35_attention_qk_payload.dl' or actual!=receipt['repaired_sources'].get(name):
            raise ValueError('unregistered source/prior change')
    for manifest in (protocol,read(out/'payload-frozen.json'),read(out/'closed.json')):
        for name,digest in manifest['files'].items():
            if sha256(out/name)!=digest:raise ValueError('frozen Q/K file changed')

def main():
    checked_history(ROOT,OUT,ROOT/'reference/benchmarks/qwen35_key_window_repair',ROOT/'reference/benchmarks/qwen35_key_window')
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
    result={'closed_sha256':sha256(OUT/'closed.json'),'reference_witnesses':len(plan['run']),'queries':len(queries),'unchanged':True,'summary_certified':read(OUT/'summary.json')['certified'],'audit_source_sha256':sha256(ROOT/'py/audit_qwen35_attention_qk.py')}
    write(ROOT/'reference/benchmarks/qwen35_attention_qk_replay.json',result);print(result)
if __name__=='__main__':main()
