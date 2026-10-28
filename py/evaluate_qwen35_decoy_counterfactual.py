"""Retain complete development decoy-edit results and model-free replay."""
from pathlib import Path
from run_qwen35_decoy_counterfactual import OUT,ROOT,INVENTORY,read,write,journal
from certificate import check,replay,sha256
from qwen35_reference_evidence import replay_archive

def main():
    protocol=read(OUT/'protocol.json')
    for n,h in protocol['files'].items():
        if sha256(OUT/n)!=h:raise ValueError('frozen inputs changed')
    for n,h in protocol['sources'].items():
        if sha256(ROOT/n)!=h:raise ValueError('frozen source changed')
    if sha256(INVENTORY/'closed.json')!=protocol['inventory_closed_sha256']:raise ValueError('inventory changed')
    rows=read(OUT/'dataset.json');obs=journal(OUT)
    if set(obs)!={r['id'] for r in rows}:raise ValueError('collection incomplete')
    for i,r in obs.items():
        e=r['reference'];again=replay_archive(OUT/'reference-evidence'/e['archive'],e['sha256'],row_id=i)
        if not again['certified'] or again['relations']!=e['query']['relations'] or again['relations']['ref']!=[[str(r['next'])]]:raise ValueError('reference replay differs')
    module=ROOT/'dl/qwen35_decoy_summary.dl'
    x=check(module,module,{'planned':''.join(f"{r['id']}\t{r['pair_a']}\t{r['long_target']}\t{r['old_decoy']}\t{r['new_decoy']}\t{r['base_reference']}\n" for r in rows),
        'ref':''.join(f"{i}\t{r['next']}\n" for i,r in obs.items())},{},evidence_dir=OUT/'certificate-evidence/summary',provenance={'source':'development-only paired decoy edit effects, not an extracted circuit certificate'})
    write(OUT/'summary.json',x)
    for before in (read(OUT/'plan.json'),read(OUT/'edit-integrity.json'),x):
        again=replay(Path(before['evidence']))
        if again['certified']!=before['certified'] or again['relations']!=before['relations']:raise ValueError('query replay differs')
    write(OUT/'replay-audit.json',{'reference_witnesses':len(obs),'queries':3,'unchanged':True})
    write(OUT/'closed.json',{'files':{str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('outcomes',x['relations']['counts'],'effects',x['relations']['effects'],flush=True)
if __name__=='__main__':main()
