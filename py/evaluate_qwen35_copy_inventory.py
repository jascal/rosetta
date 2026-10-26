"""Seal train-only baseline proposals, then retain held-out admission/certificates."""
import argparse
from pathlib import Path
from certificate import check,replay,sha256
from run_qwen35_copy_inventory import ROOT,OUT,gate,journal,proposal_gate,read,write,emit
from qwen35_reference_evidence import replay_archive


def audit_record(out,row):
    e=row['reference'];after=replay_archive(out/'reference-evidence'/e['archive'],e['sha256'],row_id=row['id'])
    if not after['certified'] or after['relations']!=e['query']['relations']:
        raise ValueError('reference witness differs')
    if after['relations']['ref']!=[[str(row['next'])]]:raise ValueError('journal differs from Datalog argmax')


def seal(a):
    rows=gate(a);observed=journal(a.out);proposal_gate(a.out)
    required={r['id'] for r in rows if r['split']=='train'}
    if set(observed)!=required:raise ValueError('sealing needs all and only training references')
    for n,row in enumerate(observed.values(),1):
        audit_record(a.out,row)
        if n%10==0:print('train witnesses replayed',n,'/',len(required),flush=True)
    proposal=read(a.out/'proposals.json');again=replay(Path(proposal['evidence']))
    if proposal['relations']!=again['relations'] or not again['certified']:raise ValueError('proposal replay differs')
    write(a.out/'training-references.json',observed)
    files=[a.out/'training-references.json',a.out/'proposals-frozen.json',a.out/'proposals.json',
        *(a.out/'proposals').rglob('*'),*Path(proposal['evidence']).rglob('*'),
        *(a.out/'reference-evidence').glob('reference-*.tar.gz')]
    sources=('py/evaluate_qwen35_copy_inventory.py','dl/qwen35_baseline_domain.dl','dl/equiv.dl')
    write(a.out/'validation-ready.json',{'protocol_sha256':sha256(a.out/'protocol.json'),
        'files':{str(p.relative_to(a.out)):sha256(p) for p in files if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in sources},'replayed_training_witnesses':len(required)})
    print('validation ready; no validation references observed',flush=True)


def ready(a):
    gate(a);proposal_gate(a.out);frozen=read(a.out/'validation-ready.json')
    if sha256(a.out/'protocol.json')!=frozen['protocol_sha256']:raise ValueError('protocol changed')
    for n,h in frozen['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'train/proposal evidence changed: {n}')
    for n,h in frozen['sources'].items():
        if sha256(ROOT/n)!=h:raise ValueError(f'evaluation source changed: {n}')
    observed=journal(a.out)
    for key,row in read(a.out/'training-references.json').items():
        if observed.get(int(key))!=row:raise ValueError('training journal changed')
    print('validation seal intact',flush=True)


def admit(a):
    ready(a);rows=gate(a);observed=journal(a.out);pairs=read(a.out/'pairs.json')
    proposal=read(a.out/'proposals.json')['relations']['proposed']
    facts={'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])),
        'sample':''.join(f"{r['id']}\t{r['split']}\t{r['group']}\n" for r in rows),
        'ref':''.join(f"{i}\t{r['next']}\n" for i,r in observed.items()),
        'proposed':''.join(f'{r}\t{c}\n' for r,c in proposal),
        'pair':''.join(f"{p['a']}\t{p['b']}\t{p['group']}\n" for p in pairs),
        'edit':''.join(f"{p['a']}\t{p['b']}\t{j}\n" for p in pairs for j in p['positions']),
        'query_start':''.join(f"{r['id']}\t{len(r['ctx'])-r['prefix']}\n" for r in rows)}
    result=check(ROOT/'dl/qwen35_baseline_admit.dl',a.out/'hypotheses.dl',facts,{},
        evidence_dir=a.out/'certificate-evidence/admit',provenance={'source':'fixed proposals, held-out validation admission; development only'})
    write(a.out/'admission.json',result)
    if not result['certified']:raise ValueError(result.get('error','incomplete or invalid observations'))
    circuit=emit(a.out/'behavior',[(int(r),int(c)) for r,c in result['relations']['admitted']])
    domains=check(ROOT/'dl/qwen35_baseline_domain.dl',circuit,{'tok':facts['tok'],
        'domain':''.join(f"{r['id']}\n" for r in rows)},{},evidence_dir=a.out/'certificate-evidence/baseline-domains',
        provenance={'source':'emitted development baseline decisions and abstentions'})
    write(a.out/'baseline-domains.json',domains)
    ids={int(i) for i,t in domains['relations']['firing']}
    evidence=check(ROOT/'dl/equiv.dl',circuit,{'domain':''.join(f'{i}\n' for i in sorted(ids)),
        'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows if r['id'] in ids for p,t in enumerate(r['ctx'])),
        'ref':''.join(f"{i}\t{observed[i]['next']}\n" for i in sorted(ids) if i in observed)},
        {'ndomain':int,'nmiss':int,'nuncov':int,'nmissing':int,'ninvalid':int},
        evidence_dir=a.out/'certificate-evidence/baseline-development-equiv',
        provenance={'source':'finite baseline development proof; not a fresh extracted-circuit confirmation'})
    write(a.out/'baseline-certificate.json',evidence)
    for row in observed.values():audit_record(a.out,row)
    for before in (read(a.out/'input-integrity.json'),read(a.out/'proposals.json'),result,domains,evidence):
        after=replay(Path(before['evidence']))
        if before['certified']!=after['certified'] or before['relations']!=after['relations']:raise ValueError('query replay differs')
    write(a.out/'replay-audit.json',{'reference_witnesses':len(observed),'baseline_queries':5,'unchanged':True})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('admitted',result['relations']['admitted'],'baseline finite development certificate',
        {k:evidence.get(k) for k in ('certified','ndomain','nmiss','nuncov')},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('seal','ready','admit'))
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();a.out=a.out.resolve();globals()[a.phase](a)


if __name__=='__main__':main()
