"""Retain all development verdicts; passing finite proofs do not imply admission."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from certificate import check,replay,sha256
from run_copy_prior_guard import ROOT,OUT,OPTIONS,gate,journal,read,write
from copy_prior_guard_artifact import candidate,emit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();a.out=a.out.resolve();a.binary=a.bundle=None
    rows=gate(a);observed=journal(a.out/'observations.jsonl')
    domains=read(a.out/'development-domains.json')
    if replay(Path(domains['evidence']))['relations']!=domains['relations']:raise ValueError('domains changed')
    lexical=(a.out/'lexical.dl').read_text()
    reports=[]
    for option in OPTIONS:
        c,k,s=option
        directory=a.out/'proposals'/str(c)
        circuit=emit(directory,candidate(option,lexical))
        ids={int(i) for opt,i,_ in domains['relations']['firing'] if int(opt)==c}
        facts={'domain':''.join(f'{i}\n' for i in sorted(ids)),
            'tok':''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows if r['id'] in ids for p,t in enumerate(r['ctx'])),
            'ref':''.join(f"{i}\t{observed[i]['response']['next']}\n" for i in sorted(ids) if i in observed)}
        result=check(ROOT/'dl/equiv.dl',circuit,facts,{'ndomain':int,'nmiss':int,'nuncov':int,'nmissing':int,'ninvalid':int},
            evidence_dir=a.out/f'certificate-evidence/development-equiv-{c}',
            provenance={'source':'DEVELOPMENT firing domain, not an admitted final test'})
        reports.append({'option':option,'certificate':result,'artifact_sha256':sha256(circuit)})
        print(c,{key:result.get(key) for key in ('certified','ndomain','nmiss','nuncov')})
        if result['certified']:
            with tempfile.TemporaryDirectory() as scratch:
                temp=Path(scratch);(temp/'facts').mkdir();(temp/'outputs').mkdir()
                for n in ('circuits.dl','run.dl'):shutil.copyfile(directory/n,temp/n)
                (temp/'facts/tok.facts').write_text(''.join(f"{r['id']}\t{p}\t{t}\n" for r in rows for p,t in enumerate(r['ctx'])))
                (temp/'facts/domain.facts').write_text(''.join(f"{r['id']}\n" for r in rows))
                proc=subprocess.run(['souffle','run.dl','-F','facts','-D','outputs'],cwd=temp,capture_output=True,text=True)
                if proc.returncode:raise ValueError(proc.stderr)
                expected=''.join(f'{i}\t{t}\n' for opt,i,t in domains['relations']['firing'] if int(opt)==c)
                staged={'domain':(temp/'facts/domain.facts').read_text(),'expected_decision':expected,
                    'actual_decision':(temp/'outputs/cdecide.csv').read_text(),
                    'actual_abstain':(temp/'outputs/abstain.csv').read_text()}
                checker=ROOT/'dl/copy_prior_guard_runtime_check.dl'
                runtime=check(checker,checker,staged,{},evidence_dir=a.out/f'certificate-evidence/runtime-{c}',
                    provenance={'source':'relocated Souffle-only artifact versus pre-reference frozen predictions and abstentions'})
                reports[-1]['runtime']=runtime
                reports[-1]['runtime_files']=sorted(str(p.relative_to(temp)) for p in temp.rglob('*') if p.is_file())
                if not runtime['certified']:raise ValueError('standalone runtime changed')
    write(a.out/'development-certificates.json',{'scope':'finite development proofs; admission gates still apply','candidates':reports})


if __name__=='__main__':main()
