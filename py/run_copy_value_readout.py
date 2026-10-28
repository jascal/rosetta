"""Build-time OV projections; Datalog computes categorical winners and checks."""
import argparse
import gzip
from pathlib import Path
import shutil
import subprocess
import tempfile

from certificate import check,replay,sha256
from run_copy_source_value_windows import ROOT,OUT as WINDOWS,read,write

OUT=ROOT/'reference/benchmarks/qwen25_copy_value_readout'
SOURCES=('py/run_copy_value_readout.py','py/certificate.py','py/run_copy_mechanism.py',
    'dl/copy_value_readout_plan.dl','dl/copy_value_readout_decode.dl','dl/copy_value_readout_runtime.dl','docs/copy-value-readout-plan.md')


def inputs(a):
    a.out.mkdir(parents=True,exist_ok=False)
    old=read(WINDOWS/'plan.json')['relations'];rows=read(WINDOWS/'dataset.json');hist=read(WINDOWS/'historical.json')
    module=ROOT/'dl/copy_value_readout_plan.dl'
    result=check(module,module,{
        'row':''.join(f"{r['id']}\t{r['expected']}\t{hist[str(r['id'])]['next']}\n" for r in rows),
        'window':''.join(f'{i}\t{j}\t{t}\n' for i,p,k,j,t in old['window_tok'] if int(k)==8),
        'base_member':''.join(f'{s}\t{l}\t{h}\n' for s,l,h in read(WINDOWS/'members.json'))},{},
        evidence_dir=a.out/'certificate-evidence/plan',provenance={'source':'metadata-only window deduplication; no numeric scores'})
    write(a.out/'plan.json',result)
    if not result['certified']:raise ValueError(result.get('error','invalid plan'))
    previous=read(WINDOWS/'frozen.json')
    write(a.out/'frozen.json',{'scope':'component categorical readout discovery, not model equivalence',
        'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()},
        'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'previous':{str(p.relative_to(ROOT)):sha256(p) for p in
            (WINDOWS/'closed.json',WINDOWS/'plan.json',WINDOWS/'lexical-values.pt',WINDOWS/'upstream-build.json',WINDOWS/'dataset.json',WINDOWS/'historical.json')},
        'checkpoint':previous['checkpoint'],'checkpoint_sha256':previous['checkpoint_sha256']})
    print('frozen:',len(result['relations']['feature']),'unique source windows;',len(result['relations']['head_set']),'head sets')


def gate(a):
    frozen=read(a.out/'frozen.json')
    for n,h in frozen['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'frozen input changed: {n}')
    for n,h in (frozen['sources']|frozen['previous']).items():
        if sha256(ROOT/n)!=h:raise ValueError(f'frozen source changed: {n}')
    return frozen,read(a.out/'plan.json')['relations']


def extract(a):
    import torch
    import transformers
    frozen,plan=gate(a);checkpoint=Path(frozen['checkpoint'])
    for n,h in frozen['checkpoint_sha256'].items():
        if sha256(checkpoint/n)!=h:raise ValueError('checkpoint changed')
    torch.set_num_threads(2)
    model=transformers.AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float32,
        attn_implementation='eager',local_files_only=True).eval()
    values=torch.load(WINDOWS/'lexical-values.pt',map_location='cpu',weights_only=True)
    build=read(WINDOWS/'upstream-build.json');dim=build['head_dim'];groups=build['num_key_value_groups']
    ids=[int(i) for i, in plan['feature']];vectors={};digests={}
    with torch.no_grad():
        for name, in plan['head_set']:
            path=a.out/f'scores-{name}.facts.gz'
            if path.exists():raise ValueError('refusing to overwrite extracted scores')
            residuals=[]
            for i in ids:
                residual=torch.zeros(model.config.hidden_size,dtype=torch.float32)
                for arm,l,h in plan['member']:
                    if arm!=name:continue
                    l,h=int(l),int(h)
                    v=values[i,12,8][l][h//groups]
                    residual+=model.model.layers[l].self_attn.o_proj.weight[:,h*dim:(h+1)*dim] @ v
                residuals.append(residual*model.model.norm.weight)
            vector=torch.stack(residuals);vectors[name]=vector
            scores=vector @ model.lm_head.weight.T
            with gzip.open(path,'wt') as stream:
                for i,score in zip(ids,scores.tolist()):
                    stream.writelines(f'{i}\t{t}\t{x:.17g}\n' for t,x in enumerate(score))
            digests[path.name]=sha256(path)
            print('extracted',name,tuple(scores.shape),flush=True)
        torch.save({'ids':ids,'vectors':vectors},a.out/'component-vectors.pt')
    write(a.out/'extracted.json',{'files':digests|{'component-vectors.pt':sha256(a.out/'component-vectors.pt')},
        'vocab_size':model.config.vocab_size,'torch':torch.__version__,'transformers':transformers.__version__,
        'precision':'float32','normalization':'learned final norm gain; common positive inverse RMS omitted',
        'head_dim':dim,'num_key_value_groups':groups})


def decode(a):
    _,plan=gate(a);extracted=read(a.out/'extracted.json')
    for n,h in extracted['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'extraction changed: {n}')
    results={};facts=[]
    for name, in plan['head_set']:
        with gzip.open(a.out/f'scores-{name}.facts.gz','rt') as stream:scores=stream.read()
        module=ROOT/'dl/copy_value_readout_decode.dl'
        result=check(module,module,{'feature':''.join(f'{i}\n' for i, in plan['feature']),
            'sample':''.join(f'{i}\t{f}\t{g}\t{t}\n' for i,f,g,t in plan['sample']),
            'vocab':f"{extracted['vocab_size']}\n",'score':scores},{},
            evidence_dir=a.out/f'certificate-evidence/decode-{name}',
            provenance={'source':'complete-vocabulary component readout; no full-model equivalence claim'})
        results[name]=result
        if not result['certified']:raise ValueError(result.get('error','incomplete component projection'))
        for f,t in result['relations']['mapped']:
            window=[int(x) for j,x in sorted((int(j),x) for feature,j,x in plan['feature_tok'] if feature==f)]
            facts.append(f'value_emit("{name}",'+','.join(map(str,window+[int(t)]))+').\n')
        print(name,'counts:',result['relations']['counts'],'map:',result['relations']['mapped'],flush=True)
    write(a.out/'readouts.json',results)
    directory=a.out/'artifact';directory.mkdir(exist_ok=False)
    (directory/'mapping.dl').write_text('.decl value_emit(a:symbol,'+','.join(f't{j}:number' for j in range(8))+',out:number)\n'+''.join(facts))
    variables=','.join(f'T{j}' for j in range(8))
    (directory/'run.dl').write_text('.decl source_window(i:number,'+','.join(f't{j}:number' for j in range(8))+')\n.input source_window\n#include "mapping.dl"\n'
        '.decl readout(i:number,a:symbol,t:number)\n.output readout\n'+
        f'readout(I,A,T) :- source_window(I,{variables}), value_emit(A,{variables},T).\n')
    with tempfile.TemporaryDirectory() as scratch:
        temp=Path(scratch);(temp/'facts').mkdir();(temp/'outputs').mkdir()
        for name in ('mapping.dl','run.dl'):shutil.copyfile(directory/name,temp/name)
        staged=[]
        for i,f,g,t in plan['sample']:
            window=[int(x) for j,x in sorted((int(j),x) for feature,j,x in plan['feature_tok'] if feature==f)]
            staged.append('\t'.join([i]+list(map(str,window)))+'\n')
        (temp/'facts/source_window.facts').write_text(''.join(staged))
        proc=subprocess.run(['souffle','run.dl','-F','facts','-D','outputs'],cwd=temp,capture_output=True,text=True)
        if proc.returncode:raise ValueError(proc.stderr)
        checker=ROOT/'dl/copy_value_readout_runtime.dl'
        runtime=check(checker,checker,{'expected':''.join(f'{i}\t{name}\t{t}\n' for name,result in results.items()
            for i,t in result['relations']['decision']),'actual':(temp/'outputs/readout.csv').read_text()},{},
            evidence_dir=a.out/'certificate-evidence/runtime',provenance={'source':'relocated categorical mapping only; no model-equivalence claim'})
        write(a.out/'runtime.json',runtime)
        if not runtime['certified']:raise ValueError('serialized component map differs')
    checks=[]
    for result in (read(a.out/'plan.json'),*results.values(),runtime):
        after=replay(Path(result['evidence']))
        if result['certified']!=after['certified'] or result['relations']!=after['relations']:raise ValueError('replay changed')
        checks.append(result['evidence'])
    write(a.out/'replay-audit.json',{'queries':checks,'unchanged':True})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','extract','decode'))
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();a.out=a.out.resolve();globals()[a.phase](a)


if __name__=='__main__':main()
