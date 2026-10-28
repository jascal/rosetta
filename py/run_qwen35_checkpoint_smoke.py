"""Pinned real-checkpoint parity of two memory-bounded FP32 forward paths."""
import argparse
import hashlib
from pathlib import Path
import resource
import time
from certificate import check,replay,sha256
from run_copy_mechanism import ROOT,read,write

OUT=ROOT/'reference/benchmarks/qwen35_4b_smoke'
MODEL=Path('/tmp/rosetta-qwen35-4b-checkpoint')
REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
SOURCES=('py/run_qwen35_checkpoint_smoke.py','py/qwen35_stream_reference.py','py/qwen35_lazy_reference.py',
    'py/certificate.py','py/run_copy_mechanism.py','dl/qwen35_checkpoint_parity.dl')


def inputs(a):
    from transformers import AutoTokenizer
    a.out.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    texts=['The','alpha beta gamma alpha beta','The']
    rows=[{'id':i,'text':t,'ids':tokenizer.encode(t,add_special_tokens=False)} for i,t in enumerate(texts)]
    write(a.out/'contexts.json',rows)
    files=[p for p in MODEL.iterdir() if p.is_file()]
    write(a.out/'protocol.json',{'scope':'architecture/precision smoke only; no copy generalization or minimized circuit',
        'model':'Qwen/Qwen3.5-4B','revision':REVISION,'checkpoint':str(MODEL),
        'checkpoint_files':{p.name:sha256(p) for p in files},
        'files':{'contexts.json':sha256(a.out/'contexts.json')},'sources':{n:sha256(ROOT/n) for n in SOURCES},
        'precision':'checkpoint tensors widened to FP32 without quantization','threads':2,'cache':False,
        'vocab_chunk':1024,'tolerance':{'absolute':0.002,'relative':0.002},'reset_pair':[0,2]})
    print('frozen inputs:',[(r['id'],r['ids']) for r in rows],flush=True)


def gate(a):
    p=read(a.out/'protocol.json')
    for n,h in p['checkpoint_files'].items():
        if sha256(MODEL/n)!=h:raise ValueError(f'checkpoint changed: {n}')
    for n,h in p['sources'].items():
        if sha256(ROOT/n)!=h:raise ValueError(f'source changed: {n}')
    for n,h in p['files'].items():
        if sha256(a.out/n)!=h:raise ValueError(f'input changed: {n}')
    return read(a.out/'contexts.json')


def run(a):
    import torch
    import transformers
    from qwen35_stream_reference import StreamReference
    from qwen35_lazy_reference import LazyReference
    rows=gate(a);torch.set_num_threads(2)
    ordinary=LazyReference(MODEL,vocab_chunk=1024);streamed=StreamReference(MODEL,vocab_chunk=1024)
    facts={'domain':''.join(f"{r['id']}\n" for r in rows),'vocab':f'{ordinary.config.vocab_size}\n',
        'ordinary':'','streamed':'','reset_pair':'0\t2\n','logits_hash':''};timings=[]
    for row in rows:
        i=row['id']
        for label,model in (('ordinary',ordinary),('streamed',streamed)):
            started=time.monotonic();scores=model.logits(row['ids'])
            elapsed=time.monotonic()-started
            facts[label]+=''.join(f'{i}\t{t}\t{x:.17g}\n' for t,x in enumerate(scores.tolist()))
            digest=hashlib.sha256(scores.numpy().tobytes()).hexdigest()
            facts['logits_hash']+=f'{i}\t{label}\t{digest}\n'
            timings.append({'i':i,'backend':label,'seconds':elapsed})
            print('completed',i,label,f'{elapsed:.2f}s',flush=True)
    module=ROOT/'dl/qwen35_checkpoint_parity.dl'
    result=check(module,module,facts,{},evidence_dir=a.out/'certificate-evidence/checkpoint-parity',
        provenance={'source':'actual pinned 4B checkpoint, official lazy graph versus explicit layer streaming, FP32; not native parity'})
    write(a.out/'parity.json',result)
    write(a.out/'resources.json',{'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'timings':timings,'torch':torch.__version__,'transformers':transformers.__version__,
        'note':'Process max RSS includes mapped file pages and fact staging, not only live private model weights.'})
    again=replay(Path(result['evidence']))
    if result['relations']!=again['relations'] or result['certified']!=again['certified']:raise ValueError('replay differs')
    write(a.out/'replay-audit.json',{'queries':1,'unchanged':True})
    write(a.out/'closed.json',{'files':{str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file() and p.name!='closed.json'}})
    print('parity:',result['certified'],result.get('error'),result.get('relations',{}).get('counts'),
        'reset_bad:',result.get('relations',{}).get('reset_bad'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('inputs','run'))
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();a.out=a.out.resolve();globals()[a.phase](a)


if __name__=='__main__':main()
