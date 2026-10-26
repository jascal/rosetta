"""Lossless per-reference Datalog evidence archives; no model required to replay."""
import gzip
from pathlib import Path
import tarfile
import tempfile
from certificate import check,replay,sha256
from run_copy_mechanism import ROOT


def extract(scores,path,row_id):
    path=Path(path)
    if path.exists():raise ValueError('refusing to overwrite reference evidence')
    module=ROOT/'dl/qwen35_reference_argmax.dl'
    with tempfile.TemporaryDirectory() as scratch:
        result=check(module,module,{'vocab':f'{len(scores)}\n',
            'score':''.join(f'{t}\t{x:.17g}\n' for t,x in enumerate(scores.tolist()))},{},
            evidence_dir=Path(scratch)/'evidence',provenance={'source':'pinned 4B FP32 reference, complete vocabulary','row':row_id})
        if not result['certified']:raise ValueError(result.get('error','incomplete oracle scores'))
        root=Path(result.pop('evidence'));path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.reference-',delete=False) as raw:
            temporary=Path(raw.name)
        with temporary.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as archive:
            for item in sorted(root.rglob('*')):
                if not item.is_file():continue
                info=archive.gettarinfo(item,arcname=str(item.relative_to(root)))
                info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
                with item.open('rb') as stream:archive.addfile(info,stream)
        temporary.replace(path)
        return {'query':result,'archive':path.name,'sha256':sha256(path)}


def replay_archive(path,digest,row_id=None):
    path=Path(path)
    if sha256(path)!=digest:raise ValueError('reference archive changed')
    with tempfile.TemporaryDirectory() as scratch:
        root=Path(scratch)
        with tarfile.open(path,'r:gz') as archive:archive.extractall(root,filter='data')
        if row_id is not None:
            import json
            if json.loads((root/'certificate.json').read_text())['provenance']['row']!=row_id:
                raise ValueError('reference evidence belongs to another row')
        return replay(root)
