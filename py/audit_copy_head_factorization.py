"""Replay factorization evidence without a model; optionally recompute tensor identities."""
import argparse
from pathlib import Path
from certificate import replay,sha256
from run_copy_head_factorization import OUT,gate,read,factors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--tensors',action='store_true',help='requires torch, but no checkpoint or inference')
    a=parser.parse_args();a.out=a.out.resolve();_,rows=gate(a)
    for name,digest in read(a.out/'closed.json')['files'].items():
        if sha256(a.out/name)!=digest:raise ValueError(f'closed evidence changed: {name}')
    count=0
    for path in sorted((a.out/'certificate-evidence').rglob('certificate.json')):
        before=read(path)['result'];after=replay(path.parent)
        recorded={p.stem:[line.split('\t') for line in p.read_text().splitlines()]
            for p in (path.parent/'outputs').glob('*.csv')}
        if before['certified']!=after['certified'] or before.get('error')!=after.get('error') or recorded!=after.get('relations',{}):
            raise ValueError(f'replay changed: {path}')
        count+=1
    if a.tensors:
        import torch
        torch.set_num_threads(2)
        cache=torch.load(a.out/'captured-factors.pt',map_location='cpu',weights_only=True)
        build=read(a.out/'upstream-build.json');dim=build['head_dim'];groups=build['num_key_value_groups']
        for i,l,h,error in read(a.out/'reconstruction.json'):
            v=factors(cache[i],cache[i],l,h,dim,groups,'identity',[])
            actual=float((v-cache[i][l]['product'][h*dim:(h+1)*dim]).abs().max())
            if actual!=error:raise ValueError(f'reconstruction changed: {i}/{l}/{h}')
        print('Saved-tensor reconstruction errors reproduce exactly; no checkpoint loaded')
    print(count,'queries replay unchanged; frozen and closed hashes unchanged')


if __name__=='__main__':main()
